"""The Blender representation of a WoW physics rig, shared by import, export
and the M2 Physics panel.

A rig lives in a ``<model>_phys`` collection:

  * a BODY is a mesh object whose origin is the body frame (axis-aligned in
    armature space, like the file). Its settings live in ``obj.m2_phys_body``
    and its mesh is generated from the shape list, so the properties - not the
    mesh - are the source of truth. Bodies collide as the convex hull of that
    mesh, which handles capsules of any orientation.
  * a JOINT is an empty sitting at the joint anchor, oriented like the joint
    frame (Z = primary axis). Its settings live in ``obj.m2_phys_joint`` and
    are mirrored onto a Blender rigid-body constraint for the preview.

Root and kinematic bodies are bone-parented (they follow the skeleton). Dynamic
bodies are free objects; the preview makes their bones follow them.
"""

from __future__ import annotations

import math

import bpy
from bpy.props import (
    BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
    FloatVectorProperty, IntProperty, StringProperty,
)
from bpy.types import PropertyGroup
from mathutils import Matrix, Quaternion, Vector

from . import phys

BUILD = "10"                  # bump on every physics change; shows in reports/logs
COLLECTION_SUFFIX = "_phys"
FOLLOW_CONSTRAINT = "m2phys_follow"

BODY_TYPE_ITEMS = [
    ("ROOT", "Root Anchor", "Follows its bone; chains hang from it. The first anchor in the file"),
    ("DYNAMIC", "Dynamic", "Simulated: swings, falls and drives its bone"),
    ("KINEMATIC", "Anchor / Collider", "Follows its bone; chains can hang from it and dynamic bodies bounce off it"),
]
# In the file, root and kinematic are the same thing: type 0, "follows its
# bone" (retail rigs have several). Type 2 exists in the spec but no retail
# file uses it; the client treats it as a body fixed in WORLD space, which
# leaves anything welded to it behind when the character moves.
_BODY_TYPE_TO_INT = {"ROOT": phys.BODY_ROOT, "DYNAMIC": phys.BODY_DYNAMIC,
                     "KINEMATIC": phys.BODY_ROOT}
_BODY_TYPE_FROM_INT = {phys.BODY_ROOT: "KINEMATIC", phys.BODY_DYNAMIC: "DYNAMIC"}

SHAPE_KIND_ITEMS = [
    ("CAPSULE", "Capsule", "A cylinder with round ends between two points"),
    ("SPHERE", "Sphere", ""),
    ("BOX", "Box", ""),
    ("POLYTOPE", "Polytope", "Convex hull from the file (read-only, kept as is)"),
]
_SHAPE_KIND_TO_INT = {"BOX": phys.SHAPE_BOX, "CAPSULE": phys.SHAPE_CAPSULE,
                      "SPHERE": phys.SHAPE_SPHERE, "POLYTOPE": phys.SHAPE_POLYTOPE}
_SHAPE_KIND_FROM_INT = {v: k for k, v in _SHAPE_KIND_TO_INT.items()}

JOINT_TYPE_ITEMS = [
    ("SHOULDER", "Shoulder (cone + twist)", "Swings inside a cone and twists within limits. The usual choice for cloth, hair and tails"),
    ("SPHERICAL", "Spherical (free swing)", "Pivots freely about the anchor. Chains and pendants"),
    ("WELD", "Weld (rigid / springy)", "Rigid at 0 Hz, springy above. Did NOT work on a player model in testing; prefer Shoulder"),
    ("REVOLUTE", "Revolute (hinge)", "Rotates about the joint's Z axis only"),
    ("PRISMATIC", "Prismatic (slider)", "Slides along the joint's Z axis only"),
    ("DISTANCE", "Distance", "Keeps the two anchors a fixed distance apart"),
]
_JOINT_TYPE_TO_INT = {"SPHERICAL": phys.JOINT_SPHERICAL, "SHOULDER": phys.JOINT_SHOULDER,
                      "WELD": phys.JOINT_WELD, "REVOLUTE": phys.JOINT_REVOLUTE,
                      "PRISMATIC": phys.JOINT_PRISMATIC, "DISTANCE": phys.JOINT_DISTANCE}
_JOINT_TYPE_FROM_INT = {v: k for k, v in _JOINT_TYPE_TO_INT.items()}


def body_type_int(token):
    return _BODY_TYPE_TO_INT.get(token, phys.BODY_DYNAMIC)


def body_type_token(value):
    return _BODY_TYPE_FROM_INT.get(int(value), "KINEMATIC")


def shape_kind_int(token):
    return _SHAPE_KIND_TO_INT.get(token, phys.SHAPE_CAPSULE)


def shape_kind_token(value):
    return _SHAPE_KIND_FROM_INT.get(int(value), "CAPSULE")


def joint_type_int(token):
    return _JOINT_TYPE_TO_INT.get(token, phys.JOINT_WELD)


def joint_type_token(value):
    return _JOINT_TYPE_FROM_INT.get(int(value), "WELD")


# ---------------------------------------------------------------------------
# Property groups
# ---------------------------------------------------------------------------

_SUPPRESS = [False]          # set while import fills properties in bulk


class suppress_updates:
    def __enter__(self):
        self.prev = _SUPPRESS[0]
        _SUPPRESS[0] = True

    def __exit__(self, *exc):
        _SUPPRESS[0] = self.prev


def stop_playback(context=None):
    """Halt animation playback. Bullet crashes if bodies or constraints are
    rebuilt while it is stepping, so every rig edit calls this first."""
    context = context or bpy.context
    screen = getattr(context, "screen", None)
    if screen is None:
        wm = context.window_manager
        screen = wm.windows[0].screen if wm and wm.windows else None
    if screen is not None and screen.is_animation_playing:
        try:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        except RuntimeError:
            pass
        return True
    return False


def _on_body_changed(self, context):
    if _SUPPRESS[0]:
        return
    obj = self.id_data
    if isinstance(obj, bpy.types.Object):
        stop_playback(context)
        rebuild_body(context, obj)


def _on_joint_changed(self, context):
    if _SUPPRESS[0]:
        return
    obj = self.id_data
    if isinstance(obj, bpy.types.Object):
        stop_playback(context)
        sync_constraint(context, obj)


class M2PhysShapeProps(PropertyGroup):
    kind: EnumProperty(name="Shape", items=SHAPE_KIND_ITEMS, default="CAPSULE",
                       update=_on_body_changed)
    # Body-local coordinates. Capsule: p1 -> p2. Sphere / box: p1 is the centre.
    p1: FloatVectorProperty(name="Start", size=3, subtype="XYZ", unit="LENGTH",
                            precision=4, update=_on_body_changed)
    p2: FloatVectorProperty(name="End", size=3, subtype="XYZ", unit="LENGTH",
                            precision=4, default=(0.0, 0.0, 0.1), update=_on_body_changed)
    radius: FloatProperty(name="Radius", min=0.0005, default=0.03, unit="LENGTH",
                          precision=4, update=_on_body_changed)
    half_extents: FloatVectorProperty(name="Half Size", size=3, subtype="XYZ",
                                      unit="LENGTH", precision=4, min=0.0005,
                                      default=(0.05, 0.05, 0.05), update=_on_body_changed)
    box_axes: FloatVectorProperty(name="Box Axes", size=9, default=phys.IDENTITY_AXES)
    friction: FloatProperty(name="Friction", min=0.0, soft_max=1.0, default=0.7,
                            update=_on_body_changed)
    restitution: FloatProperty(name="Bounciness", min=0.0, soft_max=1.0, default=0.1,
                               description="Restitution: 0 = dead stop, 1 = perfectly bouncy",
                               update=_on_body_changed)
    density: FloatProperty(name="Density", min=0.0001, default=1000.0,
                           description="Mass per volume. Mass = density x the shape's volume",
                           update=_on_body_changed)
    # Carried through for an exact rewrite.
    unk_hex: StringProperty(default="00000000")
    x14: IntProperty(default=0)
    x18: FloatProperty(default=1.0)
    x1c: IntProperty(default=0)
    x1e: IntProperty(default=0)
    polytope_index: IntProperty(default=-1)


def _get_mass(self):
    return body_mass(self.id_data)


def _set_mass(self, value):
    obj = self.id_data
    set_body_mass(obj, max(float(value), 1e-5))
    if not _SUPPRESS[0]:
        stop_playback()
        rebuild_body(bpy.context, obj)
        for coll in obj.users_collection:       # spring strength follows mass
            if coll.m2_phys_rig.is_rig:
                for joint in joints_of(coll, obj):
                    sync_constraint(bpy.context, joint)


class M2PhysBodyProps(PropertyGroup):
    is_body: BoolProperty(default=False)
    mass: FloatProperty(name="Mass", get=_get_mass, set=_set_mass, min=0.0, precision=4,
                        description="Total mass. Changing it rescales the density of "
                                    "every shape, which is what the file stores")
    body_type: EnumProperty(name="Type", items=BODY_TYPE_ITEMS, default="DYNAMIC",
                            update=_on_body_changed)
    bone: StringProperty(name="Bone", description="The bone this body is attached to",
                         update=_on_body_changed)
    shapes: CollectionProperty(type=M2PhysShapeProps)
    active_shape: IntProperty(default=0)
    drag: FloatProperty(name="Drag", min=0.0, soft_max=10.0, default=0.0,
                        description="Air resistance. Higher values settle faster and swing less",
                        update=_on_body_changed)
    # Retail dynamic bodies carry unk1 = 5..10 and x28 = 0.01 with absolute
    # positions; the one retail rig with unk1 = 0 stores zero positions and
    # lets the joints place the bodies. Export follows that rule, so a body
    # with unk1 = 0 is written bone-relative.
    unk0: FloatProperty(name="unk0", default=1.0)
    x1c: FloatProperty(name="x1c", default=1.0)
    unk1: FloatProperty(name="unk1", default=10.0)
    x28: FloatProperty(name="x28", default=0.01)
    x2c_hex: StringProperty(default="00000000")
    pad_a_hex: StringProperty(default="0000")
    pad_b_hex: StringProperty(default="0000")
    # What the file said, and where import put the body. While the object has
    # not moved, export writes file_position back unchanged.
    file_position: FloatVectorProperty(size=3, precision=6)
    rest_location: FloatVectorProperty(size=3, precision=6)
    has_file_position: BoolProperty(default=False)
    file_index: IntProperty(default=-1)        # keeps the file's body order
    file_body_type: IntProperty(default=-1)    # raw type from the file, rewritten if unchanged
    file_body_token: StringProperty(default="")
    # Blender uses an object's origin as its centre of mass, so the object sits
    # at the shapes' mass centre. This is that centre in body-frame coordinates;
    # the body frame the file describes is the object matrix minus this offset.
    origin_offset: FloatVectorProperty(size=3, precision=7)


class M2PhysJointProps(PropertyGroup):
    is_joint: BoolProperty(default=False)
    joint_type: EnumProperty(name="Type", items=JOINT_TYPE_ITEMS, default="SHOULDER",
                             update=_on_joint_changed)
    body_a: bpy.props.PointerProperty(name="Body A", type=bpy.types.Object,
                                      description="The parent side (closer to the root)",
                                      update=_on_joint_changed)
    body_b: bpy.props.PointerProperty(name="Body B", type=bpy.types.Object,
                                      description="The child side",
                                      update=_on_joint_changed)
    # shoulder
    lower_twist: FloatProperty(name="Twist Min", default=-10.0, min=-180.0, max=0.0,
                               description="Degrees of twist about the joint's Z axis",
                               update=_on_joint_changed)
    upper_twist: FloatProperty(name="Twist Max", default=10.0, min=0.0, max=180.0,
                               update=_on_joint_changed)
    cone_angle: FloatProperty(name="Cone Angle", default=20.0, min=0.0, max=180.0,
                              description="How far the child may swing away from the joint's Z axis, in degrees",
                              update=_on_joint_changed)
    max_motor_torque: FloatProperty(name="Motor Torque", default=0.0, min=0.0)
    motor_mode: IntProperty(name="Motor Mode", default=1, min=0,
                            description="1 = spring back to the rest pose (position motor)")
    motor_frequency_hz: FloatProperty(
        name="Stiffness (Hz)", default=1.0, min=0.0, soft_max=10.0,
        description="Spring that pulls the part back to its rest pose. Higher resists wind, "
                    "movement and collisions more; 0 = free. Retail uses 1 to 3",
        update=_on_joint_changed)
    motor_damping_ratio: FloatProperty(
        name="Stiffness Damping", default=0.7, min=0.0, soft_max=2.0,
        description="1 settles without wobble, lower wobbles longer", update=_on_joint_changed)
    # spherical
    friction_torque: FloatProperty(name="Friction Torque", default=0.0, min=0.0,
                                   description="Resistance to swinging", update=_on_joint_changed)
    # weld
    angular_frequency_hz: FloatProperty(name="Spring Frequency", default=0.0, min=0.0,
                                        soft_max=30.0,
                                        description="0 = rigid. Above 0 the joint becomes a rotational spring; "
                                                    "higher is stiffer (Hz)",
                                        update=_on_joint_changed)
    angular_damping_ratio: FloatProperty(name="Spring Damping", default=1.0, min=0.0,
                                         soft_max=2.0,
                                         description="1 = settles without wobbling, below 1 = wobbles",
                                         update=_on_joint_changed)
    linear_frequency_hz: FloatProperty(name="Linear Frequency", default=0.0, min=0.0,
                                       update=_on_joint_changed)
    linear_damping_ratio: FloatProperty(name="Linear Damping", default=0.0, min=0.0,
                                        update=_on_joint_changed)
    unk70: FloatProperty(default=0.0)
    # revolute / prismatic
    lower_limit: FloatProperty(name="Lower Limit", default=0.0, update=_on_joint_changed)
    upper_limit: FloatProperty(name="Upper Limit", default=0.0, update=_on_joint_changed)
    x68: FloatProperty(default=0.0)
    x70: FloatProperty(default=0.0)
    # distance
    distance_factor: FloatProperty(name="Distance Factor", default=1.0, min=0.0)
    unk_hex: StringProperty(default="00000000")
    # Imported frames + the transforms they were valid for. While nothing has
    # moved, export writes them back unchanged.
    frame_a: FloatVectorProperty(size=12, precision=7)
    frame_b: FloatVectorProperty(size=12, precision=7)
    rest_matrix: FloatVectorProperty(size=16, precision=7)
    has_file_frames: BoolProperty(default=False)
    file_index: IntProperty(default=-1)


class M2PhysRigProps(PropertyGroup):
    """Per-collection data that has no object to live on."""
    is_rig: BoolProperty(default=False)
    version: IntProperty(name="Version", default=phys.DEFAULT_VERSION, min=0, max=6)
    phyt: IntProperty(default=0)
    has_phyt: BoolProperty(default=False)
    chunk_order: StringProperty(default="")
    tags: StringProperty(default="")            # "body=BDY4,shape=SHP2"
    shoulder_size: IntProperty(default=0)
    raw_chunks: StringProperty(default="")      # "PLYT=<hex>;PHYV=<hex>"
    armature: bpy.props.PointerProperty(type=bpy.types.Object)


CLASSES = (M2PhysShapeProps, M2PhysBodyProps, M2PhysJointProps, M2PhysRigProps)


def register_props():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    register_handlers()
    print("[phys] Elune M2 physics build %s loaded" % BUILD, flush=True)
    bpy.types.Object.m2_phys_body = bpy.props.PointerProperty(type=M2PhysBodyProps)
    bpy.types.Object.m2_phys_joint = bpy.props.PointerProperty(type=M2PhysJointProps)
    bpy.types.Collection.m2_phys_rig = bpy.props.PointerProperty(type=M2PhysRigProps)


def unregister_props():
    unregister_handlers()
    for owner, attr in ((bpy.types.Object, "m2_phys_body"),
                        (bpy.types.Object, "m2_phys_joint"),
                        (bpy.types.Collection, "m2_phys_rig")):
        if hasattr(owner, attr):
            delattr(owner, attr)
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# Finding things
# ---------------------------------------------------------------------------

def is_body(obj) -> bool:
    return obj is not None and getattr(obj, "m2_phys_body", None) is not None \
        and obj.m2_phys_body.is_body


def is_joint(obj) -> bool:
    return obj is not None and getattr(obj, "m2_phys_joint", None) is not None \
        and obj.m2_phys_joint.is_joint


def rig_collections():
    return [c for c in bpy.data.collections if c.m2_phys_rig.is_rig]


def active_armature(context):
    for cand in (context.active_object, getattr(context, "object", None)):
        if cand is not None and cand.type == "ARMATURE":
            return cand
    for obj in context.selected_objects:
        if obj.type == "ARMATURE":
            return obj
    obj = context.active_object
    if is_body(obj) or is_joint(obj):
        for coll in obj.users_collection:
            if coll.m2_phys_rig.is_rig and coll.m2_phys_rig.armature is not None:
                return coll.m2_phys_rig.armature
    arms = [o for o in context.scene.objects if o.type == "ARMATURE"]
    return arms[0] if len(arms) == 1 else None


def find_rig(context, armature=None):
    """The rig collection for ``armature`` (or the active one), or None."""
    armature = armature or active_armature(context)
    rigs = rig_collections()
    if armature is not None:
        for coll in rigs:
            if coll.m2_phys_rig.armature == armature:
                return coll
    obj = context.active_object
    if obj is not None:
        for coll in obj.users_collection:
            if coll.m2_phys_rig.is_rig:
                return coll
    return rigs[0] if len(rigs) == 1 and armature is None else None


def ensure_rig(context, armature, name=None):
    coll = find_rig(context, armature) if armature is not None else None
    if coll is not None and (armature is None or coll.m2_phys_rig.armature == armature):
        return coll
    base = name or (armature.name if armature is not None else "M2")
    for suffix in ("_Armature", "_armature", ".Armature"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    coll = bpy.data.collections.new(base + COLLECTION_SUFFIX)
    context.scene.collection.children.link(coll)
    coll.m2_phys_rig.is_rig = True
    coll.m2_phys_rig.armature = armature
    return coll


def is_rig_object(obj) -> bool:
    """Anything that belongs to a physics rig and must stay out of the model."""
    if is_body(obj) or is_joint(obj) or obj.get("m2_phys_follow_helper"):
        return True
    return any(c.name.endswith(COLLECTION_SUFFIX) for c in obj.users_collection)


def rig_bodies(coll):
    return [o for o in coll.objects if is_body(o)]


def rig_joints(coll):
    return [o for o in coll.objects if is_joint(o)]


def root_body(coll):
    for o in rig_bodies(coll):
        if o.m2_phys_body.body_type == "ROOT":
            return o
    return None


def body_for_bone(coll, bone_name):
    for o in rig_bodies(coll):
        if o.m2_phys_body.bone == bone_name:
            return o
    return None


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def frame_from_z(z_axis: Vector) -> Matrix:
    """A right-handed 3x3 whose Z column is ``z_axis``."""
    z = Vector(z_axis)
    if z.length < 1e-8:
        z = Vector((0.0, 0.0, 1.0))
    z.normalize()
    ref = Vector((1.0, 0.0, 0.0)) if abs(z.x) < 0.9 else Vector((0.0, 1.0, 0.0))
    y = z.cross(ref).normalized()
    x = y.cross(z).normalized()
    m = Matrix.Identity(3)
    m.col[0], m.col[1], m.col[2] = x, y, z
    return m


def mat3x4_to_matrix(m) -> Matrix:
    out = Matrix.Identity(4)
    for c in range(3):
        for r in range(3):
            out[r][c] = m[c * 3 + r]
    out[0][3], out[1][3], out[2][3] = m[9], m[10], m[11]
    return out


def matrix_to_mat3x4(mat: Matrix):
    rot = mat.to_3x3().normalized()
    vals = []
    for c in range(3):
        vals.extend((rot[0][c], rot[1][c], rot[2][c]))
    t = mat.to_translation()
    vals.extend((t.x, t.y, t.z))
    return tuple(vals)


def _capsule_geometry(p1, p2, radius, segments=12, rings=4):
    p1, p2 = Vector(p1), Vector(p2)
    basis = frame_from_z(p2 - p1)
    length = (p2 - p1).length
    verts, faces = [], []
    rows = []
    for end, sign in ((p1, -1.0), (p2, 1.0)):
        ring_range = range(rings, -1, -1) if sign < 0 else range(0, rings + 1)
        for k in ring_range:
            a = (math.pi / 2.0) * k / rings          # 0 = equator, pi/2 = pole
            rr, zz = math.cos(a) * radius, math.sin(a) * radius * sign
            base_z = 0.0 if sign < 0 else length
            if k == rings:
                rows.append([len(verts)])
                verts.append(p1 + basis @ Vector((0.0, 0.0, base_z + zz)))
                continue
            row = []
            for s in range(segments):
                ang = 2.0 * math.pi * s / segments
                local = Vector((math.cos(ang) * rr, math.sin(ang) * rr, base_z + zz))
                row.append(len(verts))
                verts.append(p1 + basis @ local)
            rows.append(row)
    for a, b in zip(rows, rows[1:]):
        if len(a) == 1:
            faces.extend((a[0], b[(s + 1) % segments], b[s]) for s in range(segments))
        elif len(b) == 1:
            faces.extend((a[s], a[(s + 1) % segments], b[0]) for s in range(segments))
        else:
            faces.extend((a[s], a[(s + 1) % segments], b[(s + 1) % segments], b[s])
                         for s in range(segments))
    return verts, faces


def _sphere_geometry(center, radius, segments=12, rings=8):
    c = Vector(center)
    verts, faces = [], []
    for r in range(rings + 1):
        theta = math.pi * r / rings
        for s in range(segments):
            phi = 2.0 * math.pi * s / segments
            verts.append(c + Vector((math.sin(theta) * math.cos(phi) * radius,
                                     math.sin(theta) * math.sin(phi) * radius,
                                     math.cos(theta) * radius)))
    for r in range(rings):
        for s in range(segments):
            a, b = r * segments + s, r * segments + (s + 1) % segments
            faces.append((a, b, b + segments, a + segments))
    return verts, faces


def _box_geometry(center, axes, half):
    c = Vector(center)
    rot = Matrix.Identity(3)
    for col in range(3):
        for row in range(3):
            rot[row][col] = axes[col * 3 + row]
    verts = [c + rot @ Vector((x * half[0], y * half[1], z * half[2]))
             for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return verts, faces


def _marker_geometry(size=0.015):
    verts = [Vector(v) * size for v in
             ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
             (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    return verts, faces


def _polytope_vertices(obj, index):
    for coll in obj.users_collection:
        if not coll.m2_phys_rig.is_rig:
            continue
        raw = parse_raw_chunks(coll.m2_phys_rig.raw_chunks).get("PLYT")
        if raw:
            hulls = phys._plyt_vertices(raw)
            if 0 <= index < len(hulls):
                return hulls[index]
    return []


def shape_geometry(obj, shape):
    if shape.kind == "CAPSULE":
        return _capsule_geometry(shape.p1, shape.p2, shape.radius)
    if shape.kind == "SPHERE":
        return _sphere_geometry(shape.p1, shape.radius)
    if shape.kind == "BOX":
        return _box_geometry(shape.p1, shape.box_axes, shape.half_extents)
    pts = _polytope_vertices(obj, shape.polytope_index)
    if len(pts) >= 4:
        import bmesh
        bm = bmesh.new()
        for p in pts:
            bm.verts.new(p)
        bmesh.ops.convex_hull(bm, input=list(bm.verts))
        bm.verts.ensure_lookup_table()
        index = {v: i for i, v in enumerate(bm.verts)}
        verts = [v.co.copy() for v in bm.verts]
        faces = [tuple(index[v] for v in f.verts) for f in bm.faces]
        bm.free()
        return verts, faces
    return _marker_geometry(0.03)


def shape_volume(shape) -> float:
    if shape.kind == "CAPSULE":
        length = (Vector(shape.p2) - Vector(shape.p1)).length
        return math.pi * shape.radius ** 2 * length + 4.0 / 3.0 * math.pi * shape.radius ** 3
    if shape.kind == "SPHERE":
        return 4.0 / 3.0 * math.pi * shape.radius ** 3
    if shape.kind == "BOX":
        h = shape.half_extents
        return 8.0 * h[0] * h[1] * h[2]
    return 1e-4


def body_mass(obj) -> float:
    return sum(s.density * shape_volume(s) for s in obj.m2_phys_body.shapes)


def set_body_mass(obj, mass: float):
    """Rescale every shape's density so the body weighs ``mass``."""
    shapes = obj.m2_phys_body.shapes
    volume = sum(shape_volume(s) for s in shapes)
    if volume <= 0.0:
        return
    with suppress_updates():
        current = body_mass(obj)
        for s in shapes:
            s.density = (s.density * mass / current) if current > 1e-9 else mass / volume


def body_com_local(obj) -> Vector:
    """Centre of mass of the shapes, in body-frame coordinates."""
    total, weight = Vector((0.0, 0.0, 0.0)), 0.0
    for s in obj.m2_phys_body.shapes:
        centre = (Vector(s.p1) + Vector(s.p2)) * 0.5 if s.kind == "CAPSULE" else Vector(s.p1)
        m = max(s.density * shape_volume(s), 1e-12)
        total += centre * m
        weight += m
    return total / weight if weight > 0.0 else Vector((0.0, 0.0, 0.0))


# Every rig object is parented to a bone with matrix_parent_inverse equal to the
# inverse of that bone's REST tail matrix (armature space) and matrix_basis
# equal to the object's rest-pose transform in armature space. So:
#     world  = arm @ posed_tail @ rest_tail^-1 @ basis     (moves with the bone)
#     rest   = arm @ basis                                  (what the file stores)
# and the values export the same whatever pose or frame the scene is on.

def rest_world(obj) -> Matrix:
    """World matrix of a rig object with the skeleton in its rest pose."""
    if obj.parent is not None and obj.parent_type == "BONE" and obj.parent_bone:
        return obj.parent.matrix_world @ obj.matrix_basis
    return obj.matrix_world.copy()


def set_rest_world(obj, world: Matrix):
    if obj.parent is not None and obj.parent_type == "BONE" and obj.parent_bone:
        obj.matrix_basis = obj.parent.matrix_world.inverted() @ world
    else:
        obj.matrix_world = world


def attach_to_bone(obj, arm, bone_name: str):
    """Bone-parent ``obj`` (or free it when ``bone_name`` is empty) without
    moving it out of its rest-pose place."""
    world = rest_world(obj)
    bone = arm.data.bones.get(bone_name) if (arm is not None and bone_name) else None
    if bone is None:
        if obj.parent is not None:
            obj.parent = None
            obj.matrix_parent_inverse = Matrix.Identity(4)
        obj.matrix_world = world
        return
    if not (obj.parent == arm and obj.parent_type == "BONE" and obj.parent_bone == bone_name):
        obj.parent = arm
        obj.parent_type = "BONE"
        obj.parent_bone = bone_name
    rest_tail = bone.matrix_local @ Matrix.Translation((0.0, bone.length, 0.0))
    obj.matrix_parent_inverse = rest_tail.inverted()
    obj.matrix_basis = arm.matrix_world.inverted() @ world


def anchor_bone_for(coll, arm, bone_name: str) -> str:
    """The bone a simulated body should ride on: the nearest ancestor that is
    not itself driven by a dynamic body. Dynamic bodies are parented there so
    they travel with the character whenever the simulation is not stepping,
    and so the follow constraints never form a dependency cycle."""
    if arm is None or not bone_name or bone_name not in arm.data.bones:
        return ""
    dynamic = {o.m2_phys_body.bone for o in rig_bodies(coll)
               if o.m2_phys_body.body_type == "DYNAMIC"}
    bone = arm.data.bones[bone_name].parent
    while bone is not None and bone.name in dynamic:
        bone = bone.parent
    if bone is not None:
        return bone.name
    root = root_body(coll)
    if root is not None and root.m2_phys_body.bone and root.m2_phys_body.bone not in dynamic:
        return root.m2_phys_body.bone
    return ""


def body_frame_world(obj) -> Matrix:
    """Rest-pose world matrix of the body frame the .phys file describes."""
    return rest_world(obj) @ Matrix.Translation(-Vector(obj.m2_phys_body.origin_offset))


def set_body_frame_world(obj, frame: Matrix):
    set_rest_world(obj, frame @ Matrix.Translation(Vector(obj.m2_phys_body.origin_offset)))


def body_extent(obj) -> float:
    """Rough reach of the body from its origin (for spring inertia)."""
    reach = 0.01
    for s in obj.m2_phys_body.shapes:
        if s.kind == "CAPSULE":
            reach = max(reach, Vector(s.p1).length + s.radius, Vector(s.p2).length + s.radius)
        elif s.kind == "SPHERE":
            reach = max(reach, Vector(s.p1).length + s.radius)
        else:
            reach = max(reach, Vector(s.p1).length + max(s.half_extents))
    return reach


def _recentre(obj):
    """Slide the object's origin onto the current centre of mass without moving
    the body frame. Returns that centre."""
    props = obj.m2_phys_body
    com = body_com_local(obj)
    delta = com - Vector(props.origin_offset)
    if delta.length > 1e-9:
        set_rest_world(obj, rest_world(obj) @ Matrix.Translation(delta))
        with suppress_updates():
            props.origin_offset = com
    return com


_TYPE_COLORS = {"ROOT": (1.0, 0.75, 0.2, 1.0), "DYNAMIC": (0.35, 0.75, 1.0, 1.0),
                "KINEMATIC": (0.6, 1.0, 0.5, 1.0)}


def rebuild_body_mesh(obj):
    com = _recentre(obj)
    verts, faces = [], []
    for shape in obj.m2_phys_body.shapes:
        v, f = shape_geometry(obj, shape)
        base = len(verts)
        verts.extend(Vector(p) - com for p in v)
        faces.extend(tuple(i + base for i in face) for face in f)
    if not verts:
        verts, faces = _marker_geometry()
    me = obj.data
    me.clear_geometry()
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.update()
    obj.display_type = "WIRE"
    obj.show_in_front = True
    obj.color = _TYPE_COLORS.get(obj.m2_phys_body.body_type, (1, 1, 1, 1))


# ---------------------------------------------------------------------------
# Rigid body world
# ---------------------------------------------------------------------------

def ensure_world(context):
    scene = context.scene
    if scene.rigidbody_world is None:
        try:
            bpy.ops.rigidbody.world_add()
        except RuntimeError:
            with context.temp_override(scene=scene):
                bpy.ops.rigidbody.world_add()
    rbw = scene.rigidbody_world
    if rbw.collection is None:
        rbw.collection = bpy.data.collections.new("RigidBodyWorld")
    if rbw.constraints is None:
        rbw.constraints = bpy.data.collections.new("RigidBodyConstraints")
    # WoW rigs are a few centimetres across; the defaults tunnel and jitter.
    if hasattr(rbw, "substeps_per_frame"):
        rbw.substeps_per_frame = max(rbw.substeps_per_frame, 20)
    rbw.solver_iterations = max(rbw.solver_iterations, 30)
    rbw.point_cache.frame_start = min(rbw.point_cache.frame_start, scene.frame_start)
    rbw.point_cache.frame_end = max(rbw.point_cache.frame_end, scene.frame_end)
    return rbw


def drag_to_damping(drag: float) -> float:
    """WoW drag is an exponential decay rate; Blender damping is the fraction of
    velocity lost per second."""
    return max(0.0, min(1.0, 1.0 - math.exp(-max(drag, 0.0))))


def _world_matrix_of_bone(arm, bone_name):
    bone = arm.data.bones.get(bone_name) if arm is not None else None
    if bone is None:
        return None
    return arm.matrix_world @ bone.matrix_local


def _rig_armature(obj):
    for coll in obj.users_collection:
        if coll.m2_phys_rig.is_rig:
            return coll.m2_phys_rig.armature
    return None


def rebuild_body(context, obj):
    """Regenerate mesh, parenting and rigid-body settings from the properties."""
    if not is_body(obj):
        return
    props = obj.m2_phys_body
    rebuild_body_mesh(obj)
    arm = _rig_armature(obj)
    dynamic = props.body_type == "DYNAMIC"
    coll = next((c for c in obj.users_collection if c.m2_phys_rig.is_rig), None)
    if dynamic and coll is not None:
        attach_to_bone(obj, arm, anchor_bone_for(coll, arm, props.bone))
    else:
        attach_to_bone(obj, arm, props.bone)

    rbw = ensure_world(context)
    if obj.name not in rbw.collection.objects:
        rbw.collection.objects.link(obj)
    rb = obj.rigid_body
    if rb is None:
        return
    rb.type = "ACTIVE" if dynamic else "PASSIVE"
    rb.kinematic = not dynamic
    rb.collision_shape = "CONVEX_HULL" if props.shapes else "SPHERE"
    rb.mesh_source = "BASE"
    rb.use_margin = True
    rb.collision_margin = 0.0
    rb.mass = max(body_mass(obj), 1e-4) if dynamic else 1.0
    if props.shapes:
        rb.friction = props.shapes[0].friction
        rb.restitution = props.shapes[0].restitution
    rb.linear_damping = drag_to_damping(props.drag) if dynamic else 0.0
    rb.angular_damping = max(0.1, drag_to_damping(props.drag)) if dynamic else 0.0
    # Shapeless bodies (the root, usually) are anchors only: they touch nothing.
    if not props.shapes:
        rb.collision_collections = [False] * 20


def apply_self_collision(coll, enabled: bool):
    """With self-collision off each dynamic body gets a private layer, so they
    only ever hit kinematic bodies (which sit on every layer)."""
    dynamic = [o for o in rig_bodies(coll) if o.m2_phys_body.body_type == "DYNAMIC"]
    for i, obj in enumerate(dynamic):
        if obj.rigid_body is None:
            continue
        layers = [False] * 20
        layers[0 if enabled else i % 20] = True
        obj.rigid_body.collision_collections = layers
    for obj in rig_bodies(coll):
        if obj.m2_phys_body.body_type == "DYNAMIC" or obj.rigid_body is None:
            continue
        obj.rigid_body.collision_collections = (
            [True] * 20 if obj.m2_phys_body.shapes else [False] * 20)


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def _lock_linear(con):
    for axis in "xyz":
        setattr(con, "use_limit_lin_" + axis, True)
        setattr(con, "limit_lin_%s_lower" % axis, 0.0)
        setattr(con, "limit_lin_%s_upper" % axis, 0.0)


def _limit_angular(con, axis, lower_deg, upper_deg):
    free = (upper_deg - lower_deg) >= 359.0
    setattr(con, "use_limit_ang_" + axis, not free)
    setattr(con, "limit_ang_%s_lower" % axis, math.radians(lower_deg))
    setattr(con, "limit_ang_%s_upper" % axis, math.radians(upper_deg))


def _spring_constants(body_b, frequency_hz, damping_ratio, angular=True):
    """Spring stiffness / damping that give body B the requested natural
    frequency about the joint."""
    mass = max(body_mass(body_b), 1e-4) if body_b is not None else 1.0
    inertia = mass
    if angular:
        reach = body_extent(body_b) if body_b is not None else 0.1
        inertia = mass * reach * reach / 3.0
    omega = 2.0 * math.pi * frequency_hz
    stiffness = inertia * omega * omega
    damping = 2.0 * damping_ratio * math.sqrt(max(stiffness * inertia, 0.0))
    return stiffness, damping


def sync_constraint(context, empty):
    """Mirror the joint properties onto Blender's rigid-body constraint."""
    if not is_joint(empty):
        return
    props = empty.m2_phys_joint
    rbw = ensure_world(context)
    if empty.name not in rbw.constraints.objects:
        rbw.constraints.objects.link(empty)
    con = empty.rigid_body_constraint
    if con is None:
        return
    usable = (is_body(props.body_a) and is_body(props.body_b)
              and props.body_a != props.body_b)
    con.object1 = props.body_a if usable else None
    con.object2 = props.body_b if usable else None
    con.enabled = usable
    con.disable_collisions = True
    con.use_breaking = False
    if hasattr(con, "use_override_solver_iterations"):
        con.use_override_solver_iterations = True
        con.solver_iterations = 40

    kind = props.joint_type
    if kind == "SPHERICAL":
        con.type = "POINT"
    elif kind == "SHOULDER":
        sprung = props.motor_mode != 0 and props.motor_frequency_hz > 0.0
        con.type = "GENERIC_SPRING" if sprung else "GENERIC"
        _lock_linear(con)
        cone = min(props.cone_angle, 180.0)
        _limit_angular(con, "x", -cone, cone)
        _limit_angular(con, "y", -min(cone, 89.0), min(cone, 89.0))   # Bullet caps Y at 90
        _limit_angular(con, "z", props.lower_twist, props.upper_twist)
        if sprung:
            if hasattr(con, "spring_type"):
                con.spring_type = "SPRING2"
            k, c = _spring_constants(props.body_b, props.motor_frequency_hz,
                                     props.motor_damping_ratio)
            for axis in "xyz":
                setattr(con, "use_spring_" + axis, False)
                setattr(con, "use_spring_ang_" + axis, True)
                setattr(con, "spring_stiffness_ang_" + axis, k)
                setattr(con, "spring_damping_ang_" + axis, c)
    elif kind == "WELD":
        soft_ang = props.angular_frequency_hz > 0.0
        soft_lin = props.linear_frequency_hz > 0.0
        if not soft_ang and not soft_lin:
            con.type = "FIXED"
        else:
            con.type = "GENERIC_SPRING"
            if hasattr(con, "spring_type"):
                con.spring_type = "SPRING2"
            for axis in "xyz":
                if soft_lin:
                    k, c = _spring_constants(props.body_b, props.linear_frequency_hz,
                                             props.linear_damping_ratio, angular=False)
                    setattr(con, "use_limit_lin_" + axis, False)
                    setattr(con, "use_spring_" + axis, True)
                    setattr(con, "spring_stiffness_" + axis, k)
                    setattr(con, "spring_damping_" + axis, c)
                else:
                    setattr(con, "use_spring_" + axis, False)
                if soft_ang:
                    k, c = _spring_constants(props.body_b, props.angular_frequency_hz,
                                             props.angular_damping_ratio)
                    setattr(con, "use_limit_ang_" + axis, False)
                    setattr(con, "use_spring_ang_" + axis, True)
                    setattr(con, "spring_stiffness_ang_" + axis, k)
                    setattr(con, "spring_damping_ang_" + axis, c)
                else:
                    setattr(con, "use_spring_ang_" + axis, False)
                    _limit_angular(con, axis, 0.0, 0.0)
            if not soft_lin:
                _lock_linear(con)
    elif kind == "REVOLUTE":
        con.type = "GENERIC"
        _lock_linear(con)
        _limit_angular(con, "x", 0.0, 0.0)
        _limit_angular(con, "y", 0.0, 0.0)
        if props.lower_limit == 0.0 and props.upper_limit == 0.0:
            _limit_angular(con, "z", -180.0, 180.0)
        else:
            _limit_angular(con, "z", props.lower_limit, props.upper_limit)
    elif kind == "PRISMATIC":
        con.type = "GENERIC"
        _lock_linear(con)
        con.limit_lin_z_lower = props.lower_limit
        con.limit_lin_z_upper = props.upper_limit
        for axis in "xyz":
            _limit_angular(con, axis, 0.0, 0.0)
    else:                                  # DISTANCE: a loose tether
        con.type = "GENERIC"
        reach = 0.0
        if props.body_a is not None and props.body_b is not None:
            reach = (body_frame_world(props.body_a).translation
                     - body_frame_world(props.body_b).translation).length
        reach *= max(props.distance_factor, 0.0)
        for axis in "xyz":
            setattr(con, "use_limit_lin_" + axis, True)
            setattr(con, "limit_lin_%s_lower" % axis, -reach)
            setattr(con, "limit_lin_%s_upper" % axis, reach)
            setattr(con, "use_limit_ang_" + axis, False)

    empty.empty_display_type = "ARROWS"
    empty.empty_display_size = 0.04
    empty.show_in_front = True
    coll = next((c for c in empty.users_collection if c.m2_phys_rig.is_rig), None)
    arm = coll.m2_phys_rig.armature if coll is not None else None
    child = props.body_b if usable else None
    if child is not None and arm is not None:
        cb = child.m2_phys_body
        bone = anchor_bone_for(coll, arm, cb.bone) if cb.body_type == "DYNAMIC" else cb.bone
        attach_to_bone(empty, arm, bone)


# ---------------------------------------------------------------------------
# Creating rig pieces
# ---------------------------------------------------------------------------

def new_body(context, coll, name, world_matrix: Matrix, body_type="DYNAMIC", bone=""):
    me = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, me)
    coll.objects.link(obj)
    obj.matrix_world = world_matrix
    with suppress_updates():
        props = obj.m2_phys_body
        props.is_body = True
        props.body_type = body_type
        props.bone = bone
        if body_type != "DYNAMIC":          # retail anchors: [1, 1, 0, 0, 0.5]
            props.unk1 = 0.0
            props.x28 = 0.5
    return obj


def add_shape(obj, kind="CAPSULE", **values):
    with suppress_updates():
        shape = obj.m2_phys_body.shapes.add()
        shape.kind = kind
        for key, value in values.items():
            setattr(shape, key, value)
    return shape


def new_joint(context, coll, name, world_matrix: Matrix, body_a, body_b,
              joint_type="SHOULDER", **values):
    empty = bpy.data.objects.new(name, None)
    coll.objects.link(empty)
    empty.matrix_world = world_matrix
    with suppress_updates():
        props = empty.m2_phys_joint
        props.is_joint = True
        props.joint_type = joint_type
        props.body_a = body_a
        props.body_b = body_b
        for key, value in values.items():
            setattr(props, key, value)
    sync_constraint(context, empty)
    return empty


def joints_of(coll, body):
    return [j for j in rig_joints(coll)
            if j.m2_phys_joint.body_a == body or j.m2_phys_joint.body_b == body]


def detach_from_world(obj):
    """Take an object out of the rigid-body world cleanly. Deleting a body
    that a constraint still points at leaves Bullet with a dangling reference
    and crashes the next frame step, so joints are unhooked first."""
    stop_playback()
    scene = bpy.context.scene
    rbw = scene.rigidbody_world if scene is not None else None
    if is_body(obj):
        for coll in obj.users_collection:
            if coll.m2_phys_rig.is_rig:
                for joint in joints_of(coll, obj):
                    con = joint.rigid_body_constraint
                    if con is not None:
                        con.enabled = False
                        if con.object1 == obj:
                            con.object1 = None
                        if con.object2 == obj:
                            con.object2 = None
    con = obj.rigid_body_constraint
    if con is not None:
        con.enabled = False
        con.object1 = None
        con.object2 = None
        if rbw is not None and rbw.constraints is not None \
                and obj.name in rbw.constraints.objects:
            rbw.constraints.objects.unlink(obj)
    if obj.rigid_body is not None and rbw is not None and rbw.collection is not None \
            and obj.name in rbw.collection.objects:
        rbw.collection.objects.unlink(obj)
    if rbw is not None:
        rbw.point_cache.frame_start = rbw.point_cache.frame_start   # invalidate cache


def remove_object(obj):
    detach_from_world(obj)
    data = obj.data if obj.type == "MESH" else None
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is not None and data.users == 0:
        bpy.data.meshes.remove(data)


# ---------------------------------------------------------------------------
# Playback guard
# ---------------------------------------------------------------------------

def _body_usable(obj, rbw, view_layer):
    if obj is None or not is_body(obj) or obj.rigid_body is None:
        return False
    if obj.hide_viewport or (rbw.collection is not None
                             and obj.name not in rbw.collection.objects):
        return False
    try:
        if view_layer is not None and obj.hide_get(view_layer=view_layer):
            return False
    except (RuntimeError, TypeError):
        pass
    return True


def sanitize_constraints(scene, view_layer=None):
    """Disable any joint whose bodies are hidden, deleted or out of the world.
    Runs before every frame step: Bullet dereferences freed bodies through
    such constraints and crashes."""
    rbw = scene.rigidbody_world
    if rbw is None:
        return 0
    changed = 0
    for coll in rig_collections():
        for joint in rig_joints(coll):
            con = joint.rigid_body_constraint
            if con is None:
                continue
            props = joint.m2_phys_joint
            ok = (_body_usable(props.body_a, rbw, view_layer)
                  and _body_usable(props.body_b, rbw, view_layer)
                  and props.body_a != props.body_b)
            if ok:
                if con.object1 != props.body_a or con.object2 != props.body_b:
                    con.object1, con.object2 = props.body_a, props.body_b
                    changed += 1
            elif con.enabled or con.object1 is not None or con.object2 is not None:
                con.enabled = False
                con.object1 = None
                con.object2 = None
                changed += 1
            if ok and not con.enabled:
                con.enabled = True
                changed += 1
    return changed


@bpy.app.handlers.persistent
def _frame_change_pre(scene, *args):
    try:
        if scene.rigidbody_world is not None and any(rig_collections()):
            sanitize_constraints(scene, getattr(bpy.context, "view_layer", None))
    except Exception:  # noqa: BLE001 - a guard must never raise into playback
        pass


def register_handlers():
    if _frame_change_pre not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(_frame_change_pre)


def unregister_handlers():
    if _frame_change_pre in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(_frame_change_pre)


# ---------------------------------------------------------------------------
# Rig-level storage
# ---------------------------------------------------------------------------

def parse_raw_chunks(text: str):
    out = {}
    for part in (text or "").split(";"):
        if "=" in part:
            key, hexed = part.split("=", 1)
            try:
                out[key] = bytes.fromhex(hexed)
            except ValueError:
                pass
    return out


def format_raw_chunks(chunks) -> str:
    return ";".join("%s=%s" % (k, bytes(v).hex()) for k, v in chunks.items())


def parse_tags(text: str):
    return dict(p.split("=", 1) for p in (text or "").split(",") if "=" in p)


def format_tags(tags) -> str:
    return ",".join("%s=%s" % kv for kv in tags.items())


# ---------------------------------------------------------------------------
# Preview: make bones follow their dynamic bodies
# ---------------------------------------------------------------------------

FOLLOW_HELPER_PREFIX = "phys_follow_"


def clear_follow(arm, coll=None):
    removed = 0
    if arm is not None and arm.pose is not None:
        for pb in arm.pose.bones:
            for con in [c for c in pb.constraints if c.name == FOLLOW_CONSTRAINT]:
                pb.constraints.remove(con)
                removed += 1
    helpers = [o for o in bpy.data.objects if o.get("m2_phys_follow_helper")
               and (coll is None or coll in o.users_collection)]
    for helper in helpers:
        bpy.data.objects.remove(helper, do_unlink=True)
    return removed


def wire_follow(coll):
    """Make each simulated bone follow its body.

    The client poses a physics bone as ``body x body_rest^-1 x bone_rest``. A
    helper empty parked at the bone's rest matrix and parented to the body
    carries exactly that transform, and the bone copies it in world space.
    Rest matrices come from the rig's stored rest pose, so this is right on
    any frame, not only when the skeleton happens to be at rest.
    """
    arm = coll.m2_phys_rig.armature
    if arm is None:
        return 0
    clear_follow(arm, coll)
    wired = 0
    for body in rig_bodies(coll):
        props = body.m2_phys_body
        if props.body_type != "DYNAMIC":
            continue
        pb = arm.pose.bones.get(props.bone)
        if pb is None:
            continue
        helper = bpy.data.objects.new(FOLLOW_HELPER_PREFIX + pb.name, None)
        coll.objects.link(helper)
        helper["m2_phys_follow_helper"] = True
        helper.empty_display_type = "PLAIN_AXES"
        helper.empty_display_size = 0.005
        helper.hide_select = True
        helper.parent = body
        helper.matrix_parent_inverse = rest_world(body).inverted()
        helper.matrix_basis = arm.matrix_world @ pb.bone.matrix_local
        con = pb.constraints.new("COPY_TRANSFORMS")
        con.name = FOLLOW_CONSTRAINT
        con.target = helper
        con.owner_space = "WORLD"
        con.target_space = "WORLD"
        wired += 1
    return wired


def reset_simulation(context):
    scene = context.scene
    rbw = scene.rigidbody_world
    if rbw is None:
        return
    rbw.point_cache.frame_start = scene.frame_start
    rbw.point_cache.frame_end = max(scene.frame_end, scene.frame_start + 1)
    # Nudging a cache setting invalidates the cached simulation.
    rbw.point_cache.frame_start = scene.frame_start
    substeps = getattr(rbw, "substeps_per_frame", None)
    if substeps is not None:
        rbw.substeps_per_frame = substeps + 1
        rbw.substeps_per_frame = substeps
    scene.frame_set(scene.frame_start)
