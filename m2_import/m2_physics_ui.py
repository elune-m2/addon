"""M2 Physics panel: build a rigid-body rig from bones and control it live.

Two primary buttons — **Add Physics Mesh** and **Add Collision Mesh** —
each create a body attached to the active pose bone (or at the 3D cursor
if no bone is active). The panel then reflects the selection:

  - selecting a body                → live shape / friction / mass / bone
  - selecting a joint empty         → live joint type + body A/B
  - being in Pose Mode with a bone  → 'add body at this bone' shortcut
  - otherwise                       → global setup + preview controls

All property edits update Blender's rigid body settings the instant they
change, so you can drag friction/mass sliders while the sim plays.
"""

from __future__ import annotations

import math

import bpy
from bpy.props import (
    BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty,
    StringProperty,
)
from bpy.types import Operator, Panel, PropertyGroup
from mathutils import Matrix, Vector

from . import phys


PHYS_COLLECTION_SUFFIX = "_phys"

# ---------------------------------------------------------------------------
# Scene-level defaults for "add mesh" actions
# ---------------------------------------------------------------------------

class M2PhysicsProps(PropertyGroup):
    shape_type: EnumProperty(
        name="Shape",
        description="Shape kind for the next Add action",
        items=[
            ("CAPSULE", "Capsule", "Aligned to the bone axis"),
            ("BOX",     "Box",     "Aligned to the bone axis"),
            ("SPHERE",  "Sphere",  "Centred on the bone midpoint"),
        ],
        default="CAPSULE",
    )
    default_radius: FloatProperty(
        name="Default Radius / Half-thickness",
        default=0.03, min=0.001, max=10.0,
    )
    preview_start: IntProperty(name="Frame Start", default=1, min=0)
    preview_end: IntProperty(name="Frame End", default=120, min=1)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get_active_armature(context):
    for candidate in (context.active_object,
                      context.view_layer.objects.active,
                      context.object):
        if candidate is not None and candidate.type == "ARMATURE":
            return candidate
    for obj in context.selected_objects:
        if obj.type == "ARMATURE":
            return obj
    return None


def _model_name(context):
    arm = _get_active_armature(context)
    return arm.name if arm is not None else "M2"


def _ensure_world(context):
    if context.scene.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()
    if context.scene.rigidbody_world.collection is None:
        rb_coll = bpy.data.collections.new("RigidBodyWorld")
        context.scene.rigidbody_world.collection = rb_coll


def _phys_collection(context, name: str):
    coll_name = name + PHYS_COLLECTION_SUFFIX
    coll = bpy.data.collections.get(coll_name)
    if coll is None:
        coll = bpy.data.collections.new(coll_name)
        context.scene.collection.children.link(coll)
    return coll


def _find_phys_collection(context):
    """Locate the phys collection for the active rig without assuming a
    particular naming convention.

    Preference order:
      1. `<active-armature-name>_phys`   (matches what authoring creates)
      2. `<armature-name minus '_Armature' suffix>_phys` (matches what
         the M2 importer creates, since armatures are named
         `<m2filename>_Armature` but the .phys collection is named after
         the raw m2 filename)
      3. Any collection whose name ends with `_phys` and which contains
         a rigid body — used when the user has renamed either side.

    Returns None only if there is no `_phys` collection in the scene at all.
    """
    arm = _get_active_armature(context)
    if arm is not None:
        exact = bpy.data.collections.get(arm.name + PHYS_COLLECTION_SUFFIX)
        if exact is not None:
            return exact
        base = arm.name
        for suffix in ("_Armature", "_armature", ".Armature"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        candidate = bpy.data.collections.get(base + PHYS_COLLECTION_SUFFIX)
        if candidate is not None:
            return candidate
    # Last resort: pick a _phys collection that actually contains rigid bodies.
    for c in bpy.data.collections:
        if c.name.endswith(PHYS_COLLECTION_SUFFIX) and any(
                o.rigid_body is not None for o in c.objects):
            return c
    return None


def _cube_mesh(name: str, hx: float, hy: float, hz: float):
    me = bpy.data.meshes.new(name)
    verts = [(x*hx, y*hy, z*hz)
             for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    faces = [(0,1,3,2),(4,5,7,6),(0,1,5,4),(2,3,7,6),(0,2,6,4),(1,3,7,5)]
    me.from_pydata(verts, [], faces)
    me.update()
    return me


def _cylinder_mesh(name: str, radius: float, height: float, segments: int = 12,
                   along_y: bool = False):
    """Cylinder along local Z by default; pass `along_y=True` to get a
    cylinder along local +Y with its BOTTOM at the origin (extending
    from y=0 to y=height) — the geometry a bone-aligned body needs so
    its object matrix can equal the bone's rest matrix exactly."""
    me = bpy.data.meshes.new(name)
    verts, faces = [], []
    for s in range(segments):
        a = 2 * math.pi * s / segments
        cx, cy = math.cos(a) * radius, math.sin(a) * radius
        if along_y:
            # Along +Y, base at y=0, top at y=height. Cross-section in XZ.
            verts.append((cx, 0.0,   cy))
            verts.append((cx, height, cy))
        else:
            h2 = height * 0.5
            verts.append((cx, cy, -h2))
            verts.append((cx, cy, +h2))
    for s in range(segments):
        bl = 2 * s
        tl = 2 * s + 1
        br = 2 * ((s + 1) % segments)
        tr = 2 * ((s + 1) % segments) + 1
        faces.append((bl, br, tr, tl))
    faces.append(tuple(reversed([2*s for s in range(segments)])))
    faces.append(tuple([2*s + 1 for s in range(segments)]))
    me.from_pydata(verts, [], faces)
    me.update()
    return me


def _uv_sphere_mesh(name: str, radius: float, rings: int = 8, segments: int = 12):
    me = bpy.data.meshes.new(name)
    verts, faces = [], []
    for r in range(rings + 1):
        theta = math.pi * r / rings
        z = math.cos(theta) * radius
        rr = math.sin(theta) * radius
        for s in range(segments):
            phi = 2 * math.pi * s / segments
            verts.append((rr * math.cos(phi), rr * math.sin(phi), z))
    for r in range(rings):
        for s in range(segments):
            a = r * segments + s
            b = r * segments + (s + 1) % segments
            c = (r + 1) * segments + (s + 1) % segments
            d = (r + 1) * segments + s
            faces.append((a, b, c, d))
    me.from_pydata(verts, [], faces)
    me.update()
    return me


def _cube_mesh_offset_y(name: str, hx: float, hy: float, hz: float,
                        offset_y: float):
    """Cube shifted along +Y by `offset_y` so its base sits at y=0."""
    me = bpy.data.meshes.new(name)
    verts = [(x*hx, y*hy + offset_y, z*hz)
             for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    faces = [(0,1,3,2),(4,5,7,6),(0,1,5,4),(2,3,7,6),(0,2,6,4),(1,3,7,5)]
    me.from_pydata(verts, [], faces)
    me.update()
    return me


def _uv_sphere_mesh_offset_y(name: str, radius: float, offset_y: float,
                             rings: int = 8, segments: int = 12):
    """UV sphere centred at (0, offset_y, 0)."""
    me = bpy.data.meshes.new(name)
    verts, faces = [], []
    for r in range(rings + 1):
        theta = math.pi * r / rings
        y = math.cos(theta) * radius + offset_y
        rr = math.sin(theta) * radius
        for s in range(segments):
            phi = 2 * math.pi * s / segments
            verts.append((rr * math.cos(phi), y, rr * math.sin(phi)))
    for r in range(rings):
        for s in range(segments):
            a = r * segments + s
            b = r * segments + (s + 1) % segments
            c = (r + 1) * segments + (s + 1) % segments
            d = (r + 1) * segments + s
            faces.append((a, b, c, d))
    me.from_pydata(verts, [], faces)
    me.update()
    return me


def _enter_object_mode(context):
    """Force OBJECT mode; return the previous mode key so callers can restore.

    Blender's rigid-body / selection ops all require OBJECT mode context.
    User may click our buttons from Pose Mode or Edit Mode — silently
    switch, do the work, switch back.
    """
    prev = context.mode
    if prev == "OBJECT":
        return prev
    # context.mode gives verbose names like 'POSE' / 'EDIT_ARMATURE' /
    # 'EDIT_MESH'; mode_set takes 'OBJECT' / 'POSE' / 'EDIT' etc. The
    # verbose form ALSO works for the return trip because Blender maps it.
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:  # noqa: BLE001
        pass
    return prev


def _restore_mode(context, prev_mode: str):
    if prev_mode in (None, "", "OBJECT"):
        return
    # 'EDIT_MESH' / 'EDIT_ARMATURE' -> 'EDIT'; 'PAINT_WEIGHT' etc. stay.
    target = "EDIT" if prev_mode.startswith("EDIT_") else prev_mode
    try:
        bpy.ops.object.mode_set(mode=target)
    except Exception:  # noqa: BLE001
        pass


def _deselect_all(context):
    """Pure-API deselect — safe from any mode."""
    for o in context.view_layer.objects:
        try:
            o.select_set(False)
        except (RuntimeError, ReferenceError):
            pass


def _apply_rigid_body(context, obj, kind: str, blender_shape: str,
                      friction: float = 0.5, restitution: float = 0.0,
                      mass: float = 1.0, kinematic: bool = False):
    prev_mode = _enter_object_mode(context)
    try:
        _deselect_all(context)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        if obj.rigid_body is None:
            bpy.ops.rigidbody.object_add(type=kind)
        else:
            obj.rigid_body.type = kind
        rb = obj.rigid_body
        rb.collision_shape = blender_shape
        rb.friction = friction
        rb.restitution = restitution
        rb.mass = max(mass, 0.001)
        rb.kinematic = kinematic
    finally:
        _restore_mode(context, prev_mode)


def _apply_constraint(context, empty, con_type: str, obj_a, obj_b):
    prev_mode = _enter_object_mode(context)
    try:
        _deselect_all(context)
        empty.select_set(True)
        context.view_layer.objects.active = empty
        if empty.rigid_body_constraint is None:
            bpy.ops.rigidbody.constraint_add(type=con_type)
        else:
            empty.rigid_body_constraint.type = con_type
        con = empty.rigid_body_constraint
        con.enabled = True
        con.object1 = obj_a
        con.object2 = obj_b
    finally:
        _restore_mode(context, prev_mode)


def _index_bodies_by_bone(coll):
    """{bone_name: body_obj} — includes both bone-parented bodies (root
    anchors etc.) and dynamic bodies that carry the m2_phys_bone_name
    custom prop from the bone_preset operator."""
    out = {}
    if coll is None:
        return out
    for obj in coll.objects:
        if obj.rigid_body is None:
            continue
        name = ""
        if obj.parent_type == "BONE" and obj.parent_bone:
            name = obj.parent_bone
        else:
            name = obj.get("m2_phys_bone_name", "") or ""
        if name:
            out.setdefault(name, obj)
    return out


def _make_shape_mesh_at_bone(name: str, arm_obj, pose_bone, shape: str,
                             radius: float):
    """Return (mesh, blender_shape_str, world_matrix).

    The mesh is built IN the bone's local frame (origin = bone head,
    +Y = bone axis), so the returned world matrix is EXACTLY the bone's
    rest world matrix. When Copy Transforms copies the body back to the
    bone, the bone lands at its original rest pose (no leaked rotation
    or offset from mesh construction)."""
    length = max(pose_bone.bone.length, 0.01)
    if shape == "CAPSULE":
        me = _cylinder_mesh(name + "_mesh", radius=radius, height=length,
                            along_y=True)
        bshape = "CAPSULE"
    elif shape == "BOX":
        # Box centred on the bone: half-length in Y, half-thickness in X/Z.
        # We build the geometry from -length/2 to +length/2 in Y and then
        # shift up so the base sits at y=0 (bone head-aligned).
        me = _cube_mesh_offset_y(name + "_mesh",
                                 hx=radius, hy=length * 0.5, hz=radius,
                                 offset_y=length * 0.5)
        bshape = "BOX"
    else:
        # Sphere centred at bone midpoint (y = length/2).
        me = _uv_sphere_mesh_offset_y(name + "_mesh", radius=radius,
                                      offset_y=length * 0.5)
        bshape = "SPHERE"

    # REST pose matrix — bone.matrix_local is the rest bone matrix in
    # armature space (origin=head, +Y along bone). We MUST use rest, not
    # `pose_bone.matrix`, so the body's initial pose is stable regardless
    # of what animation frame the user is on when clicking the preset.
    bone_world = arm_obj.matrix_world @ pose_bone.bone.matrix_local
    return me, bshape, bone_world


def _bone_parent_keeping_transform(obj, arm_obj, pose_bone):
    """Bone-parent obj such that its current world matrix is preserved.

    Blender's `parent_type='BONE'` uses the bone TAIL as parent origin,
    so we compute matrix_parent_inverse = tail_world⁻¹ @ obj.world.
    """
    length = max(pose_bone.bone.length, 0.01)
    bone_world = arm_obj.matrix_world @ pose_bone.matrix
    tail_world = bone_world @ Matrix.Translation((0.0, length, 0.0))
    world = obj.matrix_world.copy()
    obj.parent = arm_obj
    obj.parent_type = "BONE"
    obj.parent_bone = pose_bone.name
    obj.matrix_parent_inverse = tail_world.inverted() @ world


def _make_shape_mesh_at_cursor(name: str, context, shape: str, radius: float):
    if shape == "CAPSULE":
        me = _cylinder_mesh(name + "_mesh", radius=radius, height=radius * 4)
        bshape = "CAPSULE"
    elif shape == "BOX":
        me = _cube_mesh(name + "_mesh", hx=radius, hy=radius, hz=radius)
        bshape = "BOX"
    else:
        me = _uv_sphere_mesh(name + "_mesh", radius=radius)
        bshape = "SPHERE"
    mtx = Matrix.Translation(context.scene.cursor.location)
    return me, bshape, mtx


def _resolve_armature(context, pose_bone=None, hint=None):
    """Find the armature to attach to. Priority:
       1. explicit hint (bone_preset passes this so the loop can't lose it)
       2. pose_bone.id_data → its armature object (walks scene objects)
       3. active object / selection / context.object
    Falls back to None so callers can decide whether to error or spawn at cursor."""
    if hint is not None and hint.type == "ARMATURE":
        return hint
    if pose_bone is not None:
        arm_data = pose_bone.id_data  # bpy.types.Armature
        for o in bpy.data.objects:
            if o.type == "ARMATURE" and o.data is arm_data:
                return o
    return _get_active_armature(context)


def _add_body_common(context, is_active: bool, pose_bone=None,
                     preset: str = None, arm=None):
    """Shared body-add path. Uses the given pose_bone (or the active one)."""
    _ensure_world(context)
    arm = _resolve_armature(context, pose_bone=pose_bone, hint=arm)
    props = context.scene.m2_physics
    coll = _phys_collection(context, _model_name(context))

    pb = pose_bone
    if pb is None and arm is not None and context.mode == "POSE":
        pb = context.active_pose_bone

    if pb is not None and arm is not None:
        me, bshape, world = _make_shape_mesh_at_bone(
            "phys_" + ("body" if is_active else "coll") + "_" + pb.name,
            arm, pb, props.shape_type, float(props.default_radius))
        print(f"[m2phys] placing body '{me.name}' on bone '{pb.name}' — "
              f"world head={world.translation}", flush=True)
    else:
        prefix = "phys_body" if is_active else "phys_coll"
        me, bshape, world = _make_shape_mesh_at_cursor(
            prefix, context, props.shape_type, float(props.default_radius))
        print(f"[m2phys] no bone / no armature (pose_bone={pose_bone}, "
              f"pb={pb}, arm={arm}, mode={context.mode}) — falling back to cursor",
              flush=True)

    obj = bpy.data.objects.new(me.name.rsplit("_mesh", 1)[0], me)
    coll.objects.link(obj)
    obj.display_type = "WIRE"
    # Decompose to loc/rot/scale — setting matrix_world directly sometimes
    # doesn't survive the depsgraph flush a rigidbody.object_add op triggers.
    loc, rot, scale = world.decompose()
    obj.location = loc
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = rot
    obj.scale = scale
    obj["m2_phys_body_type"] = int(
        phys.BODY_DYNAMIC if is_active else phys.BODY_KINEMATIC)
    obj["_m2_phys_blender_shape"] = bshape

    # Root/collision bodies (passive) are bone-parented so they follow the
    # armature. Dynamic bodies stay UNPARENTED — the sim needs them free
    # to move, and bone-parenting would create a feedback loop with the
    # bone's Copy Transforms follow-constraint.
    if pb is not None and arm is not None and not is_active:
        _bone_parent_keeping_transform(obj, arm, pb)

    _apply_rigid_body(
        context, obj,
        kind="ACTIVE" if is_active else "PASSIVE",
        blender_shape=bshape,
        friction=0.5, restitution=0.0, mass=1.0,
        kinematic=not is_active,
    )

    # Record which bone this body corresponds to, for export + the wire-bone
    # step (the object itself is not bone-parented when dynamic, so we can't
    # read the bone from parent_bone).
    if pb is not None:
        obj["m2_phys_bone_name"] = pb.name

    # Select the new body so its per-object controls appear immediately.
    prev_mode = _enter_object_mode(context)
    try:
        _deselect_all(context)
        obj.select_set(True)
        context.view_layer.objects.active = obj
    finally:
        _restore_mode(context, prev_mode)
    return obj


def _add_weld(context, coll, body_a, body_b):
    """Fully-rigid WELD joint (Blender FIXED). Rare — use _add_point_joint
    for anything that should swing."""
    idx = sum(1 for o in coll.objects
              if o.get("m2_phys_joint_type") == int(phys.JOINT_WELD))
    empty = bpy.data.objects.new(f"phys_join_weld_{idx:02d}", None)
    empty.empty_display_type = "PLAIN_AXES"
    empty.empty_display_size = 0.02
    coll.objects.link(empty)
    empty["m2_phys_joint_type"] = int(phys.JOINT_WELD)
    a = body_a.matrix_world.translation
    b = body_b.matrix_world.translation
    empty.location = (a + b) * 0.5
    _apply_constraint(context, empty, "FIXED", body_a, body_b)
    return empty


def _add_spring_joint(context, coll, body_a, body_b, pivot_world: Vector,
                      lin_limit: float, ang_limit_deg: float,
                      stiffness: float, damping: float):
    """GENERIC_SPRING joint at pivot_world with small allowed offset and a
    spring that pulls back to rest — the physical setup for breast /
    belly / ponytail jiggle.

    IMPORTANT: Blender's GENERIC_SPRING rest is the CONSTRAINT EMPTY's
    origin (in the constraint's local frame). So the empty MUST sit at
    body_b's origin, with matching orientation — otherwise the spring
    pulls body_b AWAY from its initial pose toward wherever the empty is.

    Exports as SHOULDER (SHOJ) in the .phys since it's the closest kind
    WoW's Domino engine supports for spring-limited joints."""
    idx = sum(1 for o in coll.objects
              if o.get("m2_phys_joint_type") == int(phys.JOINT_SHOULDER))
    empty = bpy.data.objects.new(f"phys_join_spring_{idx:02d}", None)
    empty.empty_display_type = "PLAIN_AXES"
    empty.empty_display_size = 0.03
    coll.objects.link(empty)
    empty["m2_phys_joint_type"] = int(phys.JOINT_SHOULDER)
    # Align the constraint frame to body_b's world matrix — spring rest
    # is then body_b's initial pose exactly.
    empty.matrix_world = body_b.matrix_world.copy()
    empty["m2_phys_pivot_hint"] = list(pivot_world)  # kept for export

    _apply_constraint(context, empty, "GENERIC_SPRING", body_a, body_b)

    con = empty.rigid_body_constraint
    ang = math.radians(ang_limit_deg)
    # Translation: limit ± lin_limit around rest, spring pulls back.
    for axis in ("x", "y", "z"):
        setattr(con, f"use_limit_lin_{axis}", True)
        setattr(con, f"limit_lin_{axis}_lower", -lin_limit)
        setattr(con, f"limit_lin_{axis}_upper", +lin_limit)
        setattr(con, f"use_spring_{axis}", True)
        setattr(con, f"spring_stiffness_{axis}", stiffness)
        setattr(con, f"spring_damping_{axis}", damping)
    # Rotation: soft angular limits so it can wobble a bit.
    for axis in ("x", "y", "z"):
        setattr(con, f"use_limit_ang_{axis}", True)
        setattr(con, f"limit_ang_{axis}_lower", -ang)
        setattr(con, f"limit_ang_{axis}_upper", +ang)
        # Angular spring: added in Blender 3.0. Named differently across
        # versions — try each naming, log if none work.
        wired_ang = False
        for name_use, name_stf, name_dmp in (
            (f"use_spring_ang_{axis}",
             f"spring_stiffness_ang_{axis}",
             f"spring_damping_ang_{axis}"),
            # some builds use singular prefix
            (f"use_angular_spring_{axis}",
             f"angular_spring_stiffness_{axis}",
             f"angular_spring_damping_{axis}"),
        ):
            if hasattr(con, name_use):
                setattr(con, name_use, True)
                setattr(con, name_stf, stiffness * 0.5)
                setattr(con, name_dmp, damping)
                wired_ang = True
                break
        if not wired_ang:
            print(f"[m2phys] no angular spring API on this Blender build "
                  f"for axis {axis}; rotation will only be limited", flush=True)
    return empty


def _add_point_joint(context, coll, body_a, body_b, pivot_world: Vector):
    """POINT / spherical joint (free rotation, no translation) at
    `pivot_world`. Standard for cloth/hair/chain — bodies pivot freely
    about the joint anchor. Exports as SPHJ (SPHERICAL) in the .phys."""
    idx = sum(1 for o in coll.objects
              if o.get("m2_phys_joint_type") == int(phys.JOINT_SPHERICAL))
    empty = bpy.data.objects.new(f"phys_join_point_{idx:02d}", None)
    empty.empty_display_type = "PLAIN_AXES"
    empty.empty_display_size = 0.02
    coll.objects.link(empty)
    empty["m2_phys_joint_type"] = int(phys.JOINT_SPHERICAL)
    empty.location = pivot_world
    _apply_constraint(context, empty, "POINT", body_a, body_b)
    return empty


def _wire_bone_to_body(pose_bone, body_obj):
    """Make the bone rotate to match the physics body. Uses Damped Track
    so the bone tracks the body's origin — this is the standard bone-
    physics recipe and doesn't have the space-conversion ambiguities
    that Copy Rotation on bones can trip over. The body is built along
    the bone axis with its origin at the bone HEAD, so the natural
    'track' direction is +Y = bone axis. As the body swings, its origin
    stays roughly at the bone head, but its ROTATION shifts — Damped
    Track uses the body's rotated +Y as the target direction, giving the
    correct bone follow.

    Falls back to Copy Rotation if that turns out not to work in
    testing."""
    for c in list(pose_bone.constraints):
        if c.name.startswith("m2phys_"):
            pose_bone.constraints.remove(c)

    # Copy Rotation, POSE space owner — cleanest way to force the bone's
    # effective rotation to equal the body's world rotation while still
    # keeping the bone's head anchored to its parent bone.
    c = pose_bone.constraints.new("COPY_ROTATION")
    c.name = "m2phys_follow"
    c.target = body_obj
    c.owner_space = "POSE"
    c.target_space = "WORLD"
    c.influence = 1.0
    print(f"[m2phys] wired bone '{pose_bone.name}' Copy Rotation -> "
          f"body '{body_obj.name}' (owner=POSE, target=WORLD)", flush=True)
    return c


def _is_phys_body(obj) -> bool:
    if obj is None or obj.rigid_body is None:
        return False
    return any(c.name.endswith(PHYS_COLLECTION_SUFFIX)
               for c in obj.users_collection)


def _is_phys_joint(obj) -> bool:
    if obj is None or obj.rigid_body_constraint is None:
        return False
    return any(c.name.endswith(PHYS_COLLECTION_SUFFIX)
               for c in obj.users_collection)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class M2PHYS_OT_add_physics_mesh(Operator):
    bl_idname = "m2phys.add_physics_mesh"
    bl_label = "Add Physics Mesh"
    bl_description = ("Create a DYNAMIC body attached to the active pose bone "
                      "(falls, gets pushed around by the sim). If no bone is "
                      "active, drops one at the 3D cursor")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = _add_body_common(context, is_active=True)
        self.report({"INFO"}, "Added physics mesh '%s'" % obj.name)
        return {"FINISHED"}


class M2PHYS_OT_add_collision_mesh(Operator):
    bl_idname = "m2phys.add_collision_mesh"
    bl_label = "Add Collision Mesh"
    bl_description = ("Create a PASSIVE/kinematic body attached to the active "
                      "pose bone (drives collisions but is not moved by the "
                      "sim itself — great as an anchor or a bone-driven "
                      "collider that other bodies bounce off)")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = _add_body_common(context, is_active=False)
        self.report({"INFO"}, "Added collision mesh '%s'" % obj.name)
        return {"FINISHED"}


_PRESETS = {
    # Light, high air-drag, soft-ish so it drapes and settles.
    "CLOTH":  dict(mass=0.10, friction=0.70, restitution=0.00,
                   linear_damping=0.40, angular_damping=0.60,
                   joint="POINT"),
    # Heavier metal links, low damping so they swing freely; slight bounce.
    "CHAIN":  dict(mass=1.50, friction=0.40, restitution=0.05,
                   linear_damping=0.10, angular_damping=0.15,
                   joint="POINT"),
    # Standalone bouncy body: no joint, high restitution. Needs a
    # collider mesh (character body, ground) to bounce off.
    "BOUNCE": dict(mass=0.50, friction=0.20, restitution=0.90,
                   linear_damping=0.04, angular_damping=0.10,
                   joint="NONE"),
    # Breast / belly / ponytail jiggle: body springs back to rest with
    # small displacement, damped so it doesn't oscillate forever.
    # Gravity ON so the preview shows immediate motion (small ~5mm sag
    # at rest given the stiffness). If that sag annoys you, turn it off
    # via the body's Rigid Body panel — the spring alone will hold it.
    "JIGGLE": dict(mass=0.20, friction=0.30, restitution=0.00,
                   linear_damping=0.35, angular_damping=0.55,
                   use_gravity=True,
                   joint="SPRING",
                   spring_lin_limit=0.05,     # ±5cm translation
                   spring_ang_limit_deg=25.0, # ±25° rotation
                   spring_stiffness=800.0,    # snappy return (heavier vs gravity)
                   spring_damping=15.0),
}


def _apply_preset_to(obj, preset: dict):
    """Push preset numbers into obj.rigid_body AND the m2_phys_* mirror props."""
    if obj.rigid_body is None:
        return False
    rb = obj.rigid_body
    rb.mass = preset["mass"]
    rb.friction = preset["friction"]
    rb.restitution = preset["restitution"]
    rb.linear_damping = preset["linear_damping"]
    rb.angular_damping = preset["angular_damping"]
    # Some presets (JIGGLE) turn gravity off so the body rests exactly at
    # its parent's motion instead of sagging under gravity.
    rb.enabled = True
    try:
        rb.use_gravity = bool(preset.get("use_gravity", True))
    except AttributeError:
        pass
    obj["m2_phys_density"] = preset["mass"]
    obj["m2_phys_friction"] = preset["friction"]
    obj["m2_phys_restitution"] = preset["restitution"]
    return True


class M2PHYS_OT_bone_preset(Operator):
    """One-shot: select bone(s) → click preset → everything set up correctly.

    Adds a body on each selected pose bone (skipping bones that already
    have one), auto-welds each new body to the nearest ancestor bone's
    body, creates a root anchor on the top-most bone's parent if the rig
    has no root yet, and applies the preset numbers to every created or
    already-existing body on the selection.
    """
    bl_idname = "m2phys.bone_preset"
    bl_label = "Preset to Bone"
    bl_description = ("Select one or more pose bones and click a preset — the "
                      "capsules, root anchor, and welds are all created and "
                      "tuned in one go")
    bl_options = {"REGISTER", "UNDO"}

    preset: EnumProperty(
        items=[("CLOTH", "Cloth", ""),
               ("CHAIN", "Chain", ""),
               ("BOUNCE", "Bounce", ""),
               ("JIGGLE", "Jiggle", "")],
        default="CLOTH",
    )

    def execute(self, context):
        arm = _get_active_armature(context)
        if arm is None:
            self.report({"ERROR"}, "No active armature")
            return {"CANCELLED"}
        if context.mode != "POSE":
            self.report({"ERROR"}, "Enter Pose Mode and select at least one bone")
            return {"CANCELLED"}
        selected = list(context.selected_pose_bones or [])
        if not selected:
            self.report({"ERROR"}, "Select at least one pose bone")
            return {"CANCELLED"}

        _ensure_world(context)
        coll = _phys_collection(context, _model_name(context))
        by_bone = _index_bodies_by_bone(coll)

        # Sort selection by parent-chain depth (roots first) so auto-weld
        # can always find a parent body already-built.
        def depth(pb):
            n, cur = 0, pb.parent
            while cur is not None:
                n += 1
                cur = cur.parent
            return n
        ordered = sorted(selected, key=depth)

        # Ensure a root anchor. If none exists, put one on the parent of
        # the top-most selected bone (or on the top-most bone itself if
        # it has no parent).
        root = next((o for o in coll.objects if o.get("m2_phys_root")), None)
        if root is None:
            top_pb = ordered[0]
            root_pb = top_pb.parent if top_pb.parent is not None else top_pb
            root = _add_body_common(context, is_active=False, pose_bone=root_pb, arm=arm)
            root["m2_phys_root"] = True
            root["m2_phys_body_type"] = int(phys.BODY_ROOT)
            # Root is passive + kinematic (follows the bone).
            _apply_rigid_body(
                context, root, kind="PASSIVE",
                blender_shape=root.get("_m2_phys_blender_shape", "BOX"),
                friction=0.5, restitution=0.0, mass=1.0, kinematic=True,
            )
            by_bone = _index_bodies_by_bone(coll)

        # Create a body for every selected bone that doesn't have one.
        # POINT joint (spherical) placed at the bone HEAD connects each
        # body to the parent-chain body — this is what lets it swing.
        # Copy Transforms bone constraint makes the bone follow the body
        # in real time (no bake needed).
        created = []
        for pb in ordered:
            existing = by_bone.get(pb.name)
            if existing is not None:
                created.append(existing)
                continue
            body = _add_body_common(context, is_active=True, pose_bone=pb, arm=arm)
            created.append(body)
            by_bone[pb.name] = body

            # Find the closest ancestor with a body (skipping bones the
            # user didn't select — matches Blender's parent chain).
            parent_body = None
            cur = pb.parent
            while cur is not None and parent_body is None:
                parent_body = by_bone.get(cur.name)
                cur = cur.parent
            if parent_body is None:
                parent_body = root

            # Joint at bone HEAD (world, rest pose) — the natural pivot.
            # Preset chooses the constraint kind: POINT for chain/cloth,
            # GENERIC_SPRING for jiggle (springs back), none for bounce.
            head_world = arm.matrix_world @ pb.bone.head_local
            joint_kind = _PRESETS[self.preset].get("joint", "POINT")
            if parent_body is not None and joint_kind != "NONE":
                if joint_kind == "SPRING":
                    p = _PRESETS[self.preset]
                    _add_spring_joint(
                        context, coll, parent_body, body, head_world,
                        lin_limit=p["spring_lin_limit"],
                        ang_limit_deg=p["spring_ang_limit_deg"],
                        stiffness=p["spring_stiffness"],
                        damping=p["spring_damping"],
                    )
                else:
                    _add_point_joint(context, coll, parent_body, body, head_world)

            # Wire this bone to the body so the viewport shows motion.
            _wire_bone_to_body(pb, body)

        # Apply the tuned numbers to everything we just touched.
        p = _PRESETS[self.preset]
        for o in created:
            _apply_preset_to(o, p)

        self.report(
            {"INFO"},
            "%s: %d bodies (root '%s')"
            % (self.preset.title(), len(created),
               root.name if root is not None else "none"),
        )
        return {"FINISHED"}


class M2PHYS_OT_apply_preset(Operator):
    bl_idname = "m2phys.apply_preset"
    bl_label = "Apply Preset"
    bl_description = "Apply a tuned physics preset to every selected body"
    bl_options = {"REGISTER", "UNDO"}

    preset: EnumProperty(
        name="Preset",
        items=[
            ("CLOTH",  "Cloth",
             "Light, drapes and settles: mass 0.1, friction 0.7, damping 0.4/0.6"),
            ("CHAIN",  "Chain",
             "Metal chain link: mass 1.5, friction 0.4, low damping, slight bounce"),
            ("BOUNCE", "Bounce",
             "Bouncy ball: restitution 0.9, low friction, low damping"),
        ],
        default="CLOTH",
    )

    def execute(self, context):
        bodies = [o for o in context.selected_objects if _is_phys_body(o)]
        if not bodies:
            self.report({"ERROR"}, "Select at least one physics body")
            return {"CANCELLED"}
        p = _PRESETS[self.preset]
        n = 0
        for o in bodies:
            if _apply_preset_to(o, p):
                n += 1
        self.report({"INFO"}, "%s preset applied to %d body(ies)"
                    % (self.preset.title(), n))
        return {"FINISHED"}


class M2PHYS_OT_make_collider(Operator):
    """Turn each selected mesh object into a PASSIVE physics collider so
    the sim's dynamic bodies actually bounce off it. Use for:
      • the character mesh itself (add a Convex Hull for whole-body collision)
      • a ground plane
      • static props / environment meshes

    The mesh is NOT moved into the phys collection — it stays in its home
    collection. Set collision_shape to MESH (exact geometry, slower) or
    CONVEX_HULL (fast, approximate)."""
    bl_idname = "m2phys.make_collider"
    bl_label = "Make Collider"
    bl_description = ("Add a PASSIVE Rigid Body to every selected mesh so "
                      "physics capsules collide with it. Use Convex Hull for "
                      "speed, or Mesh for exact geometry")
    bl_options = {"REGISTER", "UNDO"}

    shape: EnumProperty(
        name="Collision Shape",
        items=[
            ("CONVEX_HULL", "Convex Hull",
             "Fast, wraps the mesh in a convex volume — good default"),
            ("MESH", "Mesh (exact)",
             "Uses the actual mesh geometry — slower but exact"),
        ],
        default="CONVEX_HULL",
    )

    def execute(self, context):
        _ensure_world(context)
        meshes = [o for o in context.selected_objects
                  if o.type == "MESH" and not _is_phys_body(o)]
        if not meshes:
            self.report({"ERROR"}, "Select at least one mesh object "
                                    "(not already a physics body)")
            return {"CANCELLED"}
        for obj in meshes:
            _apply_rigid_body(
                context, obj, kind="PASSIVE",
                blender_shape=self.shape,
                friction=0.5, restitution=0.0, mass=1.0,
                kinematic=True,   # animate/pose freely; sim still collides
            )
            obj["m2_phys_collider"] = True
        self.report({"INFO"}, "Made %d mesh(es) into collider(s) (%s)"
                    % (len(meshes), self.shape))
        return {"FINISHED"}


class M2PHYS_OT_unmake_collider(Operator):
    bl_idname = "m2phys.unmake_collider"
    bl_label = "Remove Collider"
    bl_description = "Strip the collider Rigid Body from selected meshes"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        n = 0
        prev_mode = _enter_object_mode(context)
        try:
            for obj in context.selected_objects:
                if obj.get("m2_phys_collider") and obj.rigid_body is not None:
                    _deselect_all(context)
                    obj.select_set(True)
                    context.view_layer.objects.active = obj
                    try:
                        bpy.ops.rigidbody.object_remove()
                    except RuntimeError:
                        pass
                    try:
                        del obj["m2_phys_collider"]
                    except KeyError:
                        pass
                    n += 1
        finally:
            _restore_mode(context, prev_mode)
        self.report({"INFO"}, "Removed collider from %d mesh(es)" % n)
        return {"FINISHED"}


class M2PHYS_OT_connect_selected(Operator):
    bl_idname = "m2phys.connect_selected"
    bl_label = "Connect Selected (Weld)"
    bl_description = "Weld the two selected physics bodies together"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        sel = [o for o in context.selected_objects if _is_phys_body(o)]
        if len(sel) != 2:
            self.report({"ERROR"}, "Select exactly two physics bodies")
            return {"CANCELLED"}
        coll = next(
            (c for o in sel for c in o.users_collection
             if c.name.endswith(PHYS_COLLECTION_SUFFIX)),
            _phys_collection(context, _model_name(context)),
        )
        _add_weld(context, coll, sel[0], sel[1])
        self.report({"INFO"}, "Welded %s <-> %s" % (sel[0].name, sel[1].name))
        return {"FINISHED"}


class M2PHYS_OT_mark_root(Operator):
    bl_idname = "m2phys.mark_root"
    bl_label = "Mark as Root Anchor"
    bl_description = ("Turn the active body into the phys ROOT anchor "
                      "(PASSIVE + kinematic; only ONE per rig)")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = context.active_object
        if not _is_phys_body(obj):
            self.report({"ERROR"}, "Select a physics body first")
            return {"CANCELLED"}
        # Clear existing roots in the same phys collection.
        for c in obj.users_collection:
            if c.name.endswith(PHYS_COLLECTION_SUFFIX):
                for other in c.objects:
                    if other is not obj:
                        try:
                            del other["m2_phys_root"]
                        except KeyError:
                            pass
        obj["m2_phys_root"] = True
        obj["m2_phys_body_type"] = int(phys.BODY_ROOT)
        _apply_rigid_body(
            context, obj, kind="PASSIVE",
            blender_shape=obj.get("_m2_phys_blender_shape", "BOX"),
            friction=obj.rigid_body.friction,
            restitution=obj.rigid_body.restitution,
            mass=obj.rigid_body.mass,
            kinematic=True,
        )
        self.report({"INFO"}, "Marked '%s' as root" % obj.name)
        return {"FINISHED"}


class M2PHYS_OT_delete_body(Operator):
    bl_idname = "m2phys.delete_body"
    bl_label = "Delete Selected Physics Object"
    bl_description = "Remove the selected body / joint from the phys rig"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        removed = 0
        for obj in list(context.selected_objects):
            if _is_phys_body(obj) or _is_phys_joint(obj):
                bpy.data.objects.remove(obj, do_unlink=True)
                removed += 1
        self.report({"INFO"}, "Removed %d object(s)" % removed)
        return {"FINISHED"}


class M2PHYS_OT_preview_bake(Operator):
    bl_idname = "m2phys.preview_bake"
    bl_label = "Bake Preview & Wire Bones"
    bl_description = ("Bake the rigid-body sim to keyframes across the frame "
                      "range, then add Copy Transforms bone constraints so "
                      "bones follow the baked physics. Use Rebuild Physics "
                      "afterwards to keep editing the setup")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.m2_physics
        arm = _get_active_armature(context)
        if arm is None:
            self.report({"ERROR"}, "Need an active armature")
            return {"CANCELLED"}
        rbw = context.scene.rigidbody_world
        if rbw is None:
            self.report({"ERROR"}, "No Rigid Body World — add a body first")
            return {"CANCELLED"}
        coll = _find_phys_collection(context)
        if coll is None:
            self.report({"ERROR"}, "No phys collection")
            return {"CANCELLED"}

        bodies = [o for o in coll.objects
                  if o.rigid_body is not None and o.rigid_body.type == "ACTIVE"]
        if not bodies:
            self.report({"WARNING"}, "No ACTIVE bodies to bake")
            return {"CANCELLED"}

        # Break bone-parent-driven depsgraph cycles before the bake.
        # Cycle shape (from the .phys import path + a preset that adds
        # a Copy Rotation constraint on a bone):
        #   Sim → active body → bone parent → constrained bone → body
        # Unparenting an ACTIVE body while KEEPING its world transform
        # cuts the loop at the "bone parent" edge. We stash the bone
        # name on the object so the .phys exporter still knows which
        # bone this body corresponds to.
        broken = 0
        for o in bodies:
            if o.parent_type != "BONE" or not o.parent_bone:
                continue
            bone_name = o.parent_bone
            world = o.matrix_world.copy()
            o["m2_phys_bone_name"] = bone_name
            o.parent = None
            o.matrix_world = world
            broken += 1
        if broken:
            print(f"[m2phys] preview_bake: unparented {broken} active "
                  f"body(ies) from their bones to break sim cycles "
                  f"(bone name preserved via m2_phys_bone_name prop)",
                  flush=True)

        rbw.point_cache.frame_start = int(props.preview_start)
        rbw.point_cache.frame_end = int(props.preview_end)
        context.scene.frame_start = int(props.preview_start)
        context.scene.frame_end = int(props.preview_end)

        prev_mode = _enter_object_mode(context)
        try:
            _deselect_all(context)
            for o in bodies:
                o.select_set(True)
            context.view_layer.objects.active = bodies[0]
            try:
                bpy.ops.rigidbody.bake_to_keyframes(
                    frame_start=int(props.preview_start),
                    frame_end=int(props.preview_end),
                    step=1,
                )
            except RuntimeError as exc:
                self.report({"ERROR"}, "bake_to_keyframes failed: %s" % exc)
                return {"CANCELLED"}
        finally:
            _restore_mode(context, prev_mode)

        wired = 0
        for body in bodies:
            # Prefer live bone-parenting; fall back to the imported /
            # cycle-broken bodies' m2_phys_bone_name stash.
            if body.parent_type == "BONE" and body.parent_bone:
                bname = body.parent_bone
            else:
                bname = str(body.get("m2_phys_bone_name", "") or "")
            if not bname:
                continue
            pb = arm.pose.bones.get(bname)
            if pb is None:
                continue
            for c in list(pb.constraints):
                if c.name.startswith("m2phys_preview"):
                    pb.constraints.remove(c)
            c = pb.constraints.new("COPY_TRANSFORMS")
            c.name = "m2phys_preview"
            c.target = body
            c.influence = 1.0
            wired += 1

        self.report({"INFO"}, "Baked %d bodies, wired %d bones" % (len(bodies), wired))
        return {"FINISHED"}


class M2PHYS_OT_preview_clear(Operator):
    bl_idname = "m2phys.preview_clear"
    bl_label = "Clear Preview"
    bl_description = "Remove Copy Transforms bone constraints from preview bake"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        arm = _get_active_armature(context)
        if arm is None:
            self.report({"ERROR"}, "No armature")
            return {"CANCELLED"}
        n = 0
        for pb in arm.pose.bones:
            for c in list(pb.constraints):
                if c.name.startswith("m2phys_preview"):
                    pb.constraints.remove(c)
                    n += 1
        self.report({"INFO"}, "Removed %d preview constraint(s)" % n)
        return {"FINISHED"}


class M2PHYS_OT_rebuild(Operator):
    bl_idname = "m2phys.rebuild"
    bl_label = "Rebuild Physics from Data"
    bl_description = ("Restore Rigid Body settings on every object in the "
                      "phys collection from the stored m2_phys_* custom "
                      "properties (undoes a preview bake so you can keep "
                      "editing)")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        coll = _find_phys_collection(context)
        if coll is None:
            self.report({"ERROR"}, "No phys collection")
            return {"CANCELLED"}
        _ensure_world(context)
        n = 0
        for obj in coll.objects:
            if obj.data is None:
                continue  # joint empties
            body_type = int(obj.get("m2_phys_body_type", phys.BODY_DYNAMIC))
            is_root = bool(obj.get("m2_phys_root", False))
            shape = obj.get("_m2_phys_blender_shape", "BOX")
            friction = float(obj.get("m2_phys_friction", 0.5))
            rest = float(obj.get("m2_phys_restitution", 0.0))
            density = float(obj.get("m2_phys_density", 1.0))
            _apply_rigid_body(
                context, obj,
                kind=("PASSIVE" if is_root or body_type != phys.BODY_DYNAMIC
                      else "ACTIVE"),
                blender_shape=shape,
                friction=friction, restitution=rest, mass=density,
                kinematic=is_root,
            )
            n += 1
        self.report({"INFO"}, "Rebuilt %d body(ies)" % n)
        return {"FINISHED"}


class M2PHYS_OT_remove_all(Operator):
    bl_idname = "m2phys.remove_all"
    bl_label = "Remove All Physics"
    bl_description = "Delete the entire phys collection and everything inside"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        coll = _find_phys_collection(context)
        if coll is None:
            self.report({"INFO"}, "Nothing to remove")
            return {"CANCELLED"}
        for o in list(coll.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.collections.remove(coll)
        self.report({"INFO"}, "All physics removed")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Panel — content depends on what's selected
# ---------------------------------------------------------------------------

class M2PHYS_OT_reveal_rig_collection(Operator):
    """Un-hide a phys collection and select its members so the user can
    actually see the imported rig in the viewport. Fixes 'the import
    said it worked but I don't see anything' by removing the two
    common causes: collection is view-layer excluded, or its members
    are hide-viewport'd."""

    bl_idname = "m2phys.reveal_rig_collection"
    bl_label = "Reveal Rig"
    bl_description = ("Unhide the selected phys collection + all its members, "
                      "select them, and frame the viewport on the rig.")
    bl_options = {"REGISTER", "UNDO"}

    collection_name: bpy.props.StringProperty()

    def execute(self, context):
        coll = bpy.data.collections.get(self.collection_name)
        if coll is None:
            self.report({"WARNING"}, f"Collection {self.collection_name!r} not found")
            return {"CANCELLED"}
        coll.hide_viewport = False
        coll.hide_select = False
        # Un-exclude from the active view layer if it's excluded.
        vl = context.view_layer
        def _find(layer_coll, target):
            if layer_coll.collection == target:
                return layer_coll
            for ch in layer_coll.children:
                r = _find(ch, target)
                if r is not None:
                    return r
            return None
        lc = _find(vl.layer_collection, coll)
        if lc is not None:
            lc.exclude = False
            lc.hide_viewport = False
        # Un-hide + select every object in the collection.
        for obj in coll.objects:
            obj.hide_viewport = False
            obj.hide_set(False)
            obj.select_set(True)
        # Set an active object so the panel's per-body / per-joint sections
        # can start showing something useful.
        if coll.objects:
            context.view_layer.objects.active = coll.objects[0]
        # Frame the viewport on the selection if we can.
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                for region in area.regions:
                    if region.type == "WINDOW":
                        with context.temp_override(area=area, region=region):
                            try:
                                bpy.ops.view3d.view_selected(use_all_regions=False)
                            except RuntimeError:
                                pass
                break
        self.report({"INFO"},
                    f"Revealed {coll.name} ({len(coll.objects)} object(s))")
        return {"FINISHED"}


class VIEW3D_PT_m2_physics(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "M2"
    bl_label = "M2 Physics"
    bl_idname = "VIEW3D_PT_m2_physics"

    def draw(self, context):
        layout = self.layout
        props = context.scene.m2_physics

        # --- Imported / detected rig status ------------------------------
        # Tells the user at a glance whether a .phys was loaded alongside
        # the .m2 (the M2 loader auto-calls load_phys_into_scene) and how
        # many bodies/joints landed in the scene. Without this box the
        # only signal was "there's a `<name>_phys` collection in the
        # outliner" — easy to miss when the collection is collapsed.
        self._draw_rig_status(layout, context)

        # --- One-click bone preset (the primary workflow) ----------------
        n_sel_bones = (len(context.selected_pose_bones)
                       if context.mode == "POSE" and context.selected_pose_bones
                       else 0)
        box = layout.box()
        box.label(text="Preset to Selected Bone(s)", icon="BONE_DATA")
        grid = box.column(align=True)
        grid.enabled = n_sel_bones > 0
        row = grid.row(align=True)
        row.operator("m2phys.bone_preset", text="Cloth", icon="MOD_CLOTH").preset = "CLOTH"
        row.operator("m2phys.bone_preset", text="Chain", icon="LINKED").preset = "CHAIN"
        row = grid.row(align=True)
        row.operator("m2phys.bone_preset", text="Jiggle", icon="FORCE_HARMONIC").preset = "JIGGLE"
        row.operator("m2phys.bone_preset", text="Bounce", icon="FORCE_FORCE").preset = "BOUNCE"
        if n_sel_bones == 0:
            box.label(text="→ enter Pose Mode + select bones",
                      icon="INFO")
        else:
            box.label(text=f"{n_sel_bones} bone(s) selected",
                      icon="CHECKMARK")

        layout.separator()

        # --- Colliders on existing meshes --------------------------------
        box = layout.box()
        box.label(text="Selected mesh → collider:", icon="MOD_PHYSICS")
        row = box.row(align=True)
        row.operator("m2phys.make_collider", text="Make Collider (Hull)"
                     ).shape = "CONVEX_HULL"
        row = box.row(align=True)
        row.operator("m2phys.make_collider", text="Exact Mesh Collider"
                     ).shape = "MESH"
        row.operator("m2phys.unmake_collider", text="", icon="X")

        layout.separator()

        # --- Manual add-mesh (advanced) ----------------------------------
        col = layout.column(align=True)
        col.label(text="Manual Add:")
        row = col.row(align=True)
        row.prop(props, "shape_type", text="")
        row.prop(props, "default_radius", text="Size")
        row = col.row(align=True)
        row.operator("m2phys.add_physics_mesh", icon="RIGID_BODY", text="Physics Mesh")
        row.operator("m2phys.add_collision_mesh", icon="MOD_PHYSICS", text="Collision Mesh")

        active = context.active_object

        # --- Selected body: per-object controls --------------------------
        if _is_phys_body(active):
            layout.separator()
            box = layout.box()
            box.label(text=f"Body: {active.name}", icon="RIGID_BODY")
            self._draw_body_controls(box, context, active)

        # --- Selected joint: per-joint controls --------------------------
        elif _is_phys_joint(active):
            layout.separator()
            box = layout.box()
            box.label(text=f"Joint: {active.name}", icon="CONSTRAINT")
            self._draw_joint_controls(box, context, active)

        # --- Pose-mode bone hint -----------------------------------------
        elif context.mode == "POSE" and context.active_pose_bone:
            layout.separator()
            box = layout.box()
            box.label(text=f"Active bone: {context.active_pose_bone.name}",
                      icon="BONE_DATA")
            box.label(text="Click 'Add Physics/Collision Mesh' above",
                      icon="INFO")

        # --- Global status + connect + preview + remove ------------------
        layout.separator()
        row = layout.row(align=True)
        row.operator("m2phys.connect_selected", icon="LINKED")
        row.operator("m2phys.delete_body", icon="X", text="")

        layout.separator()
        box = layout.box()
        box.label(text="Preview (bake sim -> bones)")
        row = box.row(align=True)
        row.prop(props, "preview_start")
        row.prop(props, "preview_end")
        col = box.column(align=True)
        col.operator("m2phys.preview_bake", icon="REC")
        col.operator("m2phys.preview_clear", icon="CANCEL")
        col.operator("m2phys.rebuild", icon="FILE_REFRESH")

        layout.separator()
        layout.operator("m2phys.remove_all", icon="TRASH")

    # --- helpers -----------------------------------------------------

    def _draw_rig_status(self, layout, context):
        """Show which phys collections exist, how many bodies/joints each
        contains, and offer quick reveal/hide + jump-in buttons. Helps
        the user notice that an imported .phys is present but its
        collection is currently collapsed/hidden."""
        rig_colls = [c for c in bpy.data.collections
                     if c.name.endswith(PHYS_COLLECTION_SUFFIX)]
        if not rig_colls:
            box = layout.box()
            row = box.row()
            row.label(text="No phys rig loaded", icon="INFO")
            return
        box = layout.box()
        box.label(text="Detected Phys Rigs:", icon="RIGID_BODY")
        for c in rig_colls:
            n_bodies = sum(1 for o in c.objects if o.rigid_body is not None)
            n_joints = sum(1 for o in c.objects
                           if o.rigid_body_constraint is not None)
            row = box.row(align=True)
            row.label(
                text=f"{c.name}  ({n_bodies} bodies, {n_joints} joints)")
            # Reveal-in-outliner + select-first-body button.
            op = row.operator("m2phys.reveal_rig_collection", text="", icon="RESTRICT_SELECT_OFF")
            op.collection_name = c.name

    def _draw_body_controls(self, layout, context, obj):
        rb = obj.rigid_body
        # Presets — one-click tuning for the two or three shapes people
        # actually build (soft cloth, swingy chain, bouncy ball).
        n_selected = sum(1 for o in context.selected_objects if _is_phys_body(o))
        preset_row = layout.row(align=True)
        preset_row.label(text=f"Preset (applies to {n_selected}):", icon="PRESET")
        row = layout.row(align=True)
        row.operator("m2phys.apply_preset", text="Cloth", icon="MOD_CLOTH").preset = "CLOTH"
        row.operator("m2phys.apply_preset", text="Chain", icon="LINKED").preset = "CHAIN"
        row.operator("m2phys.apply_preset", text="Bounce", icon="FORCE_FORCE").preset = "BOUNCE"
        layout.separator()

        col = layout.column(align=True)
        col.prop(rb, "type", text="Kind")
        col.prop(rb, "collision_shape", text="Shape")
        col.prop(rb, "mass")
        col.prop(rb, "friction")
        col.prop(rb, "restitution")
        col.prop(rb, "linear_damping", text="Linear Damping")
        col.prop(rb, "angular_damping", text="Angular Damping")
        col.prop(rb, "kinematic", text="Kinematic (follow parent transform)")

        # Bone parent info (read-only display + bone picker)
        arm = _get_active_armature(context)
        if arm is None:
            arm = obj.parent if (obj.parent and obj.parent.type == "ARMATURE") else None
        if arm is not None:
            row = layout.row(align=True)
            row.prop_search(obj, "parent_bone", arm.data, "bones",
                            text="Bone")

        # Root toggle + delete
        row = layout.row(align=True)
        if obj.get("m2_phys_root"):
            row.label(text="ROOT ANCHOR", icon="PINNED")
        else:
            row.operator("m2phys.mark_root", icon="PINNED",
                         text="Set as Root Anchor")
        row.operator("m2phys.delete_body", icon="TRASH", text="")

    def _draw_joint_controls(self, layout, context, obj):
        con = obj.rigid_body_constraint
        col = layout.column(align=True)
        col.prop(con, "type", text="Type")
        col.prop(con, "enabled")
        col.prop(con, "object1", text="Body A")
        col.prop(con, "object2", text="Body B")
        if con.type in {"GENERIC", "GENERIC_SPRING"}:
            layout.label(text="Limits / spring live in Blender's Physics tab",
                         icon="INFO")
        layout.operator("m2phys.delete_body", icon="TRASH", text="Delete Joint")


# ---------------------------------------------------------------------------
_classes = (
    M2PhysicsProps,
    M2PHYS_OT_add_physics_mesh,
    M2PHYS_OT_add_collision_mesh,
    M2PHYS_OT_bone_preset,
    M2PHYS_OT_apply_preset,
    M2PHYS_OT_make_collider,
    M2PHYS_OT_unmake_collider,
    M2PHYS_OT_connect_selected,
    M2PHYS_OT_mark_root,
    M2PHYS_OT_delete_body,
    M2PHYS_OT_preview_bake,
    M2PHYS_OT_preview_clear,
    M2PHYS_OT_rebuild,
    M2PHYS_OT_remove_all,
    M2PHYS_OT_reveal_rig_collection,
    VIEW3D_PT_m2_physics,
)


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.m2_physics = PointerProperty(type=M2PhysicsProps)


def unregister():
    try:
        del bpy.types.Scene.m2_physics
    except Exception:  # noqa: BLE001
        pass
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
