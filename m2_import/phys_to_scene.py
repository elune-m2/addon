"""Rebuild Blender's rigid-body world from a .phys file.

Inverse of `phys_from_scene`. On import we look for a `<name>.phys`
sibling next to the .m2 and, if found, materialize each Body as a
primitive (cube/sphere/cylinder — matching its shape) parented to the
right bone, then wire up Rigid Body settings and constraint empties for
each Joint. The user can then tweak shapes/joints in Blender and
re-export to a byte-similar .phys.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import bpy
from mathutils import Matrix, Vector

from . import phys


_PHYS_COLLECTION_SUFFIX = "_phys"


def _bone_name_by_m2_index(arm_obj) -> List[str]:
    """M2-index → bone name. Same topological order as the exporter."""
    from . import from_scene
    return [b.name for b in from_scene._topo_bones(arm_obj.data)]


def _ensure_phys_collection(context, name: str):
    scene = context.scene
    coll_name = name + _PHYS_COLLECTION_SUFFIX
    coll = bpy.data.collections.get(coll_name)
    if coll is None:
        coll = bpy.data.collections.new(coll_name)
        scene.collection.children.link(coll)
    return coll


def _ensure_rigid_body_world(context):
    if context.scene.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()
    if context.scene.rigidbody_world.collection is None:
        rb_coll = bpy.data.collections.new("RigidBodyWorld")
        context.scene.rigidbody_world.collection = rb_coll


def _wow_to_blender_vec(v, mirror_x: bool) -> Vector:
    return Vector((-v[0], v[1], v[2])) if mirror_x else Vector(v)


def _parent_to_bone(obj, arm_obj, bone_name: str, local_pos: Vector):
    """Parent `obj` to a bone; put it at the given local offset from bone head."""
    obj.parent = arm_obj
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    # Blender's BONE parenting places the child at the bone's TAIL by default,
    # then applies matrix_parent_inverse. Set matrix_world so the object ends
    # up at bone.head + local_pos in armature space.
    bone = arm_obj.data.bones.get(bone_name)
    if bone is None:
        obj.matrix_world = arm_obj.matrix_world @ Matrix.Translation(local_pos)
        return
    bone_world = arm_obj.matrix_world @ bone.matrix_local
    obj.matrix_world = bone_world @ Matrix.Translation(local_pos)


def _make_shape_object(doc: phys.PhysDoc, shape_slot: int, name: str,
                       coll) -> Optional[object]:
    """Create a primitive mesh sized to match the shape at doc.shapes[slot]."""
    if shape_slot < 0 or shape_slot >= len(doc.shapes):
        return None
    s = doc.shapes[shape_slot]

    if s.shape_type == phys.SHAPE_BOX and s.shape_index < len(doc.boxes):
        bx = doc.boxes[s.shape_index]
        hx, hy, hz = bx.half_extents
        # Build a cube mesh with matching dimensions directly, no ops call.
        me = bpy.data.meshes.new(name + "_mesh")
        verts = [(x*hx, y*hy, z*hz)
                 for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        faces = [(0,1,3,2),(4,5,7,6),(0,1,5,4),(2,3,7,6),(0,2,6,4),(1,3,7,5)]
        me.from_pydata(verts, [], faces)
        me.update()
        obj = bpy.data.objects.new(name, me)
        coll.objects.link(obj)
        blender_shape = "BOX"
    elif s.shape_type == phys.SHAPE_SPHERE and s.shape_index < len(doc.spheres):
        sp = doc.spheres[s.shape_index]
        r = sp.radius
        me = _icosphere_mesh(name + "_mesh", r, subdivisions=2)
        obj = bpy.data.objects.new(name, me)
        coll.objects.link(obj)
        blender_shape = "SPHERE"
    elif s.shape_type == phys.SHAPE_CAPSULE and s.shape_index < len(doc.capsules):
        c = doc.capsules[s.shape_index]
        p1, p2 = Vector(c.p1), Vector(c.p2)
        axis = p2 - p1
        length = axis.length
        radius = c.radius
        me = _cylinder_mesh(name + "_mesh", radius, length + 2*radius, segments=12)
        obj = bpy.data.objects.new(name, me)
        coll.objects.link(obj)
        blender_shape = "CAPSULE"
        # Local origin is midpoint of p1/p2 in body space; cylinder built
        # along +Z. If p1/p2 aren't Z-aligned, the round-trip loses a
        # rotation — accept for now; retail samples are all Z-aligned.
        mid = (p1 + p2) * 0.5
        obj.location = mid
    else:
        return None

    obj["m2_phys_shape_type"] = int(s.shape_type)
    obj["m2_phys_shape_index"] = int(s.shape_index)
    obj["m2_phys_friction"] = float(s.friction)
    obj["m2_phys_restitution"] = float(s.restitution)
    obj["m2_phys_density"] = float(s.density)
    # SHAP has a 4-byte "unk" field the sample .phys uses non-zero. Stash
    # it as hex so the exporter can round-trip the exact byte pattern
    # rather than emitting zero — otherwise re-export != original file.
    try:
        obj["m2_phys_shape_unk"] = bytes(s.unk).hex()
    except Exception:  # noqa: BLE001 — defensive: writer always supplies 4 bytes
        obj["m2_phys_shape_unk"] = "00000000"
    obj["_m2_phys_blender_shape"] = blender_shape
    return obj


def _icosphere_mesh(name: str, radius: float, subdivisions: int = 2):
    """Build a simple UV sphere mesh without bpy.ops."""
    import math
    me = bpy.data.meshes.new(name)
    rings = max(6, subdivisions * 4)
    segments = max(8, subdivisions * 6)
    verts = []
    faces = []
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


def _cylinder_mesh(name: str, radius: float, height: float, segments: int = 12):
    import math
    me = bpy.data.meshes.new(name)
    verts = []
    h2 = height * 0.5
    for s in range(segments):
        a = 2 * math.pi * s / segments
        x, y = math.cos(a) * radius, math.sin(a) * radius
        verts.append((x, y, -h2))
        verts.append((x, y, +h2))
    faces = []
    for s in range(segments):
        bl = 2 * s
        tl = 2 * s + 1
        br = 2 * ((s + 1) % segments)
        tr = 2 * ((s + 1) % segments) + 1
        faces.append((bl, br, tr, tl))
    # Caps
    bottom = [2*s for s in range(segments)]
    top = [2*s + 1 for s in range(segments)]
    faces.append(tuple(reversed(bottom)))
    faces.append(tuple(top))
    me.from_pydata(verts, [], faces)
    me.update()
    return me


def _apply_rigid_body(context, obj, is_active: bool, blender_shape: str,
                      friction: float, restitution: float, density: float,
                      kinematic: bool = False):
    """Add the object to the rigid body world with matching settings.

    kinematic=True gives a PASSIVE body that follows its parent-transform
    (bone parenting, animation) instead of the physics sim — the correct
    setup for a WoW phys root, which anchors the whole chain to a bone.
    """
    context.view_layer.objects.active = obj
    obj.select_set(True)
    if obj.rigid_body is None:
        try:
            bpy.ops.rigidbody.object_add(type="ACTIVE" if is_active else "PASSIVE")
        except RuntimeError:
            return
    else:
        obj.rigid_body.type = "ACTIVE" if is_active else "PASSIVE"
    obj.rigid_body.collision_shape = blender_shape
    obj.rigid_body.friction = friction
    obj.rigid_body.restitution = restitution
    obj.rigid_body.mass = max(density, 0.001)
    if kinematic:
        obj.rigid_body.kinematic = True
    obj.select_set(False)


def _apply_constraint(context, empty, con_type: str, obj_a, obj_b):
    context.view_layer.objects.active = empty
    empty.select_set(True)
    if empty.rigid_body_constraint is None:
        try:
            bpy.ops.rigidbody.constraint_add(type=con_type)
        except RuntimeError:
            return
    else:
        empty.rigid_body_constraint.type = con_type
    con = empty.rigid_body_constraint
    con.enabled = True
    con.object1 = obj_a
    con.object2 = obj_b
    empty.select_set(False)


def _joint_to_blender_type(joint_type: int) -> str:
    return {
        phys.JOINT_WELD: "FIXED",
        phys.JOINT_SPHERICAL: "POINT",
        phys.JOINT_SHOULDER: "GENERIC",
        phys.JOINT_REVOLUTE: "HINGE",
        phys.JOINT_PRISMATIC: "SLIDER",
        phys.JOINT_DISTANCE: "GENERIC_SPRING",
    }.get(joint_type, "FIXED")


def _discover_phys(m2_path: str, explicit_path: str = "") -> Optional[str]:
    """Pick a .phys file for this .m2.

    Priority: user-supplied path → `<stem>.phys` sibling → any single
    .phys in the same folder (helpful when the sample dump has a phys
    file with a different stem than the .m2, but only one candidate).
    Returns None if nothing suitable is found.
    """
    if explicit_path:
        if os.path.isfile(explicit_path):
            return explicit_path
        print("[phys-import] explicit path not found: " + explicit_path,
              flush=True)
    stem = os.path.splitext(m2_path)[0]
    sib = stem + ".phys"
    if os.path.isfile(sib):
        return sib
    folder = os.path.dirname(m2_path) or "."
    candidates = [f for f in os.listdir(folder) if f.lower().endswith(".phys")]
    if len(candidates) == 1:
        p = os.path.join(folder, candidates[0])
        print("[phys-import] no stem match; using the only .phys in the "
              "folder: " + candidates[0], flush=True)
        return p
    if candidates:
        print("[phys-import] no stem match; multiple .phys files present "
              "(%s) — pick one via 'Phys File Override' or rename it to "
              "'%s.phys'" % (candidates, os.path.basename(stem)),
              flush=True)
    return None


def load_phys_into_scene(context, m2_path: str, arm_obj, model_name: str,
                         mirror_x: bool = False,
                         explicit_path: str = "") -> Optional[str]:
    """If a matching .phys exists, rebuild the rigid-body world for it.

    Returns the .phys path on success, None if there is nothing to load.
    """
    phys_path = _discover_phys(m2_path, explicit_path)
    if phys_path is None:
        return None

    if arm_obj is None:
        print("[phys-import] no armature to parent physics onto: skipping "
              + phys_path, flush=True)
        return None

    try:
        with open(phys_path, "rb") as f:
            doc = phys.read(f.read())
    except Exception as exc:  # noqa: BLE001
        print("[phys-import] read failed on %s: %s" % (phys_path, exc), flush=True)
        return None

    if not doc.bodies:
        print("[phys-import] %s has no bodies: nothing to do" % phys_path,
              flush=True)
        return None

    _ensure_rigid_body_world(context)
    coll = _ensure_phys_collection(context, model_name)

    bone_names = _bone_name_by_m2_index(arm_obj)

    # Two passes so we know all objects before wiring joints.
    body_objs: List[object] = []
    for i, body in enumerate(doc.bodies):
        bone_name = bone_names[body.bone_index] if body.bone_index < len(bone_names) else ""
        obj_name = f"{model_name}_phys_body{i:02d}"

        is_root = body.type == phys.BODY_ROOT
        if body.shapes_count > 0 and body.shapes_base < len(doc.shapes):
            obj = _make_shape_object(doc, body.shapes_base,
                                     obj_name, coll)
        else:
            obj = None

        if obj is None:
            # Root / shapeless body: give it a tiny cube mesh so it can
            # carry a Rigid Body — constraints need both sides to be
            # rigid bodies, and Blender empties don't participate.
            me = bpy.data.meshes.new(obj_name + "_mesh")
            r = 0.03
            verts = [(x*r, y*r, z*r) for x in (-1,1) for y in (-1,1) for z in (-1,1)]
            faces = [(0,1,3,2),(4,5,7,6),(0,1,5,4),(2,3,7,6),(0,2,6,4),(1,3,7,5)]
            me.from_pydata(verts, [], faces)
            me.update()
            obj = bpy.data.objects.new(obj_name, me)
            coll.objects.link(obj)
            obj["_m2_phys_blender_shape"] = "BOX"

        obj.display_type = "WIRE"
        obj["m2_phys_body_type"] = int(body.type)
        obj["m2_phys_bone_index"] = int(body.bone_index)
        if is_root:
            obj["m2_phys_root"] = True

        pos = _wow_to_blender_vec(body.position, mirror_x)
        is_dynamic = (not is_root and body.type == phys.BODY_DYNAMIC)
        if bone_name and not is_dynamic:
            # Kinematic / root: bone-parenting is safe because these
            # bodies don't feed back into the sim through a bone
            # constraint.
            _parent_to_bone(obj, arm_obj, bone_name, pos)
        elif bone_name:
            # Dynamic body — DO NOT bone-parent. If a preset later adds
            # a Copy Rotation constraint on this bone (or any ancestor)
            # that reads the sim result, the parent-child chain would
            # form a depsgraph cycle:
            #   sim → body → bone parent → bone-with-Copy-Rotation → body
            # Instead, seed world matrix so the body starts at the bone
            # head + local offset; the exporter reads the bone name from
            # the custom prop below to write body.bone_index.
            bone = arm_obj.data.bones.get(bone_name)
            if bone is not None:
                bone_world = arm_obj.matrix_world @ bone.matrix_local
                obj.matrix_world = bone_world @ Matrix.Translation(pos)
            else:
                obj.location = pos
            obj["m2_phys_bone_name"] = bone_name
        else:
            obj.location = pos

        if obj.data is not None:
            # Root: PASSIVE + kinematic so it follows the bone anchor
            # instead of falling; every non-root dynamic body is ACTIVE.
            is_active = (not is_root
                         and body.type == phys.BODY_DYNAMIC)
            blender_shape = obj.get("_m2_phys_blender_shape", "BOX")
            friction = obj.get("m2_phys_friction", 0.5)
            restitution = obj.get("m2_phys_restitution", 0.0)
            density = obj.get("m2_phys_density", 1.0)
            _apply_rigid_body(context, obj, is_active, blender_shape,
                              float(friction), float(restitution),
                              float(density),
                              kinematic=is_root)

        body_objs.append(obj)

    for ji, joint in enumerate(doc.joints):
        if joint.body_a >= len(body_objs) or joint.body_b >= len(body_objs):
            continue
        con_type = _joint_to_blender_type(joint.joint_type)
        empty_name = f"{model_name}_phys_join{ji:02d}"
        empty = bpy.data.objects.new(empty_name, None)
        empty.empty_display_type = "PLAIN_AXES"
        empty.empty_display_size = 0.03
        coll.objects.link(empty)
        empty["m2_phys_joint_type"] = int(joint.joint_type)
        empty["m2_phys_joint_id"] = int(joint.joint_id)
        try:
            empty["m2_phys_joint_unk"] = bytes(joint.unk).hex()
        except Exception:  # noqa: BLE001
            empty["m2_phys_joint_unk"] = "00000000"
        # Stash the per-joint-kind payload so the exporter can emit the
        # exact WELJ/SPHJ/SHOJ record instead of a default one. Without
        # this, imported .phys files re-export with identity frames /
        # zero anchors / zero twist-cone — the rig still holds together
        # but every frame_a/frame_b/anchor/twist/cone gets wiped.
        if joint.joint_type == phys.JOINT_WELD and joint.joint_id < len(doc.weld_joints):
            w = doc.weld_joints[joint.joint_id]
            empty["m2_phys_weld_frame_a"] = list(w.frame_a)
            empty["m2_phys_weld_frame_b"] = list(w.frame_b)
            empty["m2_phys_weld_ang_freq_hz"] = float(w.angular_frequency_hz)
            empty["m2_phys_weld_ang_damp"] = float(w.angular_damping_ratio)
        elif (joint.joint_type == phys.JOINT_SPHERICAL
              and joint.joint_id < len(doc.spherical_joints)):
            s = doc.spherical_joints[joint.joint_id]
            empty["m2_phys_sph_anchor_a"] = list(s.anchor_a)
            empty["m2_phys_sph_anchor_b"] = list(s.anchor_b)
            empty["m2_phys_sph_friction"] = float(s.friction_torque)
        elif (joint.joint_type == phys.JOINT_SHOULDER
              and joint.joint_id < len(doc.shoulder_joints)):
            sh = doc.shoulder_joints[joint.joint_id]
            empty["m2_phys_shoulder_frame_a"] = list(sh.frame_a)
            empty["m2_phys_shoulder_frame_b"] = list(sh.frame_b)
            empty["m2_phys_shoulder_lower_twist"] = float(sh.lower_twist)
            empty["m2_phys_shoulder_upper_twist"] = float(sh.upper_twist)
            empty["m2_phys_shoulder_cone"] = float(sh.cone_angle)
        # Place at midpoint of the two body pivots for visibility.
        a = body_objs[joint.body_a].matrix_world.translation
        b = body_objs[joint.body_b].matrix_world.translation
        empty.location = (a + b) * 0.5
        _apply_constraint(context, empty, con_type,
                          body_objs[joint.body_a], body_objs[joint.body_b])

    print("[phys-import] %s -> %d bodies, %d shapes, %d joints"
          % (os.path.basename(phys_path), len(doc.bodies),
             len(doc.shapes), len(doc.joints)), flush=True)
    return phys_path
