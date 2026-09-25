"""Build a Blender physics rig from .phys data.

Sources, in priority order: an explicit file, the M2's embedded PFDC chunk
(what modern retail models use), a ``<name>.phys`` sibling, or the only .phys
in the folder.
"""

from __future__ import annotations

import os
from typing import List, Optional

import bpy
from mathutils import Matrix, Vector

from . import phys, phys_rig


def _bone_names(arm_obj) -> List[str]:
    """M2 bone index -> bone name, in the exporter's order."""
    from . import from_scene
    return [b.name for b in from_scene._topo_bones(arm_obj.data)]


# --- X mirroring -----------------------------------------------------------
# Import and export both support "Mirror X". Reflecting a rig is its own
# inverse, so the same function serves both directions.

def _mv(v):
    return (-v[0], v[1], v[2])


def _mm(m):
    # R' = M R M with M = diag(-1, 1, 1): flip the x component of every axis,
    # then negate the X axis so the frame stays right-handed.
    ax, ay, az = m[0:3], m[3:6], m[6:9]
    return ((ax[0], -ax[1], -ax[2]) + (-ay[0], ay[1], ay[2]) + (-az[0], az[1], az[2])
            + _mv(m[9:12]))


def mirror_doc(doc: phys.PhysDoc) -> phys.PhysDoc:
    for b in doc.bodies:
        b.position = _mv(b.position)
    for c in doc.capsules:
        c.p1, c.p2 = _mv(c.p1), _mv(c.p2)
    for s in doc.spheres:
        s.center = _mv(s.center)
    for bx in doc.boxes:
        bx.frame = _mm(bx.frame)
    for group in (doc.weld_joints, doc.shoulder_joints,
                  doc.revolute_joints, doc.prismatic_joints):
        for j in group:
            j.frame_a, j.frame_b = _mm(j.frame_a), _mm(j.frame_b)
    for group in (doc.spherical_joints, doc.distance_joints):
        for j in group:
            j.anchor_a, j.anchor_b = _mv(j.anchor_a), _mv(j.anchor_b)
    # A reflection reverses the sense of rotation about Z.
    for j in doc.shoulder_joints:
        j.lower_twist, j.upper_twist = -j.upper_twist, -j.lower_twist
    for j in doc.revolute_joints:
        j.lower_angle, j.upper_angle = -j.upper_angle, -j.lower_angle
    return doc


# --- locating the data -----------------------------------------------------

def discover(m2_path: str, explicit_path: str = "", pfdc: bytes = b""):
    """Return ``(bytes, label)`` for this model's physics, or ``(None, "")``."""
    if explicit_path:
        if os.path.isfile(explicit_path):
            with open(explicit_path, "rb") as f:
                return f.read(), explicit_path
        print("[phys] override path not found: " + explicit_path, flush=True)
    if pfdc:
        return bytes(pfdc), "PFDC chunk embedded in the .m2"
    if not m2_path:
        return None, ""
    stem = os.path.splitext(m2_path)[0]
    folder = os.path.dirname(m2_path) or "."
    path = stem + ".phys"
    if not os.path.isfile(path):
        candidates = [f for f in os.listdir(folder) if f.lower().endswith(".phys")]
        if len(candidates) == 1:
            path = os.path.join(folder, candidates[0])
            print("[phys] no '%s.phys'; using the only .phys in the folder: %s"
                  % (os.path.basename(stem), candidates[0]), flush=True)
        else:
            if candidates:
                print("[phys] several .phys files here (%s); choose one with "
                      "'Phys File Override'" % ", ".join(candidates), flush=True)
            return None, ""
    with open(path, "rb") as f:
        return f.read(), path


# --- rest transforms -------------------------------------------------------

def _joint_frames(doc: phys.PhysDoc, joint: phys.Joint):
    """(frame_a, frame_b) as 4x4 matrices in each body's local space."""
    t, i = joint.joint_type, joint.joint_id
    table = {phys.JOINT_WELD: doc.weld_joints, phys.JOINT_SHOULDER: doc.shoulder_joints,
             phys.JOINT_REVOLUTE: doc.revolute_joints,
             phys.JOINT_PRISMATIC: doc.prismatic_joints}.get(t)
    if table is not None:
        if not 0 <= i < len(table):
            return None
        return (phys_rig.mat3x4_to_matrix(table[i].frame_a),
                phys_rig.mat3x4_to_matrix(table[i].frame_b))
    table = doc.spherical_joints if t == phys.JOINT_SPHERICAL else doc.distance_joints
    if not 0 <= i < len(table):
        return None
    return (Matrix.Translation(Vector(table[i].anchor_a)),
            Matrix.Translation(Vector(table[i].anchor_b)))


def resolve_rest(doc: phys.PhysDoc) -> List[Matrix]:
    """Armature-space rest matrix of every body.

    Retail files are written two ways: bodies carry their model-space position,
    or every position is zero and only the joint frames say where things sit.
    Walking the joints outward from the root (B = A x frameA x frameB^-1) is
    right for both; bodies no joint reaches keep their own position.
    """
    n = len(doc.bodies)
    rest: List[Optional[Matrix]] = [None] * n
    roots = [i for i, b in enumerate(doc.bodies) if b.type == phys.BODY_ROOT] or ([0] if n else [])
    links = {}
    for joint in doc.joints:
        if joint.body_a < n and joint.body_b < n:
            frames = _joint_frames(doc, joint)
            if frames is not None:
                links.setdefault(joint.body_a, []).append((joint.body_b, frames, False))
                links.setdefault(joint.body_b, []).append((joint.body_a, frames, True))
    # Point joints (spherical / distance) carry no orientation, so bodies stay
    # axis-aligned through them, which is how the file stores bodies anyway.
    queue = []
    for r in roots:
        rest[r] = Matrix.Translation(Vector(doc.bodies[r].position))
        queue.append(r)
    while queue:
        a = queue.pop(0)
        for b, (fa, fb), reverse in links.get(a, ()):
            if rest[b] is not None:
                continue
            rest[b] = rest[a] @ (fb @ fa.inverted() if reverse else fa @ fb.inverted())
            queue.append(b)
    for i in range(n):
        if rest[i] is None:
            rest[i] = Matrix.Translation(Vector(doc.bodies[i].position))
    return rest


def _place_unreached(doc, rest, bone_names, arm_obj):
    """A body no joint reaches and whose position is zero sits on its bone."""
    for i, body in enumerate(doc.bodies):
        if any(abs(v) > 1e-9 for v in body.position):
            continue
        if any(j.body_a == i or j.body_b == i for j in doc.joints):
            continue
        name = bone_names[body.bone_index] if body.bone_index < len(bone_names) else ""
        bone = arm_obj.data.bones.get(name) if name else None
        if bone is not None:
            rest[i] = Matrix.Translation(bone.head_local)


# --- building --------------------------------------------------------------

def _fill_shape(shape_props, doc: phys.PhysDoc, s: phys.Shape):
    shape_props.kind = phys_rig.shape_kind_token(s.shape_type)
    shape_props.friction = s.friction
    shape_props.restitution = s.restitution
    shape_props.density = max(s.density, 0.0001)
    shape_props.unk_hex = bytes(s.unk).hex()
    shape_props.x14 = int(s.x14) if s.x14 < 0x80000000 else int(s.x14) - 0x100000000
    shape_props.x18 = s.x18
    shape_props.x1c = s.x1c
    shape_props.x1e = s.x1e
    i = s.shape_index
    if s.shape_type == phys.SHAPE_CAPSULE and 0 <= i < len(doc.capsules):
        c = doc.capsules[i]
        shape_props.p1, shape_props.p2, shape_props.radius = c.p1, c.p2, max(c.radius, 0.0005)
    elif s.shape_type == phys.SHAPE_SPHERE and 0 <= i < len(doc.spheres):
        sp = doc.spheres[i]
        shape_props.p1, shape_props.radius = sp.center, max(sp.radius, 0.0005)
    elif s.shape_type == phys.SHAPE_BOX and 0 <= i < len(doc.boxes):
        bx = doc.boxes[i]
        shape_props.p1 = bx.frame[9:12]
        shape_props.box_axes = bx.frame[0:9]
        shape_props.half_extents = [max(h, 0.0005) for h in bx.half_extents]
    else:
        shape_props.kind = "POLYTOPE"
        shape_props.polytope_index = i


def _fill_joint(props, doc: phys.PhysDoc, joint: phys.Joint):
    t, i = joint.joint_type, joint.joint_id
    props.joint_type = phys_rig.joint_type_token(t)
    props.unk_hex = bytes(joint.unk).hex()
    if t == phys.JOINT_WELD:
        w = doc.weld_joints[i]
        props.angular_frequency_hz = w.angular_frequency_hz
        props.angular_damping_ratio = w.angular_damping_ratio
        props.linear_frequency_hz = w.linear_frequency_hz
        props.linear_damping_ratio = w.linear_damping_ratio
        props.unk70 = w.unk70
    elif t == phys.JOINT_SPHERICAL:
        props.friction_torque = doc.spherical_joints[i].friction_torque
    elif t == phys.JOINT_SHOULDER:
        s = doc.shoulder_joints[i]
        props.lower_twist = max(-180.0, min(0.0, s.lower_twist))
        props.upper_twist = max(0.0, min(180.0, s.upper_twist))
        props.cone_angle = max(0.0, min(180.0, s.cone_angle))
        props.max_motor_torque = s.max_motor_torque
        props.motor_mode = int(s.motor_mode) & 0x7FFFFFFF
        props.motor_frequency_hz = s.motor_frequency_hz
        props.motor_damping_ratio = s.motor_damping_ratio
    elif t == phys.JOINT_REVOLUTE:
        r = doc.revolute_joints[i]
        props.lower_limit, props.upper_limit = r.lower_angle, r.upper_angle
        props.max_motor_torque = r.max_motor_torque
        props.motor_mode = int(r.motor_mode) & 0x7FFFFFFF
        props.motor_frequency_hz = r.motor_frequency_hz
        props.motor_damping_ratio = r.motor_damping_ratio
    elif t == phys.JOINT_PRISMATIC:
        p = doc.prismatic_joints[i]
        props.lower_limit, props.upper_limit = p.lower_limit, p.upper_limit
        props.x68, props.x70 = p.x68, p.x70
        props.max_motor_torque = p.max_motor_force
        props.motor_mode = int(p.motor_mode) & 0x7FFFFFFF
        props.motor_frequency_hz = p.motor_frequency_hz
        props.motor_damping_ratio = p.motor_damping_ratio
    elif t == phys.JOINT_DISTANCE:
        props.distance_factor = doc.distance_joints[i].distance_factor


def build_rig(context, doc: phys.PhysDoc, arm_obj, model_name: str,
              mirror_x: bool = False):
    """Create the rig collection for ``doc`` on ``arm_obj``. Returns it."""
    if mirror_x:
        mirror_doc(doc)
    phys_rig.ensure_world(context)

    old = phys_rig.find_rig(context, arm_obj)
    if old is not None and old.m2_phys_rig.armature == arm_obj:
        phys_rig.stop_playback(context)
        phys_rig.clear_follow(arm_obj, old)
        for obj in list(old.objects):
            phys_rig.remove_object(obj)
        bpy.data.collections.remove(old)

    coll = phys_rig.ensure_rig(context, arm_obj, model_name)
    rig = coll.m2_phys_rig
    rig.version = doc.version
    rig.has_phyt = doc.phyt is not None
    phyt = doc.phyt or 0
    rig.phyt = phyt if phyt < 0x80000000 else phyt - 0x100000000
    rig.chunk_order = ",".join(doc.chunk_order)
    rig.tags = phys_rig.format_tags(doc.tags)
    rig.shoulder_size = doc.shoulder_size
    rig.raw_chunks = phys_rig.format_raw_chunks(doc.raw_chunks)

    bone_names = _bone_names(arm_obj)
    arm_world = arm_obj.matrix_world
    rest = resolve_rest(doc)
    _place_unreached(doc, rest, bone_names, arm_obj)

    bodies = []
    with phys_rig.suppress_updates():
        for i, body in enumerate(doc.bodies):
            bone = bone_names[body.bone_index] if body.bone_index < len(bone_names) else ""
            if not bone:
                print("[phys] body %d names bone %d, which the armature lacks"
                      % (i, body.bone_index), flush=True)
            token = phys_rig.body_type_token(body.type)
            if body.type == phys.BODY_ROOT and not any(
                    b.m2_phys_body.body_type == "ROOT" for b in bodies):
                token = "ROOT"                      # first anchor in the file
            label = "root" if token == "ROOT" else (bone or "%02d" % i)
            obj = phys_rig.new_body(context, coll, "phys_body_" + label,
                                    arm_world @ rest[i], token, bone)
            props = obj.m2_phys_body
            props.file_body_type = int(body.type)
            props.file_body_token = token
            props.drag, props.unk0, props.x1c = body.drag, body.unk0, body.x1c
            props.unk1, props.x28 = body.unk1, body.x28
            props.x2c_hex = bytes(body.x2c).hex()
            props.pad_a_hex = bytes(body.pad_a).hex()
            props.pad_b_hex = bytes(body.pad_b).hex()
            props.file_position = body.position
            props.has_file_position = True
            props.file_index = i
            for s in doc.shapes[body.shapes_base:body.shapes_base + max(body.shapes_count, 0)]:
                _fill_shape(props.shapes.add(), doc, s)
            bodies.append(obj)

    for obj in bodies:
        phys_rig.rebuild_body(context, obj)
    context.view_layer.update()
    with phys_rig.suppress_updates():
        for obj in bodies:
            obj.m2_phys_body.rest_location = phys_rig.body_frame_world(obj).translation

    made = 0
    for ji, joint in enumerate(doc.joints):
        if joint.body_a >= len(bodies) or joint.body_b >= len(bodies):
            print("[phys] joint %d references a missing body; skipped" % ji, flush=True)
            continue
        frames = _joint_frames(doc, joint)
        if frames is None:
            print("[phys] joint %d has no data record; skipped" % ji, flush=True)
            continue
        a, b = bodies[joint.body_a], bodies[joint.body_b]
        world = arm_world @ rest[joint.body_a] @ frames[0]
        name = "phys_joint_%s" % (b.m2_phys_body.bone or "%02d" % ji)
        with phys_rig.suppress_updates():
            empty = phys_rig.new_joint(context, coll, name, world, a, b)
            props = empty.m2_phys_joint
            _fill_joint(props, doc, joint)
            props.frame_a = phys_rig.matrix_to_mat3x4(frames[0]) \
                if joint.joint_type in (phys.JOINT_SPHERICAL, phys.JOINT_DISTANCE) \
                else _raw_frame(doc, joint, "a")
            props.frame_b = phys_rig.matrix_to_mat3x4(frames[1]) \
                if joint.joint_type in (phys.JOINT_SPHERICAL, phys.JOINT_DISTANCE) \
                else _raw_frame(doc, joint, "b")
            props.has_file_frames = True
            props.file_index = ji
        phys_rig.sync_constraint(context, empty)
        with phys_rig.suppress_updates():
            props.rest_matrix = [v for row in phys_rig.rest_world(empty) for v in row]
        made += 1

    phys_rig.apply_self_collision(coll, context.scene.m2_physics.self_collision
                                  if hasattr(context.scene, "m2_physics") else False)
    print("[phys] built rig '%s' (%s): %d bodies, %d joints"
          % (coll.name, phys.summary(doc), len(bodies), made), flush=True)
    return coll


def _raw_frame(doc, joint, side):
    table = {phys.JOINT_WELD: doc.weld_joints, phys.JOINT_SHOULDER: doc.shoulder_joints,
             phys.JOINT_REVOLUTE: doc.revolute_joints,
             phys.JOINT_PRISMATIC: doc.prismatic_joints}[joint.joint_type]
    rec = table[joint.joint_id]
    return rec.frame_a if side == "a" else rec.frame_b


def load_phys_into_scene(context, m2_path: str, arm_obj, model_name: str,
                         mirror_x: bool = False, explicit_path: str = "",
                         pfdc: bytes = b"") -> Optional[str]:
    """Import this model's physics if it has any. Returns a label for where the
    data came from, or None when there was nothing to load."""
    data, label = discover(m2_path, explicit_path, pfdc)
    if data is None:
        return None
    if arm_obj is None:
        print("[phys] the model has physics but no armature was imported; "
              "skipping", flush=True)
        return None
    try:
        doc = phys.read(data)
    except Exception as exc:  # noqa: BLE001
        print("[phys] could not read %s: %s" % (label, exc), flush=True)
        return None
    if not doc.bodies:
        print("[phys] %s has no bodies; nothing to build" % label, flush=True)
        return None
    print("[phys] loading %s" % label, flush=True)
    build_rig(context, doc, arm_obj, model_name, mirror_x=mirror_x)
    return label
