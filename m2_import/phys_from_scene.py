"""Turn a Blender physics rig back into .phys data.

Everything is measured from the objects' world matrices, so bodies and joints
can be moved, rotated or scaled freely in the viewport. Bodies are written
axis-aligned in model space (as the format requires): an object's rotation and
scale are baked into its shape coordinates and joint frames.

Values imported from a file are written back untouched for as long as the
objects they describe have not moved, so an unedited rig exports byte-identical.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from mathutils import Matrix, Vector

from . import phys, phys_rig

_EPS = 1e-5


def _bone_index_map(armature_obj) -> Dict[str, int]:
    if armature_obj is None or armature_obj.type != "ARMATURE":
        return {}
    from . import from_scene
    return {b.name: i for i, b in enumerate(from_scene._topo_bones(armature_obj.data))}


def _hex(text: str, n: int) -> bytes:
    try:
        raw = bytes.fromhex(text or "")
    except ValueError:
        raw = b""
    return raw[:n].ljust(n, b"\x00")


def _close(a, b, eps=_EPS) -> bool:
    return all(abs(x - y) <= eps for x, y in zip(a, b))


def _flat(matrix: Matrix):
    return [v for row in matrix for v in row]


def _u32(value: int) -> int:
    return int(value) & 0xFFFFFFFF


class RigError(Exception):
    pass


def validate(coll) -> Tuple[List[str], List[str]]:
    """(errors, warnings) for a rig. Errors stop the export."""
    errors, warnings = [], []
    bodies = phys_rig.rig_bodies(coll)
    if not bodies:
        return ["The rig has no bodies."], warnings
    arm = coll.m2_phys_rig.armature
    bone_map = _bone_index_map(arm)
    if arm is None:
        errors.append("The rig is not linked to an armature.")

    roots = [b for b in bodies if b.m2_phys_body.body_type == "ROOT"]
    anchors = [b for b in bodies if b.m2_phys_body.body_type != "DYNAMIC"]
    if not anchors:
        errors.append("No anchor. At least one body must follow a bone (Root Anchor or "
                      "Anchor / Collider) for the rig to hang from.")

    seen_bones = {}
    for b in bodies:
        props = b.m2_phys_body
        if not props.bone:
            errors.append("'%s' has no bone assigned." % b.name)
        elif bone_map and props.bone not in bone_map:
            errors.append("'%s' uses bone '%s', which the armature does not have."
                          % (b.name, props.bone))
        elif props.body_type == "DYNAMIC":
            if props.bone in seen_bones:
                errors.append("'%s' and '%s' both drive bone '%s'."
                              % (seen_bones[props.bone], b.name, props.bone))
            seen_bones[props.bone] = b.name
        if props.body_type == "DYNAMIC" and not props.shapes:
            warnings.append("Dynamic body '%s' has no shape, so it has no mass." % b.name)

    joints = phys_rig.rig_joints(coll)
    linked = {}
    for j in joints:
        jp = j.m2_phys_joint
        if jp.body_a is None or jp.body_b is None:
            errors.append("Joint '%s' is missing a body." % j.name)
            continue
        if jp.body_a == jp.body_b:
            errors.append("Joint '%s' connects a body to itself." % j.name)
            continue
        if not (phys_rig.is_body(jp.body_a) and phys_rig.is_body(jp.body_b)):
            errors.append("Joint '%s' points at an object that is not a physics body." % j.name)
            continue
        linked.setdefault(jp.body_a.name, set()).add(jp.body_b.name)
        linked.setdefault(jp.body_b.name, set()).add(jp.body_a.name)
        if jp.joint_type in ("REVOLUTE", "PRISMATIC", "DISTANCE") and coll.m2_phys_rig.version < 2:
            errors.append("Joint '%s' is %s, which needs rig version 2 or newer."
                          % (j.name, jp.joint_type.title()))

    if anchors:
        # Anything that follows the skeleton holds a chain up.
        names = [b.name for b in anchors]
        reached, queue = set(names), list(names)
        while queue:
            for other in linked.get(queue.pop(), ()):
                if other not in reached:
                    reached.add(other)
                    queue.append(other)
        for b in bodies:
            if b.m2_phys_body.body_type == "DYNAMIC" and b.name not in reached:
                warnings.append("'%s' is not joined to the root or a kinematic body; "
                                "it will fall freely." % b.name)
    return errors, warnings


def _order_bodies(bodies):
    """Root first, then parents before children, the way retail files are laid out."""
    def key(o):
        p = o.m2_phys_body
        return (0 if p.body_type == "ROOT" else 1,
                p.file_index if p.file_index >= 0 else 1 << 20, o.name)
    return sorted(bodies, key=key)


def _body_frames(arm_inv: Matrix, obj):
    """(file frame, bake matrix): the axis-aligned frame written to the file, and
    the matrix taking this object's local coordinates into it."""
    local = arm_inv @ phys_rig.body_frame_world(obj)
    origin = local.to_translation()
    frame = Matrix.Translation(origin)
    return frame, frame.inverted() @ local


def _scale_of(matrix: Matrix) -> float:
    s = matrix.to_scale()
    return (abs(s.x) + abs(s.y) + abs(s.z)) / 3.0


def build_phys_doc(context, armature_obj=None, mirror_x: bool = False,
                   collection=None) -> Optional[phys.PhysDoc]:
    """The PhysDoc for the rig on ``armature_obj``, or None if there is no rig.
    Raises RigError when the rig cannot be written."""
    coll = collection or phys_rig.find_rig(context, armature_obj)
    if coll is None:
        return None
    bodies = phys_rig.rig_bodies(coll)
    if not bodies:
        return None
    errors, warnings = validate(coll)
    for w in warnings:
        print("[phys] warning: " + w, flush=True)
    if errors:
        raise RigError(errors[0] + (" (+%d more)" % (len(errors) - 1) if len(errors) > 1 else ""))

    rig = coll.m2_phys_rig
    arm = rig.armature or armature_obj
    arm_inv = arm.matrix_world.inverted() if arm is not None else Matrix.Identity(4)
    bone_map = _bone_index_map(arm)

    doc = phys.PhysDoc(version=rig.version)
    doc.phyt = _u32(rig.phyt) if rig.has_phyt else None
    doc.chunk_order = [t for t in rig.chunk_order.split(",") if t]
    doc.tags = phys_rig.parse_tags(rig.tags)
    doc.shoulder_size = rig.shoulder_size
    doc.raw_chunks = phys_rig.parse_raw_chunks(rig.raw_chunks)

    ordered = _order_bodies(bodies)
    index = {obj: i for i, obj in enumerate(ordered)}
    frames, bakes, moved = {}, {}, {}

    for obj in ordered:
        props = obj.m2_phys_body
        frame, bake = _body_frames(arm_inv, obj)
        frames[obj], bakes[obj] = frame, bake
        untouched = (props.has_file_position
                     and _close(phys_rig.body_frame_world(obj).translation,
                                props.rest_location)
                     and _close(_flat(bake), _flat(Matrix.Identity(4))))
        moved[obj] = not untouched
        if untouched:
            position = tuple(props.file_position)
        elif props.body_type == "DYNAMIC" and props.unk1 == 0.0:
            # Retail convention: unk1 == 0 means the position is bone-relative
            # and the joint frames carry the layout (see buckle_cloth_reputation).
            position = (0.0, 0.0, 0.0)
        else:
            position = tuple(frame.to_translation())
        scale = _scale_of(bake)

        base = len(doc.shapes)
        for s in props.shapes:
            rec = phys.Shape(shape_type=phys_rig.shape_kind_int(s.kind),
                             friction=s.friction, restitution=s.restitution,
                             density=s.density, unk=_hex(s.unk_hex, 4),
                             x14=_u32(s.x14), x18=s.x18, x1c=s.x1c, x1e=s.x1e)
            p1 = tuple(s.p1) if untouched else tuple(bake @ Vector(s.p1))
            if s.kind == "CAPSULE":
                p2 = tuple(s.p2) if untouched else tuple(bake @ Vector(s.p2))
                rec.shape_index = len(doc.capsules)
                doc.capsules.append(phys.CapsuleShape(p1, p2, s.radius * (1.0 if untouched else scale)))
            elif s.kind == "SPHERE":
                rec.shape_index = len(doc.spheres)
                doc.spheres.append(phys.SphereShape(p1, s.radius * (1.0 if untouched else scale)))
            elif s.kind == "BOX":
                axes = tuple(s.box_axes)
                half = tuple(s.half_extents)
                if not untouched:
                    rot = bake.to_3x3().normalized() @ phys_rig.mat3x4_to_matrix(axes + (0, 0, 0)).to_3x3()
                    axes = phys_rig.matrix_to_mat3x4(rot.to_4x4())[0:9]
                    sc = bake.to_scale()
                    half = (half[0] * abs(sc.x), half[1] * abs(sc.y), half[2] * abs(sc.z))
                rec.shape_index = len(doc.boxes)
                doc.boxes.append(phys.BoxShape(phys.make_mat3x4(axes, p1), half))
            else:                           # polytope: geometry stays in the raw PLYT chunk
                rec.shape_index = s.polytope_index
            doc.shapes.append(rec)

        if props.file_body_type >= 0 and props.body_type == props.file_body_token:
            body_type = props.file_body_type            # unchanged since import
        else:
            body_type = phys_rig.body_type_int(props.body_type)
        doc.bodies.append(phys.Body(
            type=body_type, position=position,
            bone_index=bone_map.get(props.bone, 0) & 0xFFFF,
            shapes_base=base if props.shapes else 0, shapes_count=len(props.shapes),
            x1c=props.x1c, unk0=props.unk0, drag=props.drag, unk1=props.unk1,
            x28=props.x28, x2c=_hex(props.x2c_hex, 4),
            pad_a=_hex(props.pad_a_hex, 2), pad_b=_hex(props.pad_b_hex, 2)))

    def _joint_key(o):
        fi = o.m2_phys_joint.file_index
        return (fi if fi >= 0 else 1 << 20, o.name)

    for empty in sorted(phys_rig.rig_joints(coll), key=_joint_key):
        jp = empty.m2_phys_joint
        a, b = jp.body_a, jp.body_b
        joint_world = phys_rig.rest_world(empty)
        keep = (jp.has_file_frames and not moved[a] and not moved[b]
                and _close(_flat(joint_world), jp.rest_matrix))
        if keep:
            frame_a, frame_b = tuple(jp.frame_a), tuple(jp.frame_b)
        else:
            world = arm_inv @ joint_world
            world = Matrix.Translation(world.to_translation()) @ world.to_3x3().normalized().to_4x4()
            frame_a = phys_rig.matrix_to_mat3x4(frames[a].inverted() @ world)
            frame_b = phys_rig.matrix_to_mat3x4(frames[b].inverted() @ world)

        kind = jp.joint_type
        if kind == "WELD":
            jid = len(doc.weld_joints)
            doc.weld_joints.append(phys.WeldJoint(
                frame_a, frame_b, jp.angular_frequency_hz, jp.angular_damping_ratio,
                jp.linear_frequency_hz, jp.linear_damping_ratio, jp.unk70))
        elif kind == "SPHERICAL":
            jid = len(doc.spherical_joints)
            doc.spherical_joints.append(phys.SphericalJoint(
                frame_a[9:12], frame_b[9:12], jp.friction_torque))
        elif kind == "SHOULDER":
            jid = len(doc.shoulder_joints)
            doc.shoulder_joints.append(phys.ShoulderJoint(
                frame_a, frame_b, jp.lower_twist, jp.upper_twist, jp.cone_angle,
                jp.max_motor_torque, _u32(jp.motor_mode),
                jp.motor_frequency_hz, jp.motor_damping_ratio))
        elif kind == "REVOLUTE":
            jid = len(doc.revolute_joints)
            doc.revolute_joints.append(phys.RevoluteJoint(
                frame_a, frame_b, jp.lower_limit, jp.upper_limit, jp.max_motor_torque,
                _u32(jp.motor_mode), jp.motor_frequency_hz, jp.motor_damping_ratio))
        elif kind == "PRISMATIC":
            jid = len(doc.prismatic_joints)
            doc.prismatic_joints.append(phys.PrismaticJoint(
                frame_a, frame_b, jp.lower_limit, jp.upper_limit, jp.x68,
                jp.max_motor_torque, jp.x70, _u32(jp.motor_mode),
                jp.motor_frequency_hz, jp.motor_damping_ratio))
        else:
            jid = len(doc.distance_joints)
            doc.distance_joints.append(phys.DistanceJoint(
                frame_a[9:12], frame_b[9:12], jp.distance_factor))
        doc.joints.append(phys.Joint(index[a], index[b], phys_rig.joint_type_int(kind),
                                     jid, _hex(jp.unk_hex, 4)))

    if mirror_x:
        from . import phys_to_scene
        phys_to_scene.mirror_doc(doc)
    return doc


def build_phys_bytes(context, armature_obj=None, mirror_x: bool = False,
                     collection=None) -> Optional[bytes]:
    doc = build_phys_doc(context, armature_obj, mirror_x=mirror_x, collection=collection)
    return None if doc is None else phys.write(doc)
