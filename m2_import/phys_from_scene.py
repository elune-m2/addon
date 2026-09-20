"""Build a PhysDoc from a Blender scene's rigid-body world.

How the user sets things up in Blender:
 - Add a mesh/empty per rigid body, parent it to the armature bone that
   the physics should follow (Object > Parent > Bone), then enable
   Physics > Rigid Body on it.
 - `rigid_body.type = ACTIVE` → dynamic body; `PASSIVE` → kinematic.
 - Exactly one root object per rig — mark it with custom property
   `m2_phys_root = True` (type is ignored for the root; it becomes
   BODY_ROOT). The root itself doesn't need a shape.
 - Set `rigid_body.collision_shape` to BOX / SPHERE / CAPSULE. Any other
   shape is skipped with a warning.
 - Add Rigid Body Constraints between bodies (Object > Rigid Body >
   Connect). Only FIXED (weld) is emitted in v0. HINGE / POINT / GENERIC
   are recorded for future upgrade to v2+.

Coordinate convention: identity Blender→WoW (matches the addon's
existing exporters; optional X mirror follows the same `mirror_x` flag).
Body position is bone-local (matches retail sample where all body
positions are zero and the bone pivot supplies the world location).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from . import phys


def _bone_index_map(armature_obj) -> Dict[str, int]:
    """Map bone name → M2 bone index (topological order — same as the addon)."""
    if armature_obj is None or armature_obj.type != "ARMATURE":
        return {}
    from . import from_scene
    order = from_scene._topo_bones(armature_obj.data)
    return {b.name: i for i, b in enumerate(order)}


def _wow_vec(v, mirror_x: bool) -> phys.Vec3:
    return (-v[0], v[1], v[2]) if mirror_x else (v[0], v[1], v[2])


def _bone_local_position(obj, arm_obj, bone_name: str, mirror_x: bool) -> phys.Vec3:
    """Object position expressed in the parent bone's local space."""
    if arm_obj is None or bone_name not in arm_obj.data.bones:
        return _wow_vec(obj.matrix_world.translation, mirror_x)
    bone = arm_obj.data.bones[bone_name]
    # Bone rest matrix in armature space; take the head (pivot).
    bone_world = arm_obj.matrix_world @ bone.matrix_local
    local = bone_world.inverted() @ obj.matrix_world.translation
    return _wow_vec(local, mirror_x)


def _mat3x4_from_matrix(m) -> phys.Mat3x4:
    """3x4 row-major mat from a Blender mathutils.Matrix."""
    return (
        m[0][0], m[0][1], m[0][2], m[0][3],
        m[1][0], m[1][1], m[1][2], m[1][3],
        m[2][0], m[2][1], m[2][2], m[2][3],
    )


def _shape_for_object(obj, doc: phys.PhysDoc,
                      shape_idx_counter: Dict[int, int]) -> Optional[int]:
    """Append the object's shape data to the right list and return its
    combined-Shape index (the SHAP index the Body will point at)."""
    rb = obj.rigid_body
    if rb is None:
        return None
    kind = rb.collision_shape

    dim = obj.dimensions  # world-space bounding-box dims (in Blender units)
    hx, hy, hz = dim[0] / 2.0, dim[1] / 2.0, dim[2] / 2.0

    # Prefer values captured at import time so a round-trip is byte-exact
    # (Blender's live rigid_body.friction etc. drift from the imported
    # values when Blender clamps or normalizes them). Fall back to the
    # live rigid-body state for freshly authored rigs.
    friction = float(obj.get("m2_phys_friction", rb.friction))
    restitution = float(obj.get("m2_phys_restitution", rb.restitution))
    if "m2_phys_density" in obj:
        density = float(obj["m2_phys_density"])
    else:
        density = max(float(rb.mass), 0.0001)
    unk_hex = obj.get("m2_phys_shape_unk", None)
    unk_bytes = b"\x00\x00\x00\x00"
    if isinstance(unk_hex, str) and len(unk_hex) == 8:
        try:
            unk_bytes = bytes.fromhex(unk_hex)
        except ValueError:
            pass

    if kind == "BOX":
        chunk_idx = len(doc.boxes)
        doc.boxes.append(phys.BoxShape(
            frame=phys.identity_mat3x4(),
            half_extents=(hx, hy, hz),
        ))
        shape_type = phys.SHAPE_BOX
    elif kind == "SPHERE":
        chunk_idx = len(doc.spheres)
        r = max(hx, hy, hz)
        doc.spheres.append(phys.SphereShape(center=(0.0, 0.0, 0.0), radius=r))
        shape_type = phys.SHAPE_SPHERE
    elif kind == "CAPSULE":
        chunk_idx = len(doc.capsules)
        # Blender capsule: aligned to local +Z, radius = max(x,y)/2,
        # length = z dim minus the two hemispherical caps.
        radius = max(hx, hy)
        half_len = max(hz - radius, 0.0)
        doc.capsules.append(phys.CapsuleShape(
            p1=(0.0, 0.0, -half_len),
            p2=(0.0, 0.0, +half_len),
            radius=radius,
        ))
        shape_type = phys.SHAPE_CAPSULE
    else:
        print(f"[phys] skipping shape on '{obj.name}': "
              f"collision_shape={kind!r} not supported (use BOX/SPHERE/CAPSULE)")
        return None

    shape_slot = len(doc.shapes)
    doc.shapes.append(phys.Shape(
        shape_type=shape_type,
        shape_index=chunk_idx,
        friction=friction,
        restitution=restitution,
        density=density,
        unk=unk_bytes,
    ))
    return shape_slot


def _joint_type_from_constraint(c) -> Optional[int]:
    t = c.type
    if t == "FIXED":
        return phys.JOINT_WELD
    if t == "POINT":
        return phys.JOINT_SPHERICAL
    if t == "HINGE":
        return phys.JOINT_REVOLUTE   # v2+, not emitted at v0
    if t == "GENERIC" or t == "GENERIC_SPRING":
        return phys.JOINT_SHOULDER   # closest analogue
    if t == "SLIDER":
        return phys.JOINT_PRISMATIC  # v2+, not emitted at v0
    return None


def build_phys_doc(context, armature_obj, mirror_x: bool = False,
                   version: int = 0) -> Optional[phys.PhysDoc]:
    """Walk the scene's rigid bodies and build a PhysDoc.

    Returns None if no rigid bodies are present.
    """
    scene = context.scene
    rbw = scene.rigidbody_world
    if rbw is None or rbw.collection is None:
        return None
    rb_objects = [o for o in rbw.collection.objects if o.rigid_body is not None]
    if not rb_objects:
        return None

    doc = phys.PhysDoc(version=version)
    bone_idx_map = _bone_index_map(armature_obj)

    # Body ordering: root first (marked with m2_phys_root=True), then the
    # rest in scene order. Joint chunk uses body indices, so we need a
    # stable object→index map.
    root_objs = [o for o in rb_objects if bool(o.get("m2_phys_root", False))]
    if len(root_objs) > 1:
        print(f"[phys] warning: more than one m2_phys_root object "
              f"({[o.name for o in root_objs]}) — using {root_objs[0].name!r}")
    root_obj = root_objs[0] if root_objs else None
    other_objs = [o for o in rb_objects if o is not root_obj]
    ordered = ([root_obj] if root_obj is not None else []) + other_objs
    obj_to_idx = {o: i for i, o in enumerate(ordered)}

    for obj in ordered:
        # Bone assignment: prefer live bone-parenting, fall back to the
        # `m2_phys_bone_name` custom prop. Dynamic bodies from the
        # importer or the authoring path are NOT bone-parented (to avoid
        # a depsgraph cycle with the bone's Copy Rotation constraint),
        # so parent_bone is empty and the custom prop is the only signal
        # of which bone this body corresponds to.
        if obj.parent_type == "BONE" and obj.parent_bone:
            bone_name = obj.parent_bone
        else:
            bone_name = str(obj.get("m2_phys_bone_name", "") or "")
        bone_index = bone_idx_map.get(bone_name, 0xFFFF)
        if bone_index == 0xFFFF and bone_name:
            print(f"[phys] warning: '{obj.name}' bone '{bone_name}' "
                  f"not in the M2 bone list — writing sentinel 0xFFFF")

        is_root = obj is root_obj
        if is_root:
            body_type = phys.BODY_ROOT
            shape_slot = None
        else:
            # Prefer the value captured on import: Blender's ACTIVE/PASSIVE
            # is coarser than .phys's DYNAMIC/KINEMATIC distinction so we'd
            # lose fidelity converting via rigid_body.type alone.
            if "m2_phys_body_type" in obj:
                body_type = int(obj["m2_phys_body_type"])
            else:
                body_type = (phys.BODY_DYNAMIC
                             if obj.rigid_body.type == "ACTIVE"
                             else phys.BODY_KINEMATIC)
            shape_slot = _shape_for_object(obj, doc, {})

        pos = _bone_local_position(obj, armature_obj, bone_name, mirror_x)

        doc.bodies.append(phys.Body(
            type=body_type,
            position=pos,
            bone_index=bone_index & 0xFFFF,
            shapes_base=(shape_slot if shape_slot is not None else 0),
            shapes_count=(1 if shape_slot is not None else 0),
        ))

    # Constraints → joints. Walk every object; only those with an active
    # rigid_body_constraint count.
    for obj in ordered:
        con = obj.rigid_body_constraint
        if con is None or not con.enabled:
            continue
        if con.object1 is None or con.object2 is None:
            continue
        if con.object1 not in obj_to_idx or con.object2 not in obj_to_idx:
            print(f"[phys] skipping constraint on '{obj.name}': "
                  f"one of the two bodies isn't in the rigid-body world")
            continue

        jt = _joint_type_from_constraint(con)
        if jt is None:
            print(f"[phys] skipping constraint on '{obj.name}': "
                  f"type={con.type!r} not mapped")
            continue

        # v0 only supports weld/spherical/shoulder. Downgrade unsupported
        # types to weld (fixed) so the rig at least holds together.
        if version < 2 and jt in (phys.JOINT_REVOLUTE, phys.JOINT_PRISMATIC,
                                  phys.JOINT_DISTANCE):
            print(f"[phys] '{obj.name}': joint type {con.type} needs "
                  f"phys version 2+, downgrading to WELD for v0")
            jt = phys.JOINT_WELD

        a = obj_to_idx[con.object1]
        b = obj_to_idx[con.object2]

        # Prefer per-joint payload captured on import so re-export is
        # byte-exact. Fresh authored constraints fall back to sensible
        # v0 defaults that match the retail buckle sample.
        def _vec3_prop(name, default):
            v = obj.get(name)
            if v is None:
                return default
            try:
                return (float(v[0]), float(v[1]), float(v[2]))
            except (TypeError, IndexError):
                return default

        def _mat3x4_prop(name):
            v = obj.get(name)
            if v is None:
                return phys.identity_mat3x4()
            try:
                return tuple(float(x) for x in list(v)[:12])
            except (TypeError, ValueError):
                return phys.identity_mat3x4()

        joint_unk_hex = obj.get("m2_phys_joint_unk", None)
        joint_unk = b"\x00\x00\x00\x00"
        if isinstance(joint_unk_hex, str) and len(joint_unk_hex) == 8:
            try:
                joint_unk = bytes.fromhex(joint_unk_hex)
            except ValueError:
                pass

        if jt == phys.JOINT_WELD:
            jid = len(doc.weld_joints)
            doc.weld_joints.append(phys.WeldJoint(
                frame_a=_mat3x4_prop("m2_phys_weld_frame_a"),
                frame_b=_mat3x4_prop("m2_phys_weld_frame_b"),
                angular_frequency_hz=float(obj.get("m2_phys_weld_ang_freq_hz", 0.0)),
                angular_damping_ratio=float(obj.get("m2_phys_weld_ang_damp", 1.0)),
            ))
        elif jt == phys.JOINT_SPHERICAL:
            jid = len(doc.spherical_joints)
            doc.spherical_joints.append(phys.SphericalJoint(
                anchor_a=_vec3_prop("m2_phys_sph_anchor_a", (0.0, 0.0, 0.0)),
                anchor_b=_vec3_prop("m2_phys_sph_anchor_b", (0.0, 0.0, 0.0)),
                friction_torque=float(obj.get("m2_phys_sph_friction", 0.0)),
            ))
        else:  # SHOULDER
            jid = len(doc.shoulder_joints)
            doc.shoulder_joints.append(phys.ShoulderJoint(
                frame_a=_mat3x4_prop("m2_phys_shoulder_frame_a"),
                frame_b=_mat3x4_prop("m2_phys_shoulder_frame_b"),
                lower_twist=float(obj.get("m2_phys_shoulder_lower_twist", 0.0)),
                upper_twist=float(obj.get("m2_phys_shoulder_upper_twist", 0.0)),
                cone_angle=float(obj.get("m2_phys_shoulder_cone", 0.0)),
            ))

        doc.joints.append(phys.Joint(
            body_a=a, body_b=b,
            joint_type=jt, joint_id=jid,
            unk=joint_unk,
        ))

    return doc
