"""M2 Physics panel: build, tune, preview and export a WoW physics rig.

The panel follows the selection: a body shows its type, bone, mass and shapes;
a joint shows its type and limits; otherwise it shows rig creation tools.
``Start Preview`` runs the rig with Blender's rigid-body solver and makes the
skeleton follow it, so the mesh moves the way it will in game.
"""

from __future__ import annotations

import os

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup
from bpy_extras.io_utils import ExportHelper, ImportHelper
from mathutils import Matrix, Vector

from . import phys, phys_from_scene, phys_rig, phys_to_scene


# ---------------------------------------------------------------------------
# Scene settings
# ---------------------------------------------------------------------------

def _on_self_collision(self, context):
    phys_rig.stop_playback(context)
    coll = phys_rig.find_rig(context)
    if coll is not None:
        phys_rig.apply_self_collision(coll, self.self_collision)
        phys_rig.reset_simulation(context)


class M2PhysicsProps(PropertyGroup):
    shape_kind: EnumProperty(
        name="Shape", items=[i for i in phys_rig.SHAPE_KIND_ITEMS if i[0] != "POLYTOPE"],
        default="CAPSULE", description="Shape for new bodies")
    radius: FloatProperty(
        name="Radius", default=0.0, min=0.0, unit="LENGTH", precision=4,
        description="Radius for new bodies. 0 = pick one from the bone's length")
    joint_type: EnumProperty(name="Joint", items=phys_rig.JOINT_TYPE_ITEMS, default="SHOULDER",
                             description="Joint type used by Connect Selected")
    self_collision: BoolProperty(
        name="Bodies Collide With Each Other", default=False,
        description="Let simulated bodies hit one another in the preview. Off, they only "
                    "hit kinematic bodies, which is steadier for strips of cloth that sit "
                    "side by side", update=_on_self_collision)


# name -> (joint type, joint values, body values, shape values, mass,
#          shape kind or None to use the panel's setting)
_PRESETS = {
    # Values from a retail cloth buckle (buckle_panstart_a_01.phys).
    # All presets use shoulder joints with a return spring: the only joint kind
    # confirmed to work on a player model in game, and what every retail
    # mount / belt uses. Numbers come from those retail rigs.
    "CLOTH": ("SHOULDER",                                   # rostrumstormgryphon reins
              dict(lower_twist=-10.0, upper_twist=10.0, cone_angle=40.0,
                   max_motor_torque=0.0, motor_mode=0, motor_frequency_hz=1.0,
                   motor_damping_ratio=0.7),
              dict(drag=0.0, unk0=1.0, unk1=10.0, x28=0.01),
              dict(friction=0.7, restitution=0.1), 2.0, None),
    "CHAIN": ("SHOULDER",                                   # belt_leather_raidmonknerubian
              dict(lower_twist=-25.0, upper_twist=25.0, cone_angle=60.0,
                   max_motor_torque=0.0, motor_mode=0, motor_frequency_hz=1.0,
                   motor_damping_ratio=0.7),
              dict(drag=3.0, unk0=1.0, unk1=10.0, x28=0.01),
              dict(friction=0.4, restitution=0.05), 2.0, None),
    "JIGGLE": ("SHOULDER",                                  # companionnetherwingdrake dangles
               dict(lower_twist=-5.0, upper_twist=5.0, cone_angle=45.0,
                    max_motor_torque=0.0, motor_mode=1, motor_frequency_hz=3.0,
                    motor_damping_ratio=0.7),
               dict(drag=6.0, unk0=0.25, unk1=6.0, x28=0.01),
               dict(friction=0.5, restitution=0.0), 1.0, None),
    # Chest, belly, thighs: a small cone so it cannot flap, a firm spring back
    # to rest and plenty of drag, so wind and armour only nudge it.
    "BODY": ("SHOULDER",
             dict(lower_twist=-3.0, upper_twist=3.0, cone_angle=12.0,
                  max_motor_torque=0.0, motor_mode=1, motor_frequency_hz=3.0,
                  motor_damping_ratio=0.7),
             dict(drag=8.0, unk0=0.1, unk1=10.0, x28=0.01),
             dict(friction=0.5, restitution=0.0), 0.8, "SPHERE"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _object_mode:
    """Run in Object Mode, then return to the mode the user was in."""

    def __init__(self, context):
        self.context = context

    def __enter__(self):
        self.prev = self.context.mode
        self.active = self.context.view_layer.objects.active
        if self.prev != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except RuntimeError:
                pass

    def __exit__(self, *exc):
        if self.prev == "OBJECT":
            return
        try:
            self.context.view_layer.objects.active = self.active
            bpy.ops.object.mode_set(mode="EDIT" if self.prev.startswith("EDIT") else self.prev)
        except (RuntimeError, ReferenceError):
            pass


class at_rest:
    """Hold the scene on the simulation's first frame so bodies are read at rest."""

    def __init__(self, context):
        self.scene = context.scene

    def __enter__(self):
        self.frame = self.scene.frame_current
        rbw = self.scene.rigidbody_world
        start = rbw.point_cache.frame_start if rbw is not None else self.scene.frame_start
        if self.frame != start:
            self.scene.frame_set(start)
        else:
            self.frame = None

    def __exit__(self, *exc):
        if self.frame is not None:
            self.scene.frame_set(self.frame)


def _bone_direction(arm, bone, preferred=()):
    """Armature-space vector along which a body on ``bone`` should lie. Imported
    M2 bones all point the same way, so the next joint's pivot is what counts."""
    head = bone.head_local
    children = [c for c in bone.children if (c.head_local - head).length > 1e-4]
    pick = next((c for c in children if c.name in preferred), None) \
        or (children[0] if children else None)
    if pick is not None:
        return pick.head_local - head
    if bone.parent is not None and (head - bone.parent.head_local).length > 1e-4:
        return head - bone.parent.head_local
    return bone.tail_local - head


def _weighted_points(arm, bone_name, min_weight=0.4):
    """Armature-space positions of the mesh vertices this bone drives."""
    arm_inv = arm.matrix_world.inverted()
    pts = []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or phys_rig.is_rig_object(obj) or obj.get("m2_bounding_box"):
            continue
        if not (obj.parent == arm or any(m.type == "ARMATURE" and m.object == arm
                                         for m in obj.modifiers)):
            continue
        vg = obj.vertex_groups.get(bone_name)
        if vg is None:
            continue
        gi = vg.index
        to_arm = arm_inv @ obj.matrix_world
        for v in obj.data.vertices:
            for g in v.groups:
                if g.group == gi:
                    if g.weight >= min_weight:
                        pts.append(to_arm @ v.co)
                    break
    return pts


def _fit_to_mesh(arm, bone, outward):
    """Fit (centre, axis, half-length, radius) to the bone's skinned vertices,
    or None when the bone drives too little mesh to measure."""
    pts = _weighted_points(arm, bone.name)
    if len(pts) < 8:
        return None
    import numpy as np
    P = np.array([tuple(p) for p in pts], dtype=float)
    c = P.mean(axis=0)
    Q = P - c
    _, vecs = np.linalg.eigh(Q.T @ Q)
    axis = vecs[:, -1]
    if float(np.dot(axis, np.array(tuple(outward)))) < 0.0:
        axis = -axis
    proj = Q @ axis
    lo, hi = float(proj.min()), float(proj.max())
    radial = np.linalg.norm(Q - np.outer(proj, axis), axis=1)
    r = float(np.percentile(radial, 90))
    centre = Vector(c) + Vector(axis) * ((lo + hi) * 0.5)
    return centre, Vector(axis), (hi - lo) * 0.5, max(r, 0.003)


def _is_chain_bone(bone):
    return any((c.head_local - bone.head_local).length > 1e-4 for c in bone.children)


def _make_bone_body(context, coll, arm, bone, body_type, kind, radius, preferred=()):
    """A body on ``bone``. Shapes are fitted to the mesh the bone deforms when
    it has one: a chain bone gets a capsule along the chain with the mesh's
    thickness, a leaf bone (a breast, an ear, a tassel tip) gets a shape
    sitting on the flesh itself rather than projected out along the bone."""
    props_scene = context.scene.m2_physics
    direction = _bone_direction(arm, bone, preferred)
    length = max(direction.length, 0.01)
    unit = direction.normalized() if direction.length > 1e-6 else Vector((0, 0, 1))
    head = bone.head_local
    world = arm.matrix_world @ Matrix.Translation(head)
    prefix = {"ROOT": "phys_body_root", "KINEMATIC": "phys_coll_"}.get(body_type, "phys_body_")
    name = prefix if body_type == "ROOT" else prefix + bone.name
    obj = phys_rig.new_body(context, coll, name, world, body_type, bone.name)
    if body_type == "ROOT":
        phys_rig.rebuild_body(context, obj)
        return obj, unit

    is_chain = _is_chain_bone(bone)
    fit = _fit_to_mesh(arm, bone, unit)
    if fit is not None and not is_chain:
        centre, axis, half_len, r = fit
        unit = axis
        r = radius or props_scene.radius or r
        local_c = centre - head
        if kind == "SPHERE" or (kind == "CAPSULE" and half_len < 1.2 * r):
            phys_rig.add_shape(obj, "SPHERE", p1=local_c, radius=max(r, half_len * 0.8))
        elif kind == "CAPSULE":
            phys_rig.add_shape(obj, "CAPSULE", p1=local_c - axis * (half_len - r),
                               p2=local_c + axis * (half_len - r), radius=r)
        else:
            axes = phys_rig.matrix_to_mat3x4(phys_rig.frame_from_z(axis).to_4x4())[0:9]
            phys_rig.add_shape(obj, "BOX", p1=local_c, box_axes=axes,
                               half_extents=(r, r, max(half_len, r)))
    else:
        r = radius or props_scene.radius or (fit[3] if fit is not None
                                              else max(0.005, min(length * 0.2, 0.08)))
        if not is_chain and fit is None:
            length = min(length, max(r * 2.0, 0.02))      # leaf bone with no mesh to measure
        if kind == "CAPSULE":
            inset = min(r, length * 0.25)
            phys_rig.add_shape(obj, "CAPSULE", p1=unit * inset, p2=unit * (length - inset), radius=r)
        elif kind == "SPHERE":
            phys_rig.add_shape(obj, "SPHERE", p1=unit * (length * 0.5), radius=max(r, length * 0.5))
        else:
            axes = phys_rig.matrix_to_mat3x4(phys_rig.frame_from_z(unit).to_4x4())[0:9]
            phys_rig.add_shape(obj, "BOX", p1=unit * (length * 0.5), box_axes=axes,
                               half_extents=(r, r, length * 0.5))
    phys_rig.rebuild_body(context, obj)
    return obj, unit


def _anchor_for(context, coll, arm, bone):
    """The body a chain starting at ``bone`` should hang from, made if needed."""
    ancestor = bone.parent
    while ancestor is not None:
        body = phys_rig.body_for_bone(coll, ancestor.name)
        if body is not None:
            return body
        ancestor = ancestor.parent
    root = phys_rig.root_body(coll)
    from . import from_scene
    anchor_bone = bone.parent or from_scene._topo_bones(arm.data)[0]
    if anchor_bone == bone:
        anchor_bone = next((b for b in arm.data.bones if b != bone), bone)
    if root is None:
        return _make_bone_body(context, coll, arm, anchor_bone, "ROOT", "CAPSULE", 0.0)[0]
    if root.m2_phys_body.bone == anchor_bone.name or bone.parent is None:
        return root
    existing = phys_rig.body_for_bone(coll, anchor_bone.name)
    if existing is not None:
        return existing
    # A second chain hanging off a different bone: a shapeless kinematic body
    # there, so the chain follows THAT bone instead of the root's.
    obj = phys_rig.new_body(context, coll, "phys_coll_" + anchor_bone.name,
                            arm.matrix_world @ Matrix.Translation(anchor_bone.head_local),
                            "KINEMATIC", anchor_bone.name)
    phys_rig.rebuild_body(context, obj)
    return obj


def _connect(context, coll, arm, body_a, body_b, joint_type, z_axis=None, **values):
    world_b = phys_rig.body_frame_world(body_b)
    if z_axis is None:
        z_axis = world_b.translation - phys_rig.body_frame_world(body_a).translation
    rot = (arm.matrix_world.to_3x3() if arm is not None else Matrix.Identity(3)) \
        @ phys_rig.frame_from_z(z_axis)
    world = Matrix.Translation(world_b.translation) @ rot.normalized().to_4x4()
    name = "phys_joint_" + (body_b.m2_phys_body.bone or body_b.name)
    return phys_rig.new_joint(context, coll, name, world, body_a, body_b, joint_type, **values)


def _topo_sorted(bones):
    def depth(b):
        d = 0
        while b.parent is not None:
            d, b = d + 1, b.parent
        return d
    return sorted(bones, key=lambda b: (depth(b), b.name))


def _refresh(context, coll):
    phys_rig.apply_self_collision(coll, context.scene.m2_physics.self_collision)
    phys_rig.sanitize_constraints(context.scene, context.view_layer)
    phys_rig.reset_simulation(context)


def _begin_edit(context, coll=None):
    """Every operator that changes the rig starts here: stop the preview so
    Bullet is idle, and drop the bone-follow constraints so the skeleton is
    back at rest while we measure and rebuild."""
    phys_rig.stop_playback(context)
    if coll is None:
        coll = phys_rig.find_rig(context)
    if coll is not None:
        phys_rig.clear_follow(coll.m2_phys_rig.armature, coll)
    context.scene.frame_set(context.scene.frame_start)


def _previewing(arm) -> bool:
    if arm is None or arm.pose is None:
        return False
    return any(c.name == phys_rig.FOLLOW_CONSTRAINT
               for pb in arm.pose.bones for c in pb.constraints)


# ---------------------------------------------------------------------------
# Operators: building
# ---------------------------------------------------------------------------

class M2PHYS_OT_bone_preset(Operator):
    """Rig the selected pose bones in one click: a body on each bone, joints
    between them, and a root anchor if the rig doesn't have one"""
    bl_idname = "m2phys.bone_preset"
    bl_label = "Rig Selected Bones"
    bl_options = {"REGISTER", "UNDO"}

    preset: EnumProperty(items=[("CLOTH", "Cloth", "Swings in a cone with drag: cloaks, tabards, hair"),
                                ("CHAIN", "Chain", "Free-swinging links: chains, pendants"),
                                ("JIGGLE", "Jiggle", "Springs back to rest: stiff cloth, ears, tails"),
                                ("BODY", "Soft Body", "Gentle bounce for chest, belly and other soft parts")])

    @classmethod
    def poll(cls, context):
        return context.mode == "POSE" and bool(context.selected_pose_bones)

    def execute(self, context):
        arm = context.active_object
        names = [pb.name for pb in context.selected_pose_bones]
        joint_type, joint_values, body_values, shape_values, mass, shape_kind = _PRESETS[self.preset]
        kind = shape_kind or context.scene.m2_physics.shape_kind
        made = 0
        with _object_mode(context), at_rest(context):
            _begin_edit(context, phys_rig.find_rig(context, arm))
            coll = phys_rig.ensure_rig(context, arm)
            for bone in _topo_sorted([arm.data.bones[n] for n in names]):
                body = phys_rig.body_for_bone(coll, bone.name)
                if body is not None and body.m2_phys_body.body_type != "DYNAMIC":
                    self.report({"WARNING"}, "'%s' already has a %s body; skipped"
                                % (bone.name, body.m2_phys_body.body_type.lower()))
                    continue
                anchor = _anchor_for(context, coll, arm, bone)
                if body is None:
                    body, unit = _make_bone_body(context, coll, arm, bone, "DYNAMIC", kind, 0.0, names)
                    _connect(context, coll, arm, anchor, body, joint_type,
                             z_axis=unit, **joint_values)
                    made += 1
                else:
                    for j in phys_rig.joints_of(coll, body):
                        if j.m2_phys_joint.body_b == body:
                            with phys_rig.suppress_updates():
                                j.m2_phys_joint.joint_type = joint_type
                                for k, v in joint_values.items():
                                    setattr(j.m2_phys_joint, k, v)
                            phys_rig.sync_constraint(context, j)
                with phys_rig.suppress_updates():
                    for k, v in body_values.items():
                        setattr(body.m2_phys_body, k, v)
                    for s in body.m2_phys_body.shapes:
                        for k, v in shape_values.items():
                            setattr(s, k, v)
                phys_rig.set_body_mass(body, mass)
                phys_rig.rebuild_body(context, body)
            for body in phys_rig.rig_bodies(coll):    # anchors depend on the set of dynamic bones
                phys_rig.rebuild_body(context, body)
            for j in phys_rig.rig_joints(coll):       # spring strength depends on mass
                phys_rig.sync_constraint(context, j)
            _refresh(context, coll)
        self.report({"INFO"}, "%s: %d new bodies on %d bones (build %s)"
                    % (self.preset.title(), made, len(names), phys_rig.BUILD))
        return {"FINISHED"}


class M2PHYS_OT_add_body(Operator):
    """Add one body on the active pose bone"""
    bl_idname = "m2phys.add_body"
    bl_label = "Add Body"
    bl_options = {"REGISTER", "UNDO"}

    body_type: EnumProperty(items=[i for i in phys_rig.BODY_TYPE_ITEMS if i[0] != "ROOT"],
                            default="DYNAMIC")

    @classmethod
    def poll(cls, context):
        return context.mode == "POSE" and context.active_pose_bone is not None

    def execute(self, context):
        arm = context.active_object
        bone = arm.data.bones[context.active_pose_bone.name]
        with _object_mode(context), at_rest(context):
            _begin_edit(context, phys_rig.find_rig(context, arm))
            coll = phys_rig.ensure_rig(context, arm)
            if self.body_type == "DYNAMIC" and phys_rig.body_for_bone(coll, bone.name) is not None:
                self.report({"ERROR"}, "Bone '%s' already has a body." % bone.name)
                return {"CANCELLED"}
            obj, _ = _make_bone_body(context, coll, arm, bone, self.body_type,
                                     context.scene.m2_physics.shape_kind, 0.0)
            _refresh(context, coll)
        self.report({"INFO"}, "Added %s. Connect it to another body with a joint." % obj.name)
        return {"FINISHED"}


class M2PHYS_OT_connect(Operator):
    """Join the two selected bodies. The active body becomes the child"""
    bl_idname = "m2phys.connect"
    bl_label = "Connect Selected"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return len([o for o in context.selected_objects if phys_rig.is_body(o)]) == 2 \
            and phys_rig.is_body(context.active_object)

    def execute(self, context):
        child = context.active_object
        parent = next(o for o in context.selected_objects
                      if phys_rig.is_body(o) and o != child)
        coll = next((c for c in child.users_collection if c.m2_phys_rig.is_rig), None)
        if coll is None or coll not in parent.users_collection:
            self.report({"ERROR"}, "Both bodies must belong to the same rig.")
            return {"CANCELLED"}
        if any(j.m2_phys_joint.body_a in (child, parent) and j.m2_phys_joint.body_b in (child, parent)
               for j in phys_rig.rig_joints(coll)):
            self.report({"ERROR"}, "These two bodies are already connected.")
            return {"CANCELLED"}
        _begin_edit(context, coll)
        with at_rest(context):
            frame = phys_rig.body_frame_world(child)
            z_axis = (frame @ phys_rig.body_com_local(child)) - frame.translation
            _connect(context, coll, coll.m2_phys_rig.armature, parent, child,
                     context.scene.m2_physics.joint_type,
                     z_axis=z_axis if z_axis.length > 1e-5 else None)
            _refresh(context, coll)
        return {"FINISHED"}


class M2PHYS_OT_set_root(Operator):
    """Make the active body the rig's Root Anchor (written first in the file).
    The old root becomes an ordinary anchor; both follow their bones"""
    bl_idname = "m2phys.set_root"
    bl_label = "Make Root Anchor"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return phys_rig.is_body(context.active_object)

    def execute(self, context):
        obj = context.active_object
        coll = next((c for c in obj.users_collection if c.m2_phys_rig.is_rig), None)
        _begin_edit(context, coll)
        with at_rest(context):
            for other in (phys_rig.rig_bodies(coll) if coll else []):
                if other != obj and other.m2_phys_body.body_type == "ROOT":
                    other.m2_phys_body.body_type = "KINEMATIC"
            obj.m2_phys_body.body_type = "ROOT"
            if coll is not None:
                _refresh(context, coll)
        return {"FINISHED"}


class M2PHYS_OT_add_shape(Operator):
    """Add another shape to the active body"""
    bl_idname = "m2phys.add_shape"
    bl_label = "Add Shape"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return phys_rig.is_body(context.active_object)

    def execute(self, context):
        obj = context.active_object
        _begin_edit(context)
        shapes = obj.m2_phys_body.shapes
        kind = context.scene.m2_physics.shape_kind
        values = {}
        if shapes:                         # start from the last shape's material
            last = shapes[-1]
            values = dict(friction=last.friction, restitution=last.restitution,
                          density=last.density)
        phys_rig.add_shape(obj, kind, **values)
        phys_rig.rebuild_body(context, obj)
        return {"FINISHED"}


class M2PHYS_OT_remove_shape(Operator):
    """Remove this shape from the body"""
    bl_idname = "m2phys.remove_shape"
    bl_label = "Remove Shape"
    bl_options = {"REGISTER", "UNDO"}

    index: bpy.props.IntProperty()

    def execute(self, context):
        obj = context.active_object
        if not phys_rig.is_body(obj) or not 0 <= self.index < len(obj.m2_phys_body.shapes):
            return {"CANCELLED"}
        _begin_edit(context)
        obj.m2_phys_body.shapes.remove(self.index)
        phys_rig.rebuild_body(context, obj)
        return {"FINISHED"}


class M2PHYS_OT_fit_to_bone(Operator):
    """Put the body back on its bone and fit its first shape to the mesh the
    bone deforms (or lay it along the bone when it drives no mesh)"""
    bl_idname = "m2phys.fit_to_bone"
    bl_label = "Fit to Mesh"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return phys_rig.is_body(context.active_object)

    def execute(self, context):
        obj = context.active_object
        props = obj.m2_phys_body
        arm = phys_rig._rig_armature(obj)
        bone = arm.data.bones.get(props.bone) if arm is not None else None
        if bone is None:
            self.report({"ERROR"}, "Assign a bone first.")
            return {"CANCELLED"}
        _begin_edit(context)
        with at_rest(context):
            direction = _bone_direction(arm, bone)
            length = max(direction.length, 0.01)
            unit = direction.normalized()
            phys_rig.set_body_frame_world(
                obj, arm.matrix_world @ Matrix.Translation(bone.head_local))
            fit = None if _is_chain_bone(bone) else _fit_to_mesh(arm, bone, unit)
            with phys_rig.suppress_updates():
                props.has_file_position = False
                for shape in props.shapes:
                    if fit is not None:
                        centre, axis, half_len, r = fit
                        local_c = centre - bone.head_local
                        if shape.kind == "CAPSULE":
                            reach = max(half_len - shape.radius, 0.0)
                            shape.p1, shape.p2 = local_c - axis * reach, local_c + axis * reach
                        else:
                            shape.p1 = local_c
                    elif shape.kind == "CAPSULE":
                        inset = min(shape.radius, length * 0.25)
                        shape.p1, shape.p2 = unit * inset, unit * (length - inset)
                    break
            phys_rig.rebuild_body(context, obj)
            phys_rig.reset_simulation(context)
        return {"FINISHED"}


class M2PHYS_OT_delete(Operator):
    """Delete the selected bodies (with their joints) and joints"""
    bl_idname = "m2phys.delete"
    bl_label = "Delete Selected"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return any(phys_rig.is_body(o) or phys_rig.is_joint(o) for o in context.selected_objects)

    def execute(self, context):
        _begin_edit(context)
        doomed = set()
        colls = set()
        for obj in context.selected_objects:
            if phys_rig.is_joint(obj):
                doomed.add(obj)
            elif phys_rig.is_body(obj):
                doomed.add(obj)
                for coll in obj.users_collection:
                    if coll.m2_phys_rig.is_rig:
                        doomed.update(phys_rig.joints_of(coll, obj))
            colls.update(c for c in obj.users_collection if c.m2_phys_rig.is_rig)
        for coll in colls:
            phys_rig.clear_follow(coll.m2_phys_rig.armature, coll)
        n = len(doomed)
        for obj in doomed:
            phys_rig.remove_object(obj)
        phys_rig.reset_simulation(context)
        self.report({"INFO"}, "Deleted %d object(s)." % n)
        return {"FINISHED"}


class M2PHYS_OT_remove_rig(Operator):
    """Delete the whole physics rig"""
    bl_idname = "m2phys.remove_rig"
    bl_label = "Remove Physics Rig"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return phys_rig.find_rig(context) is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        coll = phys_rig.find_rig(context)
        _begin_edit(context, coll)
        for obj in list(coll.objects):
            phys_rig.remove_object(obj)
        bpy.data.collections.remove(coll)
        phys_rig.reset_simulation(context)
        return {"FINISHED"}


class M2PHYS_OT_select_rig(Operator):
    """Unhide the rig, select it and frame it in the viewport"""
    bl_idname = "m2phys.select_rig"
    bl_label = "Show Rig"
    bl_options = {"REGISTER", "UNDO"}

    collection_name: StringProperty()

    def execute(self, context):
        coll = bpy.data.collections.get(self.collection_name)
        if coll is None:
            return {"CANCELLED"}

        def unexclude(layer):
            if layer.collection == coll:
                layer.exclude = False
                layer.hide_viewport = False
            for child in layer.children:
                unexclude(child)
        unexclude(context.view_layer.layer_collection)
        coll.hide_viewport = False
        with _object_mode(context):
            for o in context.view_layer.objects:
                o.select_set(False)
            for o in coll.objects:
                if o.get("m2_phys_follow_helper"):
                    continue
                o.hide_set(False)
                o.hide_viewport = False
                o.select_set(True)
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Operators: preview
# ---------------------------------------------------------------------------

class M2PHYS_OT_preview_start(Operator):
    """Simulate the rig and make the skeleton follow it, so the model moves the
    way it will in game. Assign an animation to the armature first to see the
    rig react to movement"""
    bl_idname = "m2phys.preview_start"
    bl_label = "Start Preview"

    @classmethod
    def poll(cls, context):
        return phys_rig.find_rig(context) is not None

    def execute(self, context):
        coll = phys_rig.find_rig(context)
        errors, warnings = phys_from_scene.validate(coll)
        if errors:
            self.report({"ERROR"}, errors[0])
            return {"CANCELLED"}
        phys_rig.stop_playback(context)
        phys_rig.ensure_world(context)
        arm = coll.m2_phys_rig.armature
        phys_rig.clear_follow(arm, coll)
        context.scene.frame_set(context.scene.frame_start)
        # Hidden bodies drop out of the simulation and their joints would
        # point at nothing: unhide the rig so what you see is what runs.
        for obj in phys_rig.rig_bodies(coll) + phys_rig.rig_joints(coll):
            try:
                obj.hide_set(False)
            except RuntimeError:
                pass
            obj.hide_viewport = False
        phys_rig.apply_self_collision(coll, context.scene.m2_physics.self_collision)
        phys_rig.sanitize_constraints(context.scene, context.view_layer)
        phys_rig.reset_simulation(context)
        wired = phys_rig.wire_follow(coll)
        try:
            bpy.ops.screen.animation_play()
        except RuntimeError:
            pass
        msg = "Preview running (build %s): %d bones follow their bodies." % (phys_rig.BUILD, wired)
        if warnings:
            msg += " " + warnings[0]
        self.report({"WARNING"} if warnings else {"INFO"}, msg)
        return {"FINISHED"}


class M2PHYS_OT_preview_stop(Operator):
    """Stop the preview and put the skeleton back"""
    bl_idname = "m2phys.preview_stop"
    bl_label = "Stop Preview"

    def execute(self, context):
        phys_rig.stop_playback(context)
        for coll in phys_rig.rig_collections():
            phys_rig.clear_follow(coll.m2_phys_rig.armature, coll)
        context.scene.frame_set(context.scene.frame_start)
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Operators: files
# ---------------------------------------------------------------------------

class M2PHYS_OT_import_phys(Operator, ImportHelper):
    """Load a .phys file onto the active armature, replacing its current rig"""
    bl_idname = "m2phys.import_phys"
    bl_label = "Import .phys"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".phys"
    filter_glob: StringProperty(default="*.phys", options={"HIDDEN"})
    mirror_x: BoolProperty(name="Mirror X", default=False,
                           description="Match the setting the model was imported with")

    def execute(self, context):
        arm = phys_rig.active_armature(context)
        if arm is None:
            self.report({"ERROR"}, "Select the model's armature first.")
            return {"CANCELLED"}
        name = os.path.splitext(os.path.basename(self.filepath))[0]
        _begin_edit(context)
        with _object_mode(context):
            label = phys_to_scene.load_phys_into_scene(
                context, "", arm, name, mirror_x=self.mirror_x, explicit_path=self.filepath)
        if label is None:
            self.report({"ERROR"}, "No physics bodies found in that file.")
            return {"CANCELLED"}
        self.report({"INFO"}, "Imported physics from %s" % os.path.basename(self.filepath))
        return {"FINISHED"}


class M2PHYS_OT_export_phys(Operator, ExportHelper):
    """Write just the physics rig to a .phys file"""
    bl_idname = "m2phys.export_phys"
    bl_label = "Export .phys"

    filename_ext = ".phys"
    filter_glob: StringProperty(default="*.phys", options={"HIDDEN"})
    mirror_x: BoolProperty(name="Mirror X", default=False,
                           description="Match the setting the model was imported with")

    @classmethod
    def poll(cls, context):
        return phys_rig.find_rig(context) is not None

    def execute(self, context):
        coll = phys_rig.find_rig(context)
        try:
            with at_rest(context):
                data = phys_from_scene.build_phys_bytes(
                    context, coll.m2_phys_rig.armature, mirror_x=self.mirror_x, collection=coll)
        except phys_from_scene.RigError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        with open(self.filepath, "wb") as f:
            f.write(data)
        self.report({"INFO"}, "Wrote %s (%d bytes)" % (os.path.basename(self.filepath), len(data)))
        return {"FINISHED"}


class M2PHYS_OT_validate(Operator):
    """Check the rig for problems that would break it in game"""
    bl_idname = "m2phys.validate"
    bl_label = "Check Rig"

    @classmethod
    def poll(cls, context):
        return phys_rig.find_rig(context) is not None

    def execute(self, context):
        errors, warnings = phys_from_scene.validate(phys_rig.find_rig(context))
        for e in errors:
            self.report({"ERROR"}, e)
        for w in warnings:
            self.report({"WARNING"}, w)
        if not errors and not warnings:
            self.report({"INFO"}, "Rig is valid and ready to export.")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class VIEW3D_PT_m2_physics(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "M2"
    bl_label = "M2 Physics"
    bl_idname = "VIEW3D_PT_m2_physics"

    def draw(self, context):
        layout = self.layout
        scene_props = context.scene.m2_physics
        active = context.active_object
        coll = phys_rig.find_rig(context)

        self._draw_status(layout, context, coll)
        if phys_rig.is_body(active):
            self._draw_body(layout, context, active)
        elif phys_rig.is_joint(active):
            self._draw_joint(layout, active)
        self._draw_build(layout, context, scene_props)
        if coll is not None:
            self._draw_preview(layout, context, coll, scene_props)
            self._draw_files(layout, coll)

    # -- sections ----------------------------------------------------------

    def _draw_status(self, layout, context, coll):
        box = layout.box()
        rigs = phys_rig.rig_collections()
        legacy = [c for c in bpy.data.collections
                  if c.name.endswith(phys_rig.COLLECTION_SUFFIX) and not c.m2_phys_rig.is_rig
                  and any(o.rigid_body is not None for o in c.objects)]
        if legacy:
            col = box.column(align=True)
            col.label(text="Old-format rig: %s" % legacy[0].name, icon="ERROR")
            col.label(text="Made by an earlier version. Delete it and")
            col.label(text="re-import the physics, or rebuild it here.")
        if not rigs:
            box.label(text="No physics rig in this scene", icon="INFO")
            box.operator("m2phys.import_phys", icon="IMPORT")
            return
        for rig in rigs:
            row = box.row(align=True)
            row.label(text="%s: %d bodies, %d joints"
                      % (rig.name, len(phys_rig.rig_bodies(rig)), len(phys_rig.rig_joints(rig))),
                      icon="RIGID_BODY" if rig == coll else "DOT")
            row.operator("m2phys.select_rig", text="", icon="RESTRICT_SELECT_OFF"
                         ).collection_name = rig.name

    def _draw_body(self, layout, context, obj):
        props = obj.m2_phys_body
        box = layout.box()
        box.label(text=obj.name, icon="RIGID_BODY")
        col = box.column(align=True)
        col.prop(props, "body_type")
        arm = phys_rig._rig_armature(obj)
        if arm is not None:
            col.prop_search(props, "bone", arm.data, "bones", text="Bone")
        else:
            col.prop(props, "bone")
        if props.body_type == "DYNAMIC":
            col.prop(props, "mass")
            col.prop(props, "drag")
        row = box.row(align=True)
        row.operator("m2phys.fit_to_bone", icon="BONE_DATA")
        if props.body_type != "ROOT":
            row.operator("m2phys.set_root", icon="PINNED")

        for i, s in enumerate(props.shapes):
            sbox = box.box()
            head = sbox.row(align=True)
            head.prop(s, "kind", text="")
            head.operator("m2phys.remove_shape", text="", icon="X").index = i
            if s.kind == "POLYTOPE":
                sbox.label(text="Geometry is kept from the file", icon="INFO")
            else:
                scol = sbox.column(align=True)
                if s.kind == "CAPSULE":
                    scol.prop(s, "p1")
                    scol.prop(s, "p2")
                    scol.prop(s, "radius")
                elif s.kind == "SPHERE":
                    scol.prop(s, "p1", text="Centre")
                    scol.prop(s, "radius")
                else:
                    scol.prop(s, "p1", text="Centre")
                    scol.prop(s, "half_extents")
            mat = sbox.column(align=True)
            mat.prop(s, "friction")
            mat.prop(s, "restitution")
            if len(props.shapes) > 1:
                mat.prop(s, "density")
        box.operator("m2phys.add_shape", icon="ADD")
        box.label(text="Move, rotate or scale it freely: export bakes that in.", icon="INFO")

    def _draw_joint(self, layout, empty):
        props = empty.m2_phys_joint
        box = layout.box()
        box.label(text=empty.name, icon="CONSTRAINT")
        col = box.column(align=True)
        col.prop(props, "joint_type")
        col.prop(props, "body_a")
        col.prop(props, "body_b")
        col = box.column(align=True)
        kind = props.joint_type
        if kind == "SHOULDER":
            col.prop(props, "cone_angle")
            col.prop(props, "lower_twist")
            col.prop(props, "upper_twist")
            col.separator()
            col.prop(props, "motor_frequency_hz")
            col.prop(props, "motor_damping_ratio")
            col.prop(props, "motor_mode")
        elif kind == "SPHERICAL":
            col.prop(props, "friction_torque")
        elif kind == "WELD":
            col.prop(props, "angular_frequency_hz")
            col.prop(props, "angular_damping_ratio")
            col.prop(props, "linear_frequency_hz")
            col.prop(props, "linear_damping_ratio")
        elif kind in ("REVOLUTE", "PRISMATIC"):
            col.prop(props, "lower_limit")
            col.prop(props, "upper_limit")
            col.prop(props, "max_motor_torque")
            col.prop(props, "motor_mode")
        else:
            col.prop(props, "distance_factor")
        if kind in ("SHOULDER", "REVOLUTE", "PRISMATIC"):
            box.label(text="The blue Z arrow is the joint's axis.", icon="INFO")
        if kind in ("REVOLUTE", "PRISMATIC", "DISTANCE"):
            box.label(text="Preview of this joint type is approximate.", icon="ERROR")
        if kind == "WELD":
            box.label(text="Welds did not work on a player model in game.", icon="ERROR")
            box.label(text="Use Shoulder with Stiffness instead.")

    def _draw_build(self, layout, context, scene_props):
        box = layout.box()
        box.label(text="Build", icon="BONE_DATA")
        if context.mode == "POSE":
            n = len(context.selected_pose_bones or ())
            box.label(text="Rig %d selected bone%s as:" % (n, "" if n == 1 else "s"))
            row = box.row(align=True)
            row.operator("m2phys.bone_preset", text="Cloth", icon="MOD_CLOTH").preset = "CLOTH"
            row.operator("m2phys.bone_preset", text="Chain", icon="LINKED").preset = "CHAIN"
            row.operator("m2phys.bone_preset", text="Jiggle", icon="FORCE_HARMONIC").preset = "JIGGLE"
            row.operator("m2phys.bone_preset", text="Soft Body", icon="SPHERE").preset = "BODY"
            row = box.row(align=True)
            row.operator("m2phys.add_body", text="Add Body", icon="ADD").body_type = "DYNAMIC"
            row.operator("m2phys.add_body", text="Add Collider", icon="MESH_CAPSULE").body_type = "KINEMATIC"
        else:
            box.label(text="Select the armature and enter Pose Mode", icon="INFO")
            box.label(text="to add bodies to bones.")
        row = box.row(align=True)
        row.prop(scene_props, "shape_kind", text="")
        row.prop(scene_props, "radius")
        row = box.row(align=True)
        row.prop(scene_props, "joint_type", text="")
        row.operator("m2phys.connect", icon="LINKED")
        box.operator("m2phys.delete", icon="TRASH")

    def _draw_preview(self, layout, context, coll, scene_props):
        box = layout.box()
        box.label(text="Preview", icon="PLAY")
        if _previewing(coll.m2_phys_rig.armature):
            box.operator("m2phys.preview_stop", icon="PAUSE", depress=True)
            box.label(text="Stop the preview before editing the rig.", icon="INFO")
        else:
            box.operator("m2phys.preview_start", icon="PLAY")
        box.prop(scene_props, "self_collision")
        rbw = context.scene.rigidbody_world
        if rbw is not None:
            col = box.column(align=True)
            if hasattr(rbw, "substeps_per_frame"):
                col.prop(rbw, "substeps_per_frame", text="Substeps")
            col.prop(rbw, "solver_iterations", text="Solver Iterations")

    def _draw_files(self, layout, coll):
        box = layout.box()
        box.label(text="Export", icon="EXPORT")
        box.prop(coll.m2_phys_rig, "version", text="Format Version")
        box.operator("m2phys.validate", icon="CHECKMARK")
        row = box.row(align=True)
        row.operator("m2phys.import_phys", icon="IMPORT")
        row.operator("m2phys.export_phys", icon="EXPORT")
        box.label(text="Exporting the M2 embeds the physics.", icon="INFO")
        box.label(text="Physics build %s" % phys_rig.BUILD)
        box.operator("m2phys.remove_rig", icon="TRASH")


_classes = (
    M2PhysicsProps,
    M2PHYS_OT_bone_preset,
    M2PHYS_OT_add_body,
    M2PHYS_OT_connect,
    M2PHYS_OT_set_root,
    M2PHYS_OT_add_shape,
    M2PHYS_OT_remove_shape,
    M2PHYS_OT_fit_to_bone,
    M2PHYS_OT_delete,
    M2PHYS_OT_remove_rig,
    M2PHYS_OT_select_rig,
    M2PHYS_OT_preview_start,
    M2PHYS_OT_preview_stop,
    M2PHYS_OT_import_phys,
    M2PHYS_OT_export_phys,
    M2PHYS_OT_validate,
    VIEW3D_PT_m2_physics,
)


def register():
    phys_rig.register_props()
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.m2_physics = bpy.props.PointerProperty(type=M2PhysicsProps)


def unregister():
    if hasattr(bpy.types.Scene, "m2_physics"):
        del bpy.types.Scene.m2_physics
    for cls in reversed(_classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
    phys_rig.unregister_props()
