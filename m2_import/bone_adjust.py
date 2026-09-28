"""Bone Adjust: move, rotate or scale bones across every animation at once.

Typical uses: scale the hand / shoulder / sheath attachment bones so equipped
weapons and shoulder armour come out bigger or smaller, or turn the upper
arms outward so they clear the legs in every clip.

Workflow: ``start`` pauses the armature's animation and shows the rest pose;
the user edits bones in pose mode; ``apply`` reads each bone's offset from
the rest pose and bakes it into every animation clip; ``cancel`` throws the
edits away. Either way the armature gets its clip back.

The offset is combined with each clip's own keys as

    location' = offset_loc + offset_rot @ location
    rotation' = offset_rot @ rotation
    scale'    = scale * offset_scale          (per axis)

i.e. the bone is animated as before and the result is then turned / moved
about the bone's head in its rest frame, with the scale taken along the
bone's own axes. What you see on the rest pose is what every clip gets, a
rotation offset stays fixed relative to the parent bone (arms pushed out
from the torso stay out whatever the swing), and the result is still a
plain location / rotation / scale key, which is all an M2 track can hold.
"""
import json

import bpy
from mathutils import Quaternion, Vector

STATE_KEY = "m2_bone_adjust"          # JSON on the armature while adjusting
EPS = 1e-5


# ---------------------------------------------------------------------------
# fcurve access across the legacy and the slotted (4.4+ / 5.x) action API
def _iter_fcurves(action):
    layers = getattr(action, "layers", None) or ()
    found = False
    for layer in layers:
        for strip in getattr(layer, "strips", ()) or ():
            for cbag in getattr(strip, "channelbags", ()) or ():
                for fc in cbag.fcurves:
                    found = True
                    yield fc
    if not found:
        for fc in getattr(action, "fcurves", ()) or ():
            yield fc


def _fcurve_collection(action):
    """The collection new fcurves are created in."""
    layers = getattr(action, "layers", None)
    if layers is not None and len(layers):
        layer = layers[0]
        strip = layer.strips[0] if len(layer.strips) else layer.strips.new(type="KEYFRAME")
        slots = action.slots
        slot = slots[0] if len(slots) else action.slots.new(id_type="OBJECT", name="Object")
        try:
            return strip.channelbag(slot, ensure=True).fcurves
        except TypeError:
            return strip.channelbag(slot).fcurves
    if layers is not None and hasattr(action, "slots"):
        slot = action.slots[0] if len(action.slots) else action.slots.new(id_type="OBJECT", name="Object")
        layer = action.layers.new("Layer")
        strip = layer.strips.new(type="KEYFRAME")
        return strip.channelbag(slot, ensure=True).fcurves
    return action.fcurves


def _channel(action, bone, prop, n):
    path = 'pose.bones["%s"].%s' % (bone, prop)
    fcs = [None] * n
    for fc in _iter_fcurves(action):
        if fc.data_path == path and 0 <= fc.array_index < n:
            fcs[fc.array_index] = fc
    return path, fcs


def bone_actions(arm):
    """Every clip that can animate the armature's bones: the imported M2
    sequences plus any action with bone channels. Global-sequence clips are
    left alone (they loop on their own clock)."""
    out = []
    for act in bpy.data.actions:
        if "m2_global_sequence" in act.keys():
            continue
        if "m2_seq_index" in act.keys() or any(
                fc.data_path.startswith("pose.bones[") for fc in _iter_fcurves(act)):
            out.append(act)
    return out


# ---------------------------------------------------------------------------
# state
def is_adjusting(arm):
    return arm is not None and STATE_KEY in arm.keys()


def _reset(pb):
    pb.location = (0.0, 0.0, 0.0)
    pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    pb.rotation_euler = (0.0, 0.0, 0.0)
    pb.scale = (1.0, 1.0, 1.0)


def start(arm, context):
    """Pause the armature's animation and show the rest pose for editing."""
    if is_adjusting(arm):
        return
    ad = arm.animation_data_create()
    slot = getattr(ad, "action_slot", None)
    state = {
        "action": ad.action.name if ad.action else "",
        "slot": slot.identifier if slot is not None else "",
        "use_nla": bool(ad.use_nla),
        "frame": int(context.scene.frame_current),
        "modes": {pb.name: pb.rotation_mode for pb in arm.pose.bones},
    }
    arm[STATE_KEY] = json.dumps(state)
    ad.action = None
    ad.use_nla = False
    for pb in arm.pose.bones:
        pb.rotation_mode = "XYZ"          # friendlier fields while editing
        _reset(pb)
    if context.view_layer.objects.active is not arm:
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        context.view_layer.objects.active = arm
    arm.select_set(True)
    if context.mode != "POSE":
        bpy.ops.object.mode_set(mode="POSE")


def deltas(arm):
    """{bone name: (location, rotation, scale)} for every bone moved off its
    rest pose."""
    out = {}
    for pb in arm.pose.bones:
        loc, rot, scl = pb.matrix_basis.decompose()
        if (loc.length > EPS or abs(abs(rot.w) - 1.0) > EPS
                or any(abs(s - 1.0) > EPS for s in scl)):
            out[pb.name] = (loc.copy(), rot.normalized(), scl.copy())
    return out


def finish(arm, context):
    """Back to the clip the armature had before ``start``."""
    raw = arm.get(STATE_KEY)
    state = json.loads(raw) if raw else {}
    for pb in arm.pose.bones:
        _reset(pb)
        pb.rotation_mode = state.get("modes", {}).get(pb.name, "QUATERNION")
    ad = arm.animation_data_create()
    act = bpy.data.actions.get(state.get("action", "")) if state.get("action") else None
    ad.action = act
    if act is not None and state.get("slot") and hasattr(act, "slots"):
        for s in act.slots:
            if s.identifier == state["slot"]:
                try:
                    ad.action_slot = s
                except Exception:  # noqa: BLE001 - Blender picks one itself
                    pass
                break
    ad.use_nla = state.get("use_nla", True)
    if STATE_KEY in arm.keys():
        del arm[STATE_KEY]
    if "frame" in state:
        context.scene.frame_set(int(state["frame"]))


# ---------------------------------------------------------------------------
# baking
def _rewrite(action, path, fcs, n, default, fn, collection):
    """Rewrite one channel (location / rotation / scale) of one bone:
    every keyed frame gets fn(value). A channel the clip does not key gets
    a single key holding fn(default), if that differs from the default."""
    existing = [fc for fc in fcs if fc is not None]
    if not existing:
        new = fn(list(default))
        if all(abs(a - b) < EPS for a, b in zip(new, default)):
            return 0
        frames = [float(action.frame_range[0])]
        values = [new]
        interp = "LINEAR"
    else:
        frames = sorted({round(kp.co.x, 4) for fc in existing for kp in fc.keyframe_points})
        if not frames:
            return 0
        interp = next((kp.interpolation for fc in existing for kp in fc.keyframe_points), "LINEAR")
        values = []
        prev = None
        for f in frames:
            v = [fcs[i].evaluate(f) if fcs[i] is not None else default[i] for i in range(n)]
            nv = fn(v)
            if n == 4 and prev is not None and sum(a * b for a, b in zip(nv, prev)) < 0:
                nv = [-c for c in nv]        # keep quaternion keys on one hemisphere
            values.append(nv)
            prev = nv
    for i in range(n):
        fc = fcs[i]
        if fc is None:
            fc = collection.new(data_path=path, index=i)
            fcs[i] = fc
        fc.keyframe_points.clear()
        fc.keyframe_points.add(len(frames))
        flat = []
        for f, v in zip(frames, values):
            flat.extend((f, v[i]))
        fc.keyframe_points.foreach_set("co", flat)
        for kp in fc.keyframe_points:
            kp.interpolation = interp
        fc.update()
    return len(frames)


def bake(arm, offsets, actions=None, log=print):
    """Bake {bone: (loc, rot, scale)} into every clip. Returns a summary."""
    actions = bone_actions(arm) if actions is None else actions
    touched, keys = 0, 0
    for act in actions:
        coll = None
        changed = False
        for bone, (d_loc, d_rot, d_scl) in offsets.items():
            rot_identity = abs(abs(d_rot.w) - 1.0) < EPS
            if coll is None:
                coll = _fcurve_collection(act)
            if not rot_identity:
                path, fcs = _channel(act, bone, "rotation_quaternion", 4)
                k = _rewrite(act, path, fcs, 4, (1.0, 0.0, 0.0, 0.0),
                             lambda v, q=d_rot: list((q @ Quaternion(v)).normalized()), coll)
                keys += k; changed |= bool(k)
            if d_loc.length > EPS or not rot_identity:
                path, fcs = _channel(act, bone, "location", 3)
                k = _rewrite(act, path, fcs, 3, (0.0, 0.0, 0.0),
                             lambda v, q=d_rot, t=d_loc: list(t + q @ Vector(v)), coll)
                keys += k; changed |= bool(k)
            if any(abs(s - 1.0) > EPS for s in d_scl):
                path, fcs = _channel(act, bone, "scale", 3)
                k = _rewrite(act, path, fcs, 3, (1.0, 1.0, 1.0),
                             lambda v, s=d_scl: [v[0] * s[0], v[1] * s[1], v[2] * s[2]], coll)
                keys += k; changed |= bool(k)
        touched += int(changed)
    # A bone that now carries keys must be evaluated by the client: the
    # "transformed" flag. Physics bones (0x400) keep their flags.
    flagged = []
    for bone in offsets:
        db = arm.data.bones.get(bone)
        if db is None:
            continue
        f = int(db.get("m2_bone_flags", 0x200))
        if not f & 0x400 and not f & 0x200:
            db["m2_bone_flags"] = f | 0x200
            flagged.append(bone)
    log("[bone adjust] %d bone(s) baked into %d clip(s), %d keys written%s"
        % (len(offsets), touched, keys,
           ("; 'transformed' flag set on %s" % ", ".join(flagged)) if flagged else ""))
    return {"bones": len(offsets), "actions": touched, "keys": keys, "flagged": flagged}


def attachment_empties(arm):
    """Attachment empties parented to bones of this armature."""
    out = []
    for o in bpy.data.objects:
        if o.type == "EMPTY" and o.parent is arm and o.parent_type == "BONE" and o.parent_bone \
                and ("m2_att_id" in o.keys() or "_attach_" in o.name):
            out.append(o)
    return sorted(out, key=lambda o: int(o.get("m2_att_id", 999)))
