"""Blender-side construction of animations and attachment points."""

import bpy
from mathutils import Matrix, Vector, Quaternion

from . import names

# Bone length used by the armature build. Attachment placement relies on it,
# so it lives here and is imported by builder.py.
BONE_LENGTH = 0.1


def _conv_pos(p, mirror_x):
    x, y, z = p
    return (-x, y, z) if mirror_x else (x, y, z)


def _conv_quat(q, mirror_x):
    """M2 (x, y, z, w) -> Blender (w, x, y, z), mirroring across X if asked."""
    x, y, z, w = q
    if mirror_x:
        return (w, x, -y, -z)
    return (w, x, y, z)


def bone_name(index):
    return "bone_%03d" % index


# ---------------------------------------------------------------------------
_CAMERA_TYPE_NAMES = {-1: "free", 0: "portrait", 1: "charinfo"}


def _track_first_value(track, default):
    """First keyframe value of a track (any sequence), or ``default``."""
    if track is None:
        return default
    if track.flat is not None and track.flat[1]:
        return track.flat[1][0]
    for tl in (track.timelines or []):
        if tl:
            return tl[0][1]
    return default


def _track_keys(track, fps):
    """[(frame, value)] for the first non-empty timeline of a track, else []."""
    if track is None:
        return []
    scale = fps / 1000.0
    if track.flat is not None:
        times, values = track.flat
        return [(t * scale, v) for t, v in zip(times, values)]
    for tl in (track.timelines or []):
        if tl:
            return [(t * scale, v) for t, v in tl]
    return []


def build_cameras(model, name, mirror_x, collection, fps, log):
    """Create a Blender camera per M2Camera, aimed at a target empty."""
    if not model.cameras:
        return []
    created = []
    for ci, cam in enumerate(model.cameras):
        label = _CAMERA_TYPE_NAMES.get(cam.type, "cam%d" % cam.type)
        cdata = bpy.data.cameras.new("%s_camera_%s" % (name, label))
        cdata.clip_start = max(1e-4, cam.near_clip)
        cdata.clip_end = max(cdata.clip_start + 1.0, cam.far_clip)
        # WoW fov is the VERTICAL field of view in radians.
        fov = cam.fov_static if cam.fov is None else _track_first_value(cam.fov, cam.fov_static)
        if fov and fov > 0.0:
            cdata.sensor_fit = "VERTICAL"
            cdata.lens_unit = "FOV"
            cdata.angle_y = float(fov)
        cdata["m2_camera_type"] = cam.type

        cobj = bpy.data.objects.new(cdata.name, cdata)
        collection.objects.link(cobj)

        tgt = bpy.data.objects.new("%s_camtarget_%s" % (name, label), None)
        tgt.empty_display_type = "PLAIN_AXES"
        tgt.empty_display_size = 0.1
        collection.objects.link(tgt)

        cobj.location = _conv_pos(cam.position_base, mirror_x)
        tgt.location = _conv_pos(cam.target_base, mirror_x)

        con = cobj.constraints.new("DAMPED_TRACK")
        con.target = tgt
        con.track_axis = "TRACK_NEGATIVE_Z"

        # Animate position / target / roll if the tracks carry keyframes.
        _keyframe_camera(cobj, tgt, cam, mirror_x, fps)

        created.append(cobj)

    log("cameras: %d (%s)" % (len(created),
                              ", ".join(_CAMERA_TYPE_NAMES.get(c.type, str(c.type))
                                        for c in model.cameras)))
    return created


def _keyframe_camera(cobj, tgt, cam, mirror_x, fps):
    # The position / target tracks are RELATIVE to the base, so the absolute
    pb, tb = cam.position_base, cam.target_base
    pos_keys = _track_keys(cam.position, fps)
    for f, v in pos_keys:
        cobj.location = _conv_pos((pb[0]+v[0], pb[1]+v[1], pb[2]+v[2]), mirror_x)
        cobj.keyframe_insert("location", frame=int(round(f)))
    tgt_keys = _track_keys(cam.target, fps)
    for f, v in tgt_keys:
        tgt.location = _conv_pos((tb[0]+v[0], tb[1]+v[1], tb[2]+v[2]), mirror_x)
        tgt.keyframe_insert("location", frame=int(round(f)))
    roll_keys = _track_keys(cam.roll, fps)
    if roll_keys:
        cobj.rotation_mode = "YXZ"
        for f, v in roll_keys:
            cobj.rotation_euler = (0.0, 0.0, float(v))
            cobj.keyframe_insert("rotation_euler", index=2, frame=int(round(f)))


# ---------------------------------------------------------------------------
STANDARD_EVENTS = [
    ("$SHL", 2,  [(89, 300), (90, 200)]),   # sheathe/draw, left hand
    ("$SHR", 1,  [(89, 300), (90, 200)]),   # sheathe/draw, right hand
    ("$FL0", 47, [(4, 233)]),               # walk, left foot
    ("$FR0", 48, [(4, 33)]),                # walk, right foot
    ("$BL0", 47, [(13, 133)]),              # walk backwards, left
    ("$BR0", 48, [(13, 333)]),              # walk backwards, right
    ("$RL0", 47, [(5, 100)]),               # run, left
    ("$RR0", 48, [(5, 367)]),               # run, right
    ("$SL0", 47, [(3, 34)]),                # stop, left
    ("$SR0", 48, [(3, 34)]),                # stop, right
    ("$RL1", 47, [(143, 134)]),             # sprint, left
    ("$RR1", 48, [(143, 400)]),             # sprint, right
    ("$RL2", 47, [(187, 100)]),             # jump-land run, left
    ("$RR2", 48, [(187, 334)]),             # jump-land run, right
    ("$CSL", 21, [(124, 34)]),              # cast, left hand
    ("$CSR", 22, [(53, 200), (107, 467)]),  # cast, right hand
    ("$BTH", 17, [(0, 934)]),               # breath
]


def build_events(model, arm_obj, name, mirror_x, collection, log):
    """Create a visible marker per M2 event, parented to its bone."""
    import json
    if not getattr(model, "events", None):
        return []
    created = []
    arm_bones = arm_obj.data.bones if arm_obj is not None else None
    # Positional index -> "<id>-<var>" key so we can rewrite event times as a
    # dict the export path can look up per output sequence.
    seq_keys = ["%d-%d" % (int(s.id), int(getattr(s, "variation_index", 0)))
                for s in (model.sequences or [])]
    for e in model.events:
        label = (e.identifier or "EVT").strip().lstrip("$") or "EVT"
        obj = bpy.data.objects.new("%s_event_%s" % (name, label), None)
        obj.empty_display_type = "SPHERE"
        obj.empty_display_size = 0.06
        obj.show_name = True
        obj["m2_event_id"] = e.identifier            # e.g. "$SHR"
        obj["m2_event_data"] = int(e.data)
        obj["m2_event_bone"] = int(e.bone)
        # Store timings keyed by sequence identity, not by index. Empty lists
        # are dropped from the map so it stays small.
        times_map = {}
        for si, ts in enumerate(e.timestamps or []):
            if not ts:
                continue
            if si < len(seq_keys):
                times_map[seq_keys[si]] = [int(t) for t in ts]
        obj["m2_event_times"] = json.dumps(times_map)
        obj["m2_event_fires_in"] = len(times_map)    # sequences using it
        collection.objects.link(obj)

        pos = _conv_pos(e.position, mirror_x)
        bname = bone_name(e.bone)
        if arm_bones is not None and bname in arm_bones:
            db = arm_bones[bname]
            obj.parent = arm_obj
            obj.parent_type = "BONE"
            obj.parent_bone = bname
            # Position is absolute model space, same convention as attachments.
            local = db.matrix_local.inverted() @ Vector(pos)
            obj.matrix_parent_inverse = Matrix.Translation(
                (local.x, local.y - db.length, local.z))
        else:
            obj.location = pos
        created.append(obj)
    log("created %d event marker(s)" % len(created))
    return created


def add_missing_events(model, arm_obj, name, mirror_x, collection, log):
    """Create markers for standard events the model lacks; only fills gaps."""
    import json
    # Empty-timestamp events fire on nothing; treat as absent and rebuild their timings.
    have = set()
    broken = set()
    for e in (getattr(model, "events", None) or []):
        ident = (e.identifier or "").strip()
        if any(e.timestamps or []):
            have.add(ident)
        else:
            broken.add(ident)
    att_by_id = {a.id: a for a in model.attachments}
    seq_by_anim = {}
    for i, s in enumerate(model.sequences):
        seq_by_anim.setdefault(s.id, []).append(i)

    arm_bones = arm_obj.data.bones if arm_obj is not None else None
    created = []
    skipped_att = skipped_anim = 0
    for ident, att_id, fires in STANDARD_EVENTS:
        if ident in have:
            continue
        att = att_by_id.get(att_id)
        if att is None:
            skipped_att += 1
            continue
        # Store timings by "<id>-<var>" (matches build_events) so a subset
        # export still lands the event on the right animation.
        times_map = {}
        for aid, t in fires:
            for si in seq_by_anim.get(aid, []):
                s = model.sequences[si]
                key = "%d-%d" % (int(s.id), int(getattr(s, "variation_index", 0)))
                times_map[key] = [int(t)]
        if not times_map:
            skipped_anim += 1
            continue

        # Drop existing marker with no timings so the repaired one replaces it.
        if ident in broken:
            for old in [o for o in collection.objects
                        if o.type == "EMPTY" and o.get("m2_event_id") == ident]:
                bpy.data.objects.remove(old, do_unlink=True)
        label = ident.strip().lstrip("$") or "EVT"
        obj = bpy.data.objects.new("%s_event_%s" % (name, label), None)
        obj.empty_display_type = "SPHERE"
        obj.empty_display_size = 0.06
        obj.show_name = True
        obj["m2_event_id"] = ident
        obj["m2_event_data"] = 0
        obj["m2_event_bone"] = int(att.bone)
        obj["m2_event_times"] = json.dumps(times_map)
        obj["m2_event_fires_in"] = len(times_map)
        collection.objects.link(obj)

        pos = _conv_pos(att.position, mirror_x)
        bname = bone_name(att.bone)
        if arm_bones is not None and bname in arm_bones:
            db = arm_bones[bname]
            obj.parent = arm_obj
            obj.parent_type = "BONE"
            obj.parent_bone = bname
            local = db.matrix_local.inverted() @ Vector(pos)
            obj.matrix_parent_inverse = Matrix.Translation(
                (local.x, local.y - db.length, local.z))
        else:
            obj.location = pos
        created.append(ident)

    if created:
        fixed = [c for c in created if c in broken]
        log("added %d event(s): %s" % (len(created), " ".join(created)))
        if fixed:
            log("  (%d had no timings and were rebuilt: %s)"
                % (len(fixed), " ".join(fixed)))
    if skipped_att or skipped_anim:
        log("  (%d skipped: no attachment, %d skipped: animation not in model)"
            % (skipped_att, skipped_anim))
    return created


# ---------------------------------------------------------------------------
def build_attachments(model, arm_obj, name, mirror_x, collection, log):
    """Create an empty per attachment, parented to its bone at the right spot."""
    if not model.attachments:
        return []
    created = []
    arm_bones = arm_obj.data.bones if arm_obj is not None else None
    for att in model.attachments:
        # Known ids -> readable name, unknown -> the bare number (the "attach_"
        # prefix already marks it as an attachment, so avoid "attach_Attach_NN").
        label = names.ATTACHMENT_NAMES.get(att.id, att.id)
        empty = bpy.data.objects.new("%s_attach_%s" % (name, label), None)
        empty.empty_display_type = "ARROWS"
        empty.empty_display_size = 0.15
        empty.show_name = True
        # Identity for export: the display name is lossy (unknown ids fall back
        empty["m2_att_id"] = att.id
        empty["m2_att_bone"] = att.bone
        collection.objects.link(empty)

        pos = _conv_pos(att.position, mirror_x)
        bname = bone_name(att.bone)
        # The placement is baked into matrix_parent_inverse, not the empty's own
        if arm_bones is not None and bname in arm_bones:
            empty.parent = arm_obj
            empty.parent_type = "BONE"
            empty.parent_bone = bname
            db_ = arm_bones[bname]
            local = db_.matrix_local.inverted() @ Vector(pos)
            empty.matrix_parent_inverse = Matrix.Translation(
                (local.x, local.y - db_.length, local.z))
        elif arm_obj is not None:
            empty.parent = arm_obj
            empty.matrix_parent_inverse = Matrix.Translation(pos)
        else:
            empty.location = pos
        created.append(empty)

    log("attachments: %d empties" % len(created))
    return created


# ---------------------------------------------------------------------------
def _keys_for_sequence(track, seq_index, seq, fps):
    """Keyframes for ``track`` within one normal (non-global) sequence."""
    if track is None or track.is_empty() or track.global_sequence >= 0:
        return None
    scale = fps / 1000.0
    if track.flat is not None:
        # Legacy single timeline: slice to this sequence's [start, end] window.
        times, values = track.flat
        start, end = seq.start_timestamp, seq.end_timestamp
        out = [((t - start) * scale, v)
               for t, v in zip(times, values) if start <= t <= end]
        return out or None
    # Modern: one timeline per sequence index.
    if seq_index >= len(track.timelines):
        return None
    return [(t * scale, v) for t, v in track.timelines[seq_index]] or None


def _keys_global(track, fps):
    """Keyframes for a global-sequence track (loops independently of anims)."""
    if track is None or track.global_sequence < 0:
        return None
    scale = fps / 1000.0
    if track.flat is not None:
        times, values = track.flat
        return [(t * scale, v) for t, v in zip(times, values)] or None
    if not track.timelines:
        return None
    return [(t * scale, v) for t, v in track.timelines[0]] or None


def rest_rotation(arm_obj, bone_index):
    """A built bone's rest orientation, or None when it's axis-aligned."""
    if arm_obj is None:
        return None
    db = arm_obj.data.bones.get(bone_name(bone_index))
    if db is None:
        return None
    q = db.matrix_local.to_quaternion()
    q.normalize()
    if abs(q.w - 1.0) < 1e-6 and abs(q.x) < 1e-6 and abs(q.y) < 1e-6 \
            and abs(q.z) < 1e-6:
        return None
    return q


def _key_bone(fcurves, bone_index, bone, mirror_x, keyfn, rest=None):
    """Build location/rotation/scale F-Curves for one bone via ``keyfn(track)``."""
    path = 'pose.bones["%s"].' % bone_name(bone_index)
    if rest is None:
        conv_pos = lambda v: _conv_pos(v, mirror_x)                  # noqa: E731
        conv_rot = lambda v: _conv_quat(v, mirror_x)                 # noqa: E731
    else:
        r_inv = rest.inverted()

        def conv_pos(v):
            return tuple(r_inv @ Vector(_conv_pos(v, mirror_x)))

        def conv_rot(v):
            w, x, y, z = _conv_quat(v, mirror_x)
            m = r_inv @ Quaternion((w, x, y, z)) @ rest
            return (m.w, m.x, m.y, m.z)

    specs = (
        (bone.translation, "location", 3, conv_pos),
        (bone.rotation, "rotation_quaternion", 4, conv_rot),
        (bone.scale, "scale", 3, lambda v: tuple(v)),
    )
    last = None
    for track, prop, n_comp, convert in specs:
        keys = keyfn(track)
        if not keys:
            continue
        _add_channel(fcurves, path + prop, n_comp, keys, convert,
                     track.interpolation_type == 0)
        f = keys[-1][0]
        last = f if last is None else max(last, f)
    return last


def _new_action_channelbag(action):
    """Return (fcurves_collection, slot) for ``action`` across Blender APIs."""
    if hasattr(action, "layers") and hasattr(action, "slots"):
        try:
            try:
                slot = action.slots.new(id_type="OBJECT", name="Object")
            except TypeError:
                slot = action.slots.new("OBJECT", "Object")
            layer = action.layers.new("Layer")
            strip = layer.strips.new(type="KEYFRAME")
            cbag = strip.channelbag(slot, ensure=True)
            return cbag.fcurves, slot
        except Exception:  # noqa: BLE001 - fall through to legacy
            pass
    # Attribute-safe: on 4.4+/5.x `fcurves` is absent entirely, so a failure
    # above must not turn into an AttributeError here.
    legacy = getattr(action, "fcurves", None)
    if legacy is None:
        raise RuntimeError(
            "This Blender build has neither layered actions nor action.fcurves; "
            "cannot create animation channels.")
    return legacy, None


def _add_channel(fcurves, data_path, n_comp, keys, convert, constant):
    """Create one F-Curve per component and bulk-load the keyframes."""
    n = len(keys)
    frames = [k[0] for k in keys]
    converted = [convert(v) for _, v in keys]
    interp = 0 if constant else 1   # 0 = CONSTANT, 1 = LINEAR
    for comp in range(n_comp):
        fc = fcurves.new(data_path, index=comp)
        fc.keyframe_points.add(n)
        flat = [0.0] * (2 * n)
        for i in range(n):
            flat[2 * i] = frames[i]
            flat[2 * i + 1] = converted[i][comp]
        fc.keyframe_points.foreach_set("co", flat)
        fc.keyframe_points.foreach_set("interpolation", [interp] * n)
        fc.update()


def build_animations(model, arm_obj, mirror_x, fps, log, max_animations=0, name=None):
    """Create one Blender Action per M2 sequence and key the pose bones."""
    if arm_obj is None:
        log("animations: skipped (no armature)")
        return 0
    bones = model.bones
    if not any(b.is_animated() for b in bones):
        log("animations: none present")
        return 0

    sequences = model.sequences
    if not sequences:
        # No sequence table parsed: infer count from the longest track.
        n = 0
        for b in bones:
            if b.translation:
                n = max(n, len(b.translation.timelines))
            if b.rotation:
                n = max(n, len(b.rotation.timelines))
            if b.scale:
                n = max(n, len(b.scale.timelines))
        from .model import M2Sequence
        sequences = []
        for i in range(n):
            s = M2Sequence()
            s.id = i
            sequences.append(s)

    for pb in arm_obj.pose.bones:
        pb.rotation_mode = "QUATERNION"
    # Rest orientation per bone. None everywhere for the usual +Y armature, so
    # the conversion below costs nothing unless the bones were built tilted.
    rests = {bi: rest_rotation(arm_obj, bi) for bi in range(len(bones))}
    ad = arm_obj.animation_data_create()
    model_name = name or model.name or "M2"
    made = 0
    first = None
    first_slot = None
    first_max = 0.0
    total = len(sequences)
    if max_animations:
        log("animations: building up to %d of %d sequences" % (max_animations, total))
    else:
        log("animations: building all %d sequences (this can take a while)" % total)

    for si, seq in enumerate(sequences):
        if max_animations and made >= max_animations:
            log("animations: reached cap of %d (raise 'Max Animations' for more)"
                % max_animations)
            break
        label = names.animation_name(seq.id)
        if seq.variation_index:
            label = "%s_%d" % (label, seq.variation_index)
        action = bpy.data.actions.new("%s_%s" % (model_name, label))
        fcurves, slot = _new_action_channelbag(action)

        action_max = None
        for bi, b in enumerate(bones):
            if not b.is_animated():
                continue
            last = _key_bone(fcurves, bi, b, mirror_x,
                             lambda tr: _keys_for_sequence(tr, si, seq, fps),
                             rest=rests.get(bi))
            if last is not None:
                action_max = last if action_max is None else max(action_max, last)

        if action_max is None:
            bpy.data.actions.remove(action)
            continue

        action.use_fake_user = True   # keep every clip even when not active
        # Everything export needs to rebuild this sequence lives on the clip
        # itself, so a from-scratch export doesn't have to consult m2_meta.
        action["m2_seq_index"] = si   # map clip back to its sequence on export
        action["m2_seq_id"] = seq.id
        action["m2_seq_var"] = seq.variation_index
        action["m2_seq_flags"] = seq.flags
        action["m2_seq_duration"] = seq.duration
        action["m2_seq_blend_time"] = int(getattr(seq, "blend_time_in", 150))
        action["m2_seq_blend_out"] = int(getattr(seq, "blend_time_out", 0))
        action["m2_seq_movespeed"] = float(getattr(seq, "movespeed", 0.0))
        made += 1
        if first is None:
            first, first_slot, first_max = action, slot, action_max
        if made % 25 == 0:
            log("animations: %d clips built (seq %d/%d)" % (made, si + 1, total))

    made += _build_global_sequences(bones, fcurves_factory=_new_action_channelbag,
                                    model_name=model_name, mirror_x=mirror_x,
                                    fps=fps, log=log,
                                    global_loops=model.global_loops, rests=rests)

    # Make the first clip active so the user immediately sees an animation.
    if first is not None:
        ad.action = first
        if first_slot is not None:
            try:
                ad.action_slot = first_slot
            except Exception:  # noqa: BLE001
                pass
        scene = bpy.context.scene
        scene.frame_start = 0
        scene.frame_end = max(1, int(round(first_max)))

    # Play the scene at the SAME fps the keyframes were placed at. Keyframe
    if made:
        bpy.context.scene.render.fps = int(round(fps))
        bpy.context.scene.render.fps_base = 1.0

    log("animations: %d clips (scene fps set to %d)" % (made, int(round(fps))))
    return made


def _build_global_sequences(bones, fcurves_factory, model_name, mirror_x, fps, log,
                            global_loops=(), rests=None):
    """Build one action per global sequence used by a bone track."""
    used = sorted({tr.global_sequence
                   for b in bones
                   for tr in (b.translation, b.rotation, b.scale)
                   if tr is not None and tr.global_sequence >= 0})
    made = 0
    for gs in used:
        action = bpy.data.actions.new("%s_global_%d" % (model_name, gs))
        fcurves, _slot = fcurves_factory(action)
        keyed = False
        for bi, b in enumerate(bones):
            last = _key_bone(
                fcurves, bi, b, mirror_x,
                lambda tr, _gs=gs: _keys_global(tr, fps) if tr is not None
                and tr.global_sequence == _gs else None,
                rest=(rests or {}).get(bi))
            if last is not None:
                keyed = True
        if keyed:
            action.use_fake_user = True
            action["m2_global_sequence"] = gs
            # Loop length travels with the clip so export can rebuild the
            # global-loop table without the m2_meta blob.
            if gs < len(global_loops):
                action["m2_global_duration"] = global_loops[gs]
            made += 1
        else:
            bpy.data.actions.remove(action)
    if made:
        log("animations: %d global-sequence clips" % made)
    return made
