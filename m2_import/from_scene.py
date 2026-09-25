"""Build an :class:`M2Model` from a RAW Blender scene (no prior import / no"""

import re
import json
import zlib

import bpy
from mathutils import Vector, Matrix, Quaternion

from .model import (
    M2Model, M2Vertex, M2Bone, M2Sequence, M2AnimTrack, M2Attachment,
    M2SkinProfile, M2SubMesh, M2Batch, M2Texture, M2Material, M2Camera,
    M2Event,
)
from .scene_reader import geoset_id_from_name
from . import names
from . import anim_data


def _crc(name):
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


def _topo_bones(arm_data):
    """Armature bones ordered parents-before-children (M2 needs parent < child)."""
    import re as _re
    _num_re = _re.compile(r"bone_(\d+)")

    def _idx(b):
        m = _num_re.match(b.name)
        return (int(m.group(1)) if m else 1 << 30, b.name)

    bones = sorted(arm_data.bones, key=_idx)
    base = [b for b in bones if b.get("m2_scale_bone") is None]
    extra = [b for b in bones if b.get("m2_scale_bone") is not None]
    ordered = []
    seen = set()

    def visit(b):
        if b.name in seen:
            return
        if b.parent is not None:
            visit(b.parent)
        seen.add(b.name)
        ordered.append(b)

    for b in base:
        visit(b)
    for b in extra:      # parents are already emitted, so these just append
        visit(b)
    return ordered


def _ms(frame, fps):
    return int(round(frame / fps * 1000.0))


# A single-keyframe (static pose) clip has zero frame extent, so it would export
STATIC_POSE_MS = 1000


def _fix_degenerate_duration(seq):
    """A <=1 ms sequence is unplayable; give a static pose a real duration."""
    if seq.duration <= 1:
        seq.duration = STATIC_POSE_MS
        seq.end_timestamp = seq.duration
    return seq


# ---------------------------------------------------------------------------
def _build_bones(model, arm, mirror_x):
    ordered = _topo_bones(arm.data)
    idx = {b.name: i for i, b in enumerate(ordered)}
    key_lookup = {}
    for i, db in enumerate(ordered):
        b = M2Bone()
        h = db.head_local                      # rest head in armature space
        b.pivot = (-h.x, h.y, h.z) if mirror_x else (h.x, h.y, h.z)
        b.parent = idx[db.parent.name] if db.parent else -1
        # Prefer the real M2 fields stamped at import; a hand-built bone gets
        # sensible defaults instead.
        b.flags = int(db.get("m2_bone_flags", 0x200))
        crc = db.get("m2_name_crc")
        b.name_crc = int(crc) if crc is not None else _crc(db.name)
        b.key_bone_id = int(db.get("m2_key_bone_id", -1))
        b.submesh_id = int(db.get("m2_submesh_id", 0))
        if b.key_bone_id >= 0:
            key_lookup.setdefault(b.key_bone_id, i)
        b.translation = M2AnimTrack()
        b.rotation = M2AnimTrack()
        b.scale = M2AnimTrack()
        model.bones.append(b)

    # keyBoneLookup: the client finds named bones (root, hands, ...) through it,
    # so rebuild it rather than shipping an empty table.
    if key_lookup:
        model.key_bone_lookup = [key_lookup.get(k, 0xFFFF)
                                 for k in range(max(key_lookup) + 1)]
    return ordered, idx


def _fcurve_map(action):
    """data_path+index -> fcurve, across legacy and slotted (4.4+/5.0) actions."""
    fcurves = []
    if hasattr(action, "layers") and len(getattr(action, "layers", [])):
        for layer in action.layers:
            for strip in layer.strips:
                cbs = getattr(strip, "channelbags", None) or []
                for cb in cbs:
                    fcurves.extend(cb.fcurves)
    if not fcurves:
        fcurves = list(getattr(action, "fcurves", []))
    out = {}
    for fc in fcurves:
        out[(fc.data_path, fc.array_index)] = fc
    return out


def _keyframe_frames(fmap, base, n):
    frames = set()
    for i in range(n):
        fc = fmap.get((base, i))
        if fc:
            for kp in fc.keyframe_points:
                frames.add(kp.co[0])
    return sorted(frames)


def _find_meta(objects):
    """The ``m2_meta`` JSON from an import, if this scene came from one."""
    for o in list(objects) + list(bpy.data.objects):
        if "m2_meta" in o.keys():
            try:
                return json.loads(o["m2_meta"])
            except Exception:  # noqa: BLE001 - corrupt/foreign property
                return None
    return None


def _bone_channels(action, idx):
    """``{bone_index: {prop: {component: fcurve}}}`` for one action."""
    out = {}
    for (dp, ai), fc in _fcurve_map(action).items():
        if not dp.startswith('pose.bones["'):
            continue
        try:
            bname = dp.split('"')[1]
            prop = dp.rsplit(".", 1)[1]
        except IndexError:
            continue
        bi = idx.get(bname)
        if bi is None or prop not in ("location", "rotation_quaternion", "scale"):
            continue
        out.setdefault(bi, {}).setdefault(prop, {})[ai] = fc
    return out


def _sample(chans, prop, n, defaults, fps, f0=0.0):
    """``[(time_ms, value_tuple)]`` sampled at this property's keyframe times."""
    comps = chans.get(prop)
    if not comps:
        return None
    frames = sorted({kp.co[0] for fc in comps.values() for kp in fc.keyframe_points})
    if not frames:
        return None
    out = []
    for f in frames:
        vals = []
        for k in range(n):
            fc = comps.get(k)
            vals.append(fc.evaluate(f) if fc is not None else defaults[k])
        out.append((_ms(f - f0, fps), tuple(vals)))
    return out


def _rest_rotation(db):
    """A bone's rest orientation as a normalised quaternion, or None if none."""
    q = db.matrix_local.to_quaternion()
    q.normalize()
    if abs(q.w - 1.0) < 1e-6 and abs(q.x) < 1e-6 and abs(q.y) < 1e-6 \
            and abs(q.z) < 1e-6:
        return None
    return q


def _bone_specs(db, mirror_x):
    """Per-bone (property, components, defaults, track, converter) tuples."""
    R = _rest_rotation(db)

    if R is None:
        def loc(v):
            return (-v[0], v[1], v[2]) if mirror_x else tuple(v)

        def rot(v):
            w, x, y, z = v
            return (x, -y, -z, w) if mirror_x else (x, y, z, w)
    else:
        Rinv = R.inverted()

        def loc(v):
            m = R @ Vector(v)
            return (-m.x, m.y, m.z) if mirror_x else (m.x, m.y, m.z)

        def rot(v):
            m = R @ Quaternion((v[0], v[1], v[2], v[3])) @ Rinv
            return (m.x, -m.y, -m.z, m.w) if mirror_x else (m.x, m.y, m.z, m.w)

    return (
        ("location", 3, (0.0, 0.0, 0.0), "translation", loc),
        ("rotation_quaternion", 4, (1.0, 0.0, 0.0, 0.0), "rotation", rot),
        ("scale", 3, (1.0, 1.0, 1.0), "scale", lambda v: tuple(v)),
    )


# M2AnimTrack.interpolation_type
INTERP_NONE, INTERP_LINEAR = 0, 1


def _interp_of(chans, prop):
    """M2 interpolation for a property, from its Blender keyframes."""
    comps = chans.get(prop)
    if not comps:
        return INTERP_LINEAR
    for fc in comps.values():
        for kp in fc.keyframe_points:
            return INTERP_NONE if kp.interpolation == "CONSTANT" else INTERP_LINEAR
    return INTERP_LINEAR


def _interp_fcurves(fmap, base, n):
    """Interpolation for a property addressed through the flat fcurve map."""
    for i in range(n):
        fc = fmap.get((base, i))
        if fc:
            for kp in fc.keyframe_points:
                return INTERP_NONE if kp.interpolation == "CONSTANT" else INTERP_LINEAR
    return INTERP_LINEAR


def _store(track, si, nseq, keys, conv):
    if not track.timelines:
        track.timelines = [[] for _ in range(nseq)]
    track.timelines[si] = [(t, conv(v)) for t, v in keys]


def _apply_blend(s, action, m=None):
    """Blend times / movespeed for a sequence: clip stamp, else meta, else 150 ms."""
    m = m or {}
    if action is not None and "m2_seq_blend_time" in action.keys():
        s.blend_time_in = int(action["m2_seq_blend_time"])
        s.blend_time_out = int(action.get("m2_seq_blend_out", 0))
    elif "blend_in" in m:
        s.blend_time_in = int(m["blend_in"])
        s.blend_time_out = int(m.get("blend_out", 0))
    if action is not None and "m2_seq_movespeed" in action.keys():
        s.movespeed = float(action["m2_seq_movespeed"])
    elif "movespeed" in m:
        s.movespeed = float(m["movespeed"])
    b = None
    if action is not None and "m2_seq_bounds" in action.keys():
        b = list(action["m2_seq_bounds"])
    elif m.get("bounds"):
        b = [*m["bounds"][0], *m["bounds"][1], m["bounds"][2]]
    if b and len(b) == 7:
        s.bounds = ((b[0], b[1], b[2]), (b[3], b[4], b[5]), float(b[6]))


def _build_animation_multi(model, arm, ordered, idx, fps, mirror_x, meta):
    """Rebuild every imported sequence, not just the active action."""
    meta_seqs = (meta or {}).get("sequences") or []
    tagged = {}
    glob = {}
    for action in bpy.data.actions:
        if "m2_seq_index" in action.keys():
            tagged.setdefault(int(action["m2_seq_index"]), action)
        elif "m2_global_sequence" in action.keys():
            glob.setdefault(int(action["m2_global_sequence"]), action)
    if not tagged:
        return False

    # Keep the original order, but only emit sequences we actually have a clip
    kept = sorted(tagged)
    nseq = len(kept)
    model._kept_sequences = kept
    sequences = []
    seq_f0 = {}          # sequence index -> first Blender frame of its clip
    for si in kept:
        action = tagged[si]
        s = M2Sequence()
        # Duration must cover the clip's ACTUAL keyframes, or a clip that was
        f0, f1 = action.frame_range
        seq_f0[si] = f0
        extent = max(1, _ms(f1 - f0, fps))
        orig_dur = int(action.get("m2_seq_duration", 0))
        # Fall back to the metadata duration whichever way the id was resolved:
        if not orig_dur and si < len(meta_seqs):
            orig_dur = int(meta_seqs[si].get("dur", 0))
        # id / variation / flags still come from the clip's stamped fields.
        if "m2_seq_id" in action.keys():
            s.id = int(action["m2_seq_id"])
            s.variation_index = int(action.get("m2_seq_var", 0))
            # Same trap as the duration above: a retargeted clip keeps the stamped
            if "m2_seq_flags" in action.keys():
                s.flags = int(action["m2_seq_flags"])
            elif si < len(meta_seqs) and "flags" in meta_seqs[si]:
                s.flags = int(meta_seqs[si]["flags"])
            else:
                s.flags = 0x20
        elif si < len(meta_seqs):
            m = meta_seqs[si]
            s.id = m["id"]; s.variation_index = m["var"]; s.flags = m["flags"]
            orig_dur = orig_dur or int(m.get("dur", 0))
        else:
            s.id = si
            s.flags = 0x20
        s.duration = max(extent, orig_dur)
        s.start_timestamp = 0
        s.end_timestamp = s.duration
        _apply_blend(s, action, meta_seqs[si] if si < len(meta_seqs) else None)
        _fix_degenerate_duration(s)
        sequences.append(s)
    model.sequences = sequences

    # Alias sequences (flag 0x40) play another sequence's keys through
    # alias_next. Re-point each one at the output index of its stamped
    # "<id>-<var>" target; everything else points at itself, as retail does.
    key_to_idx = {}
    for i, s in enumerate(sequences):
        key_to_idx.setdefault("%d-%d" % (int(s.id), int(s.variation_index)), i)
    kept_pos = {si: i for i, si in enumerate(kept)}
    for i, si in enumerate(kept):
        s = sequences[i]
        action = tagged[si]
        target = i
        if "m2_seq_alias" in action.keys():
            target = key_to_idx.get(str(action["m2_seq_alias"]), -1)
        elif si < len(meta_seqs) and "alias_next" in meta_seqs[si]:
            target = kept_pos.get(int(meta_seqs[si]["alias_next"]), -1)
        if target < 0 or target == i:
            # Nothing to borrow (target not exported): play as a plain clip.
            s.flags &= ~0x40
            target = i
        s.alias_next = target

    # seq_lookup maps an animation id to its first sequence; reuse the original
    # when the sequence set came through whole, else rebuild it.
    orig_lookup = (meta or {}).get("seq_lookup") or []
    if orig_lookup and nseq == len(meta_seqs):
        model.seq_lookup = list(orig_lookup)
    else:
        lookup = {}
        for i, s in enumerate(sequences):
            lookup.setdefault(s.id, i)
        model.seq_lookup = [lookup.get(i, 0xFFFF)
                            for i in range(max(lookup) + 1 if lookup else 1)]

    # Global-loop durations: from the clips themselves where stamped, else meta.
    loops = list((meta or {}).get("global_loops") or [])
    if glob:
        need = max(glob) + 1
        while len(loops) < need:
            loops.append(0)
        for gs, action in glob.items():
            if "m2_global_duration" in action.keys():
                loops[gs] = int(action["m2_global_duration"])
            elif not loops[gs]:
                loops[gs] = max(1, _ms(action.frame_range[1], fps))
    model.global_loops = loops

    for si_new, si_old in enumerate(kept):
        chans_by_bone = _bone_channels(tagged[si_old], idx)
        f0 = seq_f0.get(si_old, 0.0)
        for bi, chans in chans_by_bone.items():
            bone = model.bones[bi]
            for prop, n, defaults, attr, conv in _bone_specs(ordered[bi], mirror_x):
                keys = _sample(chans, prop, n, defaults, fps, f0=f0)
                if keys:
                    tr = getattr(bone, attr)
                    _store(tr, si_new, nseq, keys, conv)
                    tr.interpolation_type = _interp_of(chans, prop)

    # Global sequences loop independently: a single timeline, flagged with the
    # loop index rather than slotted per sequence.
    for gs, action in glob.items():
        for bi, chans in _bone_channels(action, idx).items():
            bone = model.bones[bi]
            for prop, n, defaults, attr, conv in _bone_specs(ordered[bi], mirror_x):
                keys = _sample(chans, prop, n, defaults, fps)
                if not keys:
                    continue
                tr = getattr(bone, attr)
                tr.global_sequence = gs
                tr.timelines = [[(t, conv(v)) for t, v in keys]]
                tr.interpolation_type = _interp_of(chans, prop)

    # Pad every track to nseq so the writer emits a well-formed array.
    for b in model.bones:
        for tr in (b.translation, b.rotation, b.scale):
            if tr is None or tr.global_sequence >= 0:
                continue
            if not tr.timelines:
                tr.timelines = [[] for _ in range(nseq)]
            while len(tr.timelines) < nseq:
                tr.timelines.append([])
    return True


# Animation name (lowercased) -> AnimationData id, longest names checked first
# so "StandWound" wins over "Stand".
_ANIM_ID_BY_NAME = {name.lower(): aid for aid, name in names.ANIMATION_NAMES.items()}


def _anim_id_from_action_name(action_name):
    """(anim_id, variation) parsed from an action name, or None."""
    n = action_name.lower()
    if n.endswith("_remap"):                     # ARP retarget output suffix
        n = n[:-len("_remap")]
    tokens = [t for t in n.split("_") if t]
    if not tokens:
        return None
    var = 0
    if tokens[-1].isdigit():
        var = int(tokens[-1])
        tokens = tokens[:-1]
    if not tokens:
        return None
    aid = _ANIM_ID_BY_NAME.get(tokens[-1])
    return (aid, var) if aid is not None else None


def _build_animation_named(model, arm, ordered, idx, fps, mirror_x):
    """Build sequences from actions named after WoW animations (no import tags)."""
    named = []                                   # (anim_id, var, action)
    for action in bpy.data.actions:
        if "m2_seq_index" in action.keys() or "m2_global_sequence" in action.keys():
            continue
        res = _anim_id_from_action_name(action.name)
        if res is None:
            continue
        if not _bone_channels(action, idx):      # doesn't animate our bones
            continue
        named.append((res[0], res[1], action))
    if not named:
        return False

    named.sort(key=lambda t: (t[0], t[1], t[2].name))
    nseq = len(named)
    sequences = []
    for aid, var, action in named:
        s = M2Sequence()
        s.id = aid
        s.variation_index = var
        s.flags = 0x20                           # resident/looping default
        f0, f1 = action.frame_range
        # Retargeted clips carry no stamped duration, so a static pose (one
        # keyframe) would export as an unplayable 1 ms sequence.
        s.duration = max(1, _ms(f1 - f0, fps),
                         int(action.get("m2_seq_duration", 0)))
        s.start_timestamp = 0
        s.end_timestamp = s.duration
        s.alias_next = len(sequences)            # points at itself, like retail
        _apply_blend(s, action)
        _fix_degenerate_duration(s)
        sequences.append(s)
    model.sequences = sequences

    lookup = {}
    for i, s in enumerate(sequences):
        lookup.setdefault(s.id, i)
    model.seq_lookup = [lookup.get(i, 0xFFFF) for i in range(max(lookup) + 1)]
    model.global_loops = []

    for si, (aid, var, action) in enumerate(named):
        f0 = action.frame_range[0]
        for bi, chans in _bone_channels(action, idx).items():
            bone = model.bones[bi]
            for prop, n, defaults, attr, conv in _bone_specs(ordered[bi], mirror_x):
                keys = _sample(chans, prop, n, defaults, fps, f0=f0)
                if keys:
                    tr = getattr(bone, attr)
                    _store(tr, si, nseq, keys, conv)
                    tr.interpolation_type = _interp_of(chans, prop)

    for b in model.bones:
        for tr in (b.translation, b.rotation, b.scale):
            if tr is None or tr.global_sequence >= 0:
                continue
            if not tr.timelines:
                tr.timelines = [[] for _ in range(nseq)]
            while len(tr.timelines) < nseq:
                tr.timelines.append([])

    print("[M2] built %d sequence(s) from named actions: %s"
          % (nseq, ", ".join("%s%s" % (names.animation_name(a),
                                        "_%d" % v if v else "")
                             for a, v, _ in named)), flush=True)
    return True


def _build_animation(model, arm, ordered, fps, mirror_x):
    ad = arm.animation_data
    action = ad.action if ad else None
    if action is None:
        # No animation: a single 1-frame "Stand" so the model still has a sequence.
        seq = M2Sequence()
        seq.id = 0
        seq.duration = 1000
        seq.start_timestamp = 0
        seq.end_timestamp = 1000
        model.sequences = [seq]
        model.seq_lookup = [0]
        return

    fr = action.frame_range
    f0, f1 = fr[0], fr[1]
    seq = M2Sequence()
    seq.id = 0                                  # 0 = Stand (loops by default)
    seq.flags = 0x20                            # resident
    seq.duration = max(1, _ms(f1 - f0, fps))
    seq.start_timestamp = 0
    seq.end_timestamp = seq.duration
    _apply_blend(seq, action)
    _fix_degenerate_duration(seq)
    model.sequences = [seq]
    model.seq_lookup = [0]

    fmap = _fcurve_map(action)
    for i, db in enumerate(ordered):
        base = 'pose.bones["%s"].' % db.name
        # Pose channels are in BONE space; M2 tracks are in model space. They
        # only coincide for an axis-aligned rest bone: see _bone_specs.
        _, _, _, _, loc_conv = _bone_specs(db, mirror_x)[0]
        _, _, _, _, rot_conv = _bone_specs(db, mirror_x)[1]
        # translation
        frames = _keyframe_frames(fmap, base + "location", 3)
        if frames:
            tl = []
            for f in frames:
                v = [fmap[(base + "location", k)].evaluate(f) if (base + "location", k) in fmap else 0.0
                     for k in range(3)]
                tl.append((_ms(f - f0, fps), loc_conv(v)))
            model.bones[i].translation.timelines = [tl]
            model.bones[i].translation.interpolation_type = _interp_fcurves(
                fmap, base + "location", 3)
        # rotation (Blender quat is w,x,y,z -> M2 x,y,z,w)
        frames = _keyframe_frames(fmap, base + "rotation_quaternion", 4)
        if frames:
            tl = []
            for f in frames:
                q = [fmap[(base + "rotation_quaternion", k)].evaluate(f)
                     if (base + "rotation_quaternion", k) in fmap else (1.0 if k == 0 else 0.0)
                     for k in range(4)]
                tl.append((_ms(f - f0, fps), rot_conv(q)))
            model.bones[i].rotation.timelines = [tl]
            model.bones[i].rotation.interpolation_type = _interp_fcurves(
                fmap, base + "rotation_quaternion", 4)
        # scale
        frames = _keyframe_frames(fmap, base + "scale", 3)
        if frames:
            tl = []
            for f in frames:
                v = [fmap[(base + "scale", k)].evaluate(f) if (base + "scale", k) in fmap else 1.0
                     for k in range(3)]
                tl.append((_ms(f - f0, fps), (v[0], v[1], v[2])))
            model.bones[i].scale.timelines = [tl]
            model.bones[i].scale.interpolation_type = _interp_fcurves(
                fmap, base + "scale", 3)


# ---------------------------------------------------------------------------
def _attachment_id(obj):
    """The M2 attachment id for an empty, or None if it isn't one."""
    aid = obj.get("m2_att_id")
    if aid is not None:
        return int(aid)
    label = obj.name.split("_attach_", 1)[-1] if "_attach_" in obj.name else None
    if label is None:
        return None
    label = label.rsplit(".", 1)[0] if label.rsplit(".", 1)[-1].isdigit() \
        and "." in label else label
    if label.isdigit():
        return int(label)
    for k, v in names.ATTACHMENT_NAMES.items():
        if v == label:
            return k
    return None


HELPER_BOX_PROPS = ("m2_bounding_box", "m2_collision_box")


def is_helper_box(obj):
    """The editable render-bounds / collision-box objects: never geometry."""
    return any(obj.get(p) for p in HELPER_BOX_PROPS)


def bounds_override_from_objects(objects, mirror_x, prop="m2_bounding_box"):
    """World-space AABB of the box object flagged with ``prop``, in WoW space, or None."""
    box = next((o for o in objects if o.type == "MESH" and o.get(prop)), None)
    if box is None or not box.data.vertices:
        return None
    mw = box.matrix_world
    pts = [mw @ v.co for v in box.data.vertices]
    xs = [p.x for p in pts]; ys = [p.y for p in pts]; zs = [p.z for p in pts]

    def wow(p):
        return (-p[0], p[1], p[2]) if mirror_x else (p[0], p[1], p[2])

    a = wow((min(xs), min(ys), min(zs)))
    b = wow((max(xs), max(ys), max(zs)))
    return ((min(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2])),
            (max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2])))


def _build_cameras(model, objects, mirror_x):
    """Build M2Cameras from the Blender camera objects in the scene."""
    def wow(v):
        return (-v[0], v[1], v[2]) if mirror_x else (v[0], v[1], v[2])

    cams = [o for o in objects if o.type == "CAMERA"]
    out = []
    for cobj in cams:
        c = M2Camera()
        c.type = int(cobj.data.get("m2_camera_type", cobj.get("m2_camera_type", -1)))
        c.near_clip = float(cobj.data.clip_start)
        c.far_clip = float(cobj.data.clip_end)
        # Vertical FOV in radians.
        try:
            fov = cobj.data.angle_y
        except Exception:  # noqa: BLE001
            fov = cobj.data.angle
        c.fov_static = float(fov)
        c.fov = None            # writer emits a single-key fov track on Cata+

        loc = cobj.matrix_world.translation
        c.position_base = wow((loc.x, loc.y, loc.z))

        # Target: the aim constraint's target, on the camera or its rig empty;
        # else a point 1 unit down -Z.
        target = None
        holders = [cobj] + ([cobj.parent] if cobj.parent is not None else [])
        for holder in holders:
            for con in holder.constraints:
                if con.type in ("DAMPED_TRACK", "TRACK_TO") and con.target is not None:
                    target = con.target
                    break
            if target is not None:
                break
        if cobj.parent is not None and cobj.parent.get("m2_camera_rig"):
            roll = float(cobj.rotation_euler.z)
            if abs(roll) > 1e-6:
                c.roll = M2AnimTrack()
                c.roll.interpolation_type = 1
                c.roll.timelines = [[(0, roll)]]
        if target is not None:
            t = target.matrix_world.translation
            c.target_base = wow((t.x, t.y, t.z))
        else:
            fwd = cobj.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
            c.target_base = wow((loc.x + fwd.x, loc.y + fwd.y, loc.z + fwd.z))
        out.append(c)

    model.cameras = out
    if out:
        by_type = {}
        for i, c in enumerate(out):
            if c.type >= 0:
                by_type.setdefault(int(c.type), i)
        n_slots = (max(by_type) + 1) if by_type else len(out)
        model.camera_lookup = [by_type.get(k, 0xFFFF) for k in range(n_slots)]
        print("[M2] cameras: %d exported, lookup=%s"
              % (len(out), model.camera_lookup), flush=True)
    else:
        model.camera_lookup = []
    return out


def _build_attachments(model, objects, arm, idx, mirror_x):
    """Rebuild the attachment table from the empties in the scene."""
    if arm is None:
        return
    out = []
    for obj in objects:
        if obj.type != "EMPTY":
            continue
        aid = _attachment_id(obj)
        if aid is None:
            continue
        bi = -1
        db = None
        pose_bone = None
        if obj.parent_type == "BONE" and obj.parent_bone:
            bi = idx.get(obj.parent_bone, -1)
            db = arm.data.bones.get(obj.parent_bone)
            pose_bone = arm.pose.bones.get(obj.parent_bone)
        if bi < 0:
            stored = obj.get("m2_att_bone")
            bi = int(stored) if stored is not None else -1

        empty_arm = arm.matrix_world.inverted() @ obj.matrix_world.translation
        if db is not None and pose_bone is not None:
            local = pose_bone.matrix.inverted() @ empty_arm
            m = db.matrix_local @ local
        else:
            m = empty_arm

        att = M2Attachment()
        att.id = aid
        att.bone = bi if bi >= 0 else 0
        att.position = ((-m.x, m.y, m.z) if mirror_x else (m.x, m.y, m.z))
        out.append(att)

    out.sort(key=lambda a: a.id)
    model.attachments = out
    if out:
        lookup = {}
        for i, a in enumerate(out):
            lookup.setdefault(a.id, i)
        model.attachment_lookup = [lookup.get(i, 0xFFFF)
                                   for i in range(max(lookup) + 1)]
    else:
        model.attachment_lookup = []


def _build_events(model, objects, arm, mirror_x, source_model):
    """Rebuild the M2 event table from event empties in the scene."""
    import json as _json
    nb = len(model.bones)
    n_seq = len(model.sequences)
    # (id, var) -> output sequence index. Events store timings by sequence
    out_seq_key_to_idx = {}
    for i, s in enumerate(model.sequences):
        out_seq_key_to_idx.setdefault(
            "%d-%d" % (int(s.id), int(getattr(s, "variation_index", 0))), i)

    def _fix_times(raw):
        """Return per-output-sequence list of int timestamps."""
        try:
            data = _json.loads(raw or "{}")
        except Exception:  # noqa: BLE001
            return [[] for _ in range(n_seq)]

        out = [[] for _ in range(n_seq)]
        if isinstance(data, dict):
            for key, ts in data.items():
                if not isinstance(ts, (list, tuple)) or not ts:
                    continue
                idx = out_seq_key_to_idx.get(str(key))
                if idx is None:
                    continue
                out[idx] = [int(x) for x in ts]
        elif isinstance(data, list):
            # Legacy positional list; assume same-order same-count.
            for i, ts in enumerate(data):
                if i >= n_seq:
                    break
                if isinstance(ts, (list, tuple)) and ts:
                    out[i] = [int(x) for x in ts]
        return out

    out = []
    for obj in objects:
        if obj.type != "EMPTY":
            continue
        ident = obj.get("m2_event_id")
        if ident is None:
            continue
        ev = M2Event()
        ev.identifier = str(ident)[:4]
        ev.data = int(obj.get("m2_event_data", 0) or 0)
        # Prefer the empty's actual parent bone; fall back to the stored index.
        bi = -1
        db = None
        pose_bone = None
        if obj.parent_type == "BONE" and obj.parent_bone and arm is not None:
            db = arm.data.bones.get(obj.parent_bone)
            pose_bone = arm.pose.bones.get(obj.parent_bone)
            try:
                bi = int(obj.parent_bone.split("_")[-1])
            except Exception:  # noqa: BLE001
                bi = -1
        if bi < 0:
            stored = obj.get("m2_event_bone")
            bi = int(stored) if stored is not None else 0
        if not (0 <= bi < nb):
            print("[M2] event %s: bone %d out of range (0..%d) - dropped"
                  % (ev.identifier, bi, nb - 1), flush=True)
            continue
        ev.bone = bi
        if arm is not None and db is not None and pose_bone is not None:
            empty_arm = arm.matrix_world.inverted() @ obj.matrix_world.translation
            local = pose_bone.matrix.inverted() @ empty_arm
            m = db.matrix_local @ local
            pos = ((-m.x, m.y, m.z) if mirror_x else (m.x, m.y, m.z))
        else:
            wp = obj.matrix_world.translation
            pos = ((-wp.x, wp.y, wp.z) if mirror_x else (wp.x, wp.y, wp.z))
        ev.position = pos
        ev.timestamps = _fix_times(obj.get("m2_event_times"))
        out.append(ev)

    if out:
        model.events = out
        n_fire = sum(1 for e in out for t in e.timestamps if t)
        print("[M2] wrote %d event(s) from scene empties (%d have firing timestamps)"
              % (len(out), n_fire), flush=True)
        return

    # Fallback: no scene empties -> carry source events verbatim (old flow).
    if source_model is not None and getattr(source_model, "events", None):
        kept = [e for e in source_model.events if 0 <= e.bone < nb]
        dropped = len(source_model.events) - len(kept)
        model.events = kept
        print("[M2] no event empties in scene; carried %d event(s) from source%s"
              % (len(kept), " (%d dropped: bone out of range)" % dropped if dropped else ""),
              flush=True)
        n_src, n_out = len(source_model.sequences), len(model.sequences)
        if n_src != n_out:
            print("[M2] WARNING: source has %d sequences but %d are being exported; "
                  "event timings are per-sequence, so any animation whose index "
                  "moved will trigger the wrong event (sheathe/draw included)."
                  % (n_src, n_out), flush=True)


# ---------------------------------------------------------------------------
BLEND_OPAQUE, BLEND_ALPHA_KEY, BLEND_ALPHA = 0, 1, 2
BLEND_NO_ALPHA_ADD, BLEND_ADD, BLEND_MOD, BLEND_MOD2X, BLEND_BLEND_ADD = 3, 4, 5, 6, 7

_BLEND_BY_NAME = {
    "OPAQUE": BLEND_OPAQUE, "CLIP": BLEND_ALPHA_KEY, "HASHED": BLEND_ALPHA,
    "BLEND": BLEND_ALPHA,
    "ADD": BLEND_ADD, "MOD": BLEND_MOD, "MOD2X": BLEND_MOD2X,
    "ALPHA": BLEND_ALPHA, "ALPHA_KEY": BLEND_ALPHA_KEY,
    "NO_ALPHA_ADD": BLEND_NO_ALPHA_ADD, "BLEND_ADD": BLEND_BLEND_ADD,
}

# M2Material.flags
MF_UNLIT, MF_UNFOGGED, MF_TWO_SIDED = 0x01, 0x02, 0x04
MF_DEPTH_TEST_OFF, MF_DEPTH_WRITE_OFF = 0x08, 0x10

_DIGITS = re.compile(r"(\d{4,})")


def _ids_from(value):
    """Parse one or more FileDataIDs out of a property value."""
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [int(value)] if int(value) else []
    out = []
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
        else:
            m = _DIGITS.search(part)
            if m:
                out.append(int(m.group(1)))
    return out


def _image_ids(bmat):
    """FileDataIDs implied by the material's image texture nodes, in node order."""
    if not getattr(bmat, "use_nodes", False) or bmat.node_tree is None:
        return []
    out = []
    for node in bmat.node_tree.nodes:
        if node.type != "TEX_IMAGE" or node.image is None:
            continue
        for src in (node.image.name, getattr(node.image, "filepath", "") or ""):
            ids = _ids_from(src)
            if ids:
                out.append(ids[0])
                break
    return out


def _blend_of(bmat):
    """M2 blend mode for a Blender material."""
    explicit = bmat.get("m2_blend_mode")
    if explicit is not None:
        if isinstance(explicit, str):
            return _BLEND_BY_NAME.get(explicit.strip().upper(), BLEND_OPAQUE)
        return int(explicit)
    modern = getattr(bmat, "surface_render_method", None)  # Blender 4.2+
    if modern:
        return BLEND_ALPHA if str(modern).upper() == "BLENDED" else BLEND_OPAQUE
    legacy = getattr(bmat, "blend_method", None)           # Blender <= 4.1
    if legacy:
        return _BLEND_BY_NAME.get(str(legacy).upper(), BLEND_OPAQUE)
    return BLEND_OPAQUE


def _flags_of(bmat):
    """M2 render flags from explicit properties."""
    explicit = bmat.get("m2_render_flags")
    if explicit is not None:
        return int(explicit)
    flags = 0
    if bmat.get("m2_two_sided"):
        flags |= MF_TWO_SIDED
    if bmat.get("m2_unlit"):
        flags |= MF_UNLIT
    if bmat.get("m2_unfogged"):
        flags |= MF_UNFOGGED
    if bmat.get("m2_no_depth_test"):
        flags |= MF_DEPTH_TEST_OFF
    if bmat.get("m2_no_depth_write"):
        flags |= MF_DEPTH_WRITE_OFF
    return flags


TRANSPARENCY_OPAQUE = 0x7FFF     # fixed16 1.0


def _transparency_of(bmat):
    """Fixed16 (0..0x7FFF) transparency for a material, 0x7FFF = fully opaque."""
    if bmat is None:
        return TRANSPARENCY_OPAQUE
    val = bmat.get("m2_transparency")
    if val is None:
        val = bmat.get("m2_alpha")
    if val is None:
        return TRANSPARENCY_OPAQUE
    val = float(val)
    if val > 1.0:                       # already a fixed16 value
        return max(0, min(TRANSPARENCY_OPAQUE, int(round(val))))
    return max(0, min(TRANSPARENCY_OPAQUE, int(round(val * TRANSPARENCY_OPAQUE))))


class _MatSpec:
    """One Blender material resolved to the M2 tables it needs."""
    __slots__ = ("tex_ids", "tex_types", "tex_paths", "blend", "flags", "shader",
                 "layer", "material_index", "combo_index", "count", "origin",
                 "transparency", "batch_flags", "color", "coords", "transforms",
                 "weights", "tex_flags")


# M2Batch.flags retail writes on ordinary body batches ("static texture").
DEFAULT_BATCH_FLAGS = 0x10
NO_TRANSFORM = 0xFFFF
# Global flags meaning "animation data lives outside the M2": 0x2000 (Legion
# chunked .anim files), 0x100000 (skeleton .skel file) and 0x200000 (upgraded
# format, chunked .anim). An export embeds its sequences, so these must be off.
EXTERNAL_ANIM_FLAGS = 0x2000 | 0x100000 | 0x200000


DEFAULT_TEXTURE_FLAGS = 0x3          # wrap U and V, what retail file textures carry


def _spec_batch_defaults(spec, n):
    spec.tex_flags = [DEFAULT_TEXTURE_FLAGS] * n
    spec.batch_flags = DEFAULT_BATCH_FLAGS
    spec.color = -1
    spec.coords = list(range(n))          # layer k samples UV set k
    spec.transforms = [NO_TRANSFORM] * n
    spec.weights = [0] * n
    return spec


# Geoset GROUP (skin_section_id // 100) -> the texture type WoW conventionally
GEOSET_GROUP_TEXTURE_TYPE = {
    0: 1,    # base body -> composited character skin
    1: 6,    # hair
    2: 6,    # facial 1 (beard)
    3: 6,    # facial 2
    4: 6,    # facial 3
    15: 2,   # cloak -> its own object skin
    17: 2,   # eye glow
    19: 2,   # tail
}
DEFAULT_GEOSET_TEXTURE_TYPE = 1   # armour is painted onto the composited skin


def geoset_material_map(source_model):
    """``{geoset_id: _MatSpec}`` describing how the SOURCE M2 drew each geoset."""
    out = {}
    skin = getattr(source_model, "skin", None)
    if skin is None:
        return out
    for b in skin.batches:
        if not (0 <= b.submesh_index < len(skin.submeshes)):
            continue
        gid = skin.submeshes[b.submesh_index].skin_section_id
        if gid in out:
            continue                      # first batch wins (lowest layer)
        texes = []
        for k in range(max(1, b.texture_count)):
            idx = b.texture_combo_index + k
            if 0 <= idx < len(source_model.texture_lookup):
                ti = source_model.texture_lookup[idx]
                if 0 <= ti < len(source_model.textures):
                    texes.append(source_model.textures[ti])
        if not texes:
            continue
        spec = _MatSpec()
        spec.tex_ids = [t.file_data_id for t in texes]
        spec.tex_types = [t.type for t in texes]
        spec.tex_paths = ["" if re.fullmatch(r"FileDataID_\d+", t.filename or "") else (t.filename or "")
                          for t in texes]
        spec.count = len(texes)
        mi = b.material_index
        if 0 <= mi < len(source_model.materials):
            spec.blend = source_model.materials[mi].blend_mode
            spec.flags = source_model.materials[mi].flags
        else:
            spec.blend, spec.flags = BLEND_OPAQUE, 0
        spec.shader = b.shader_id
        spec.layer = b.material_layer
        spec.transparency = TRANSPARENCY_OPAQUE
        spec.origin = "source"
        _spec_batch_defaults(spec, len(texes))
        spec.tex_flags = [int(t.flags) for t in texes]
        spec.batch_flags = int(b.flags)
        spec.color = int(b.color_index)
        for k in range(len(texes)):
            ci, ti, wi = b.texture_coord_combo + k, b.texture_transform_combo + k, b.texture_weight_combo + k
            spec.coords[k] = (int(source_model.tex_coord_combos[ci])
                              if 0 <= ci < len(source_model.tex_coord_combos) else k)
            if 0 <= ti < len(source_model.tex_transform_combos):
                spec.transforms[k] = int(source_model.tex_transform_combos[ti])
            if 0 <= wi < len(source_model.tex_weight_combos):
                spec.weights[k] = int(source_model.tex_weight_combos[wi])
        out[gid] = spec
    return out


def _material_is_configured(bmat):
    """Does this material actually say which texture to use?"""
    if bmat is None:
        return False
    for key in ("m2_texture_ids", "m2_texture_id"):
        if _ids_from(bmat.get(key)):
            return True
    # A non-zero texture type is meaningful on its own (composited skin, etc.)
    if any(t for t in _ids_from(bmat.get("m2_texture_types"))):
        return True
    return bool(_image_ids(bmat)) or bool(_ids_from(bmat.name))


def _copy_spec(src):
    spec = _MatSpec()
    for slot in ("tex_ids", "tex_types", "blend", "flags", "shader", "layer",
                 "count", "origin", "transparency"):
        setattr(spec, slot, getattr(src, slot))
    spec.tex_ids = list(spec.tex_ids)
    spec.tex_types = list(spec.tex_types)
    spec.tex_paths = list(getattr(src, "tex_paths", ["" for _ in spec.tex_ids]))
    _spec_batch_defaults(spec, len(spec.tex_ids))
    for slot in ("batch_flags", "color"):
        if getattr(src, slot, None) is not None:
            setattr(spec, slot, getattr(src, slot))
    for slot in ("coords", "transforms", "weights", "tex_flags"):
        v = getattr(src, slot, None)
        if v:
            setattr(spec, slot, list(v))
    return spec


def _autofill_spec(geoset_id, geoset_map, default_texture_id, character_style):
    """Best-effort material for a mesh whose Blender material says nothing."""
    hit = geoset_map.get(geoset_id)
    if hit is not None:
        return _copy_spec(hit)
    spec = _MatSpec()
    ttype = 0
    if character_style:
        ttype = GEOSET_GROUP_TEXTURE_TYPE.get(geoset_id // 100,
                                              DEFAULT_GEOSET_TEXTURE_TYPE)
    # A non-zero type is supplied by the client and needs no FileDataID; a
    # type-0 slot does, so fall back to the export field there.
    spec.tex_ids = [0 if ttype else (default_texture_id or 0)]
    spec.tex_types = [ttype]
    spec.tex_paths = [""]
    spec.count = 1
    spec.blend = BLEND_OPAQUE
    spec.flags = 0
    spec.shader = 0
    spec.layer = 0
    spec.transparency = TRANSPARENCY_OPAQUE
    spec.origin = "convention" if ttype else "default"
    _spec_batch_defaults(spec, 1)
    return spec


def _resolve_material(bmat, default_texture_id):
    """Everything the M2 render path needs for one Blender material."""
    spec = _MatSpec()
    ids = _ids_from(bmat.get("m2_texture_ids")) if bmat is not None else []
    if not ids and bmat is not None:
        ids = _ids_from(bmat.get("m2_texture_id"))
    if not ids and bmat is not None:
        ids = _image_ids(bmat)
    if not ids and bmat is not None:
        ids = _ids_from(bmat.name)
    if not ids:
        ids = [default_texture_id] if default_texture_id else [0]
    spec.tex_ids = ids
    types = _ids_from(bmat.get("m2_texture_types")) if bmat is not None else []
    spec.tex_types = [types[i] if i < len(types) else 0 for i in range(len(ids))]
    # Hardcoded texture path(s) baked into the M2's inline texture
    # record. Comma-separated so the same slot semantics as
    # m2_texture_ids apply; a shorter list is padded with empty
    # strings so texture_type entries beyond the last supplied path
    # still work.
    raw_paths = str(bmat.get("m2_texture_paths", "") or "") if bmat is not None else ""
    if raw_paths:
        parts = [p.strip() for p in raw_paths.split(",")]
        # "FileDataID_123" is the importer's display name, never a real path.
        parts = ["" if re.fullmatch(r"FileDataID_\d+", p) else p for p in parts]
        spec.tex_paths = [parts[i] if i < len(parts) else ""
                          for i in range(len(ids))]
    else:
        spec.tex_paths = ["" for _ in ids]
    spec.blend = _blend_of(bmat) if bmat is not None else BLEND_OPAQUE
    spec.flags = _flags_of(bmat) if bmat is not None else 0
    spec.shader = int(bmat.get("m2_shader_id", 0)) if bmat is not None else 0
    spec.layer = int(bmat.get("m2_material_layer", 0)) if bmat is not None else 0
    spec.count = len(ids)
    spec.transparency = _transparency_of(bmat)
    if spec.transparency < TRANSPARENCY_OPAQUE and spec.blend == BLEND_OPAQUE \
            and (bmat is None or bmat.get("m2_blend_mode") is None):
        spec.blend = BLEND_ALPHA
    spec.origin = "material"
    _spec_batch_defaults(spec, len(ids))
    if bmat is not None:
        if bmat.get("m2_batch_flags") is not None:
            spec.batch_flags = int(bmat["m2_batch_flags"])
        if bmat.get("m2_color_index") is not None:
            spec.color = int(bmat["m2_color_index"])
        for slot, key, default in (("coords", "m2_texture_coords", 0),
                                   ("transforms", "m2_texture_transforms", NO_TRANSFORM),
                                   ("weights", "m2_texture_weights", 0),
                                   ("tex_flags", "m2_texture_flags", DEFAULT_TEXTURE_FLAGS)):
            vals = _ids_from(bmat.get(key))
            if vals:
                setattr(spec, slot, [vals[i] if i < len(vals) else default for i in range(len(ids))])
    return spec


def _geoset_id_of(obj):
    name_id = geoset_id_from_name(obj.name)
    if name_id is None:
        name_id = obj.get("m2_skin_section_id")
    return int(name_id) if name_id is not None else 0


def _has_real_texture(spec):
    """True if the material points at an actual texture FILE (type 0 + an id)."""
    return any(t == 0 and i for t, i in zip(spec.tex_types, spec.tex_ids))


def _remap_track(track, kept):
    """A copy of ``track`` whose per-sequence timelines follow the exported
    sequence order (``kept`` = original index per exported sequence)."""
    out = M2AnimTrack()
    out.interpolation_type = track.interpolation_type
    out.global_sequence = track.global_sequence
    src = list(track.timelines or [])
    if track.global_sequence >= 0 or kept is None:
        out.timelines = [list(tl) for tl in src]
    else:
        out.timelines = [list(src[i]) if i < len(src) else [] for i in kept]
    return out


def _carry_tracks(model, specs, source_tracks):
    """Copy the colour and texture-transform tracks the materials reference
    from the source model, renumbering them; return the index maps."""
    kept = getattr(model, "_kept_sequences", None)
    if kept is None and model.sequences:
        # Sequences were rebuilt from action names: per-sequence keys cannot be
        # matched, but global-sequence tracks (the usual UV scroll) still can.
        kept = []
    colors_src = (source_tracks or {}).get("colors") or []
    xforms_src = (source_tracks or {}).get("transforms") or []
    color_map, xform_map = {}, {}
    model.colors, model.transforms = [], []
    for spec in specs:
        c = getattr(spec, "color", -1)
        if 0 <= c < len(colors_src) and c not in color_map:
            color_map[c] = len(model.colors)
            ct, at = colors_src[c]
            model.colors.append((_remap_track(ct, kept), _remap_track(at, kept)))
        for t in getattr(spec, "transforms", ()):
            if 0 <= t < len(xforms_src) and t != NO_TRANSFORM and t not in xform_map:
                xform_map[t] = len(model.transforms)
                tr, rot, sc = xforms_src[t]
                model.transforms.append((_remap_track(tr, kept), _remap_track(rot, kept),
                                         _remap_track(sc, kept)))
    model.n_colors = len(model.colors)
    model.n_texture_transforms = len(model.transforms)
    if model.transforms or model.colors:
        print("[M2] carried %d texture transform(s) and %d colour track(s) from the source"
              % (len(model.transforms), len(model.colors)), flush=True)
    return color_map, xform_map


def _build_material_tables(model, meshes, default_texture_id, geoset_map=None,
                           source_tracks=None):
    """Build the texture / renderflag / combo tables for every mesh slot."""
    geoset_map = geoset_map or {}
    plan = {}          # (obj name, slot) -> _MatSpec
    order = []         # dedup identical specs so the tables stay small
    by_signature = {}

    character_style = any(_geoset_id_of(o) >= 100 for o in meshes)

    for obj in meshes:
        slots = list(obj.data.materials) or [None]
        gid = _geoset_id_of(obj)
        for si, bmat in enumerate(slots):
            if _material_is_configured(bmat):
                spec = _resolve_material(bmat, default_texture_id)
            else:
                # Only a material that says nothing gets filled in: from what
                # the source used for this geoset, else the group convention.
                spec = _autofill_spec(gid, geoset_map, default_texture_id,
                                      character_style)
                # Explicit render state still wins even when the texture had to
                # be inferred, so a user tweak isn't silently discarded.
                if bmat is not None:
                    if bmat.get("m2_blend_mode") is not None:
                        spec.blend = _blend_of(bmat)
                    if bmat.get("m2_render_flags") is not None:
                        spec.flags = _flags_of(bmat)
                    if bmat.get("m2_shader_id") is not None:
                        spec.shader = int(bmat["m2_shader_id"])
            sig = (tuple(spec.tex_ids), tuple(spec.tex_types), spec.blend,
                   spec.flags, spec.shader, spec.layer, spec.transparency)
            shared = by_signature.get(sig)
            if shared is None:
                by_signature[sig] = spec
                order.append(spec)
                shared = spec
            plan[(obj.name, si)] = shared

    if not order:
        spec = _resolve_material(None, default_texture_id)
        spec.origin = "material"
        order = [spec]
        plan[(None, 0)] = spec

    textures, tex_key = [], {}
    materials, mat_key = [], {}
    lookup = []
    coord, weight, transform = [], [], []
    weight_values, weight_key = [], {}   # distinct transparency -> weight track
    color_map, xform_map = _carry_tracks(model, order, source_tracks)
    for spec in order:
        spec.color = color_map.get(getattr(spec, "color", -1), -1)

    for spec in order:
        # renderflags row (deduped)
        mk = (spec.flags, spec.blend)
        mi = mat_key.get(mk)
        if mi is None:
            mi = len(materials)
            mat_key[mk] = mi
            materials.append(M2Material(spec.flags, spec.blend))
        spec.material_index = mi

        # A transparency track per distinct alpha value; the batch's combo slot
        # points at it (all layers of one material share the material's alpha).
        tv = getattr(spec, "transparency", TRANSPARENCY_OPAQUE)
        wi = weight_key.get(tv)
        if wi is None:
            wi = len(weight_values)
            weight_key[tv] = wi
            weight_values.append(tv)

        # texture rows + a contiguous slice of the combo tables per layer
        spec.combo_index = len(lookup)
        tex_paths = getattr(spec, "tex_paths", ["" for _ in spec.tex_ids])
        for i, fid in enumerate(spec.tex_ids):
            ttype = spec.tex_types[i]
            path = tex_paths[i] if i < len(tex_paths) else ""
            # Dedup key includes the path so two materials that share a
            # (type, fid) but bake different hardcoded paths still get
            # their own M2Texture rows — the client picks the row via
            # the batch's texture_combo_index so distinct rows can
            # legitimately point at different files.
            tflags = getattr(spec, "tex_flags", None) or []
            tflag = int(tflags[i]) if i < len(tflags) else DEFAULT_TEXTURE_FLAGS
            tk = (ttype, fid, path, tflag)
            ti = tex_key.get(tk)
            if ti is None:
                ti = len(textures)
                tex_key[tk] = ti
                textures.append(M2Texture(ttype, tflag, path, fid))
            lookup.append(ti)
            coords = getattr(spec, "coords", None) or []
            coord.append(int(coords[i]) if i < len(coords) else 0)   # UV set (0 = UVMap)
            weight.append(wi)        # this material's transparency track
            xf = getattr(spec, "transforms", None) or []
            transform.append(xform_map.get(xf[i], NO_TRANSFORM) if i < len(xf) else NO_TRANSFORM)

    model.textures = textures
    model.materials = materials
    model.texture_lookup = lookup
    model.tex_coord_combos = coord
    model.tex_weight_combos = weight
    model.tex_transform_combos = transform
    # textureIndicesById: texture type -> texture row. The client uses this
    # lookup when it swaps replaceable textures in (skin, hair, character
    # customization such as horns and blindfolds); without it those geosets
    # render untextured. Retail lists every type up to the highest present,
    # -1 where absent, and keeps the last row when a type repeats (type 0).
    tii = [0xFFFF] * (max((int(t.type) for t in textures), default=-1) + 1)
    for i, t in enumerate(textures):
        tii[int(t.type)] = i
    model.texture_indices_by_id = tii
    # Remember the transparency values so build_model_from_scene can emit one
    # weight track per value (keyframed for every sequence).
    model._weight_values = weight_values

    _TYPE_NAMES = {0: "file", 1: "skin(composited)", 2: "objectSkin",
                   3: "weaponBlade", 4: "weaponHandle", 5: "environment",
                   6: "hair", 7: "facialHair", 8: "skinExtra", 9: "uiSkin",
                   10: "taurenMane", 11: "monster1", 12: "monster2",
                   13: "monster3", 14: "itemIcon"}
    _OK_RUNTIME = {1, 2, 6, 7, 8}
    _ORIGIN = {"material": "from the Blender material",
               "source": "auto-filled from the source M2",
               "convention": "auto-filled by geoset convention",
               "default": "auto-filled from the Texture FileDataID field"}
    print("[M2] material bindings:", flush=True)
    seen_specs = []
    suspicious = []
    for (obj_name, slot), spec in sorted(plan.items(), key=lambda kv: str(kv[0])):
        if spec in seen_specs:
            continue
        seen_specs.append(spec)
        tex = ", ".join("%s:%s" % (_TYPE_NAMES.get(t, "type%d" % t), i)
                        for t, i in zip(spec.tex_types, spec.tex_ids))
        print("   %-38s slot %d -> [%s] blend=%d shader=%d  (%s)"
              % (obj_name, slot, tex, spec.blend, spec.shader,
                 _ORIGIN.get(spec.origin, spec.origin)), flush=True)
        for t, i in zip(spec.tex_types, spec.tex_ids):
            if t and t not in _OK_RUNTIME:
                suspicious.append((obj_name, t))
            elif t == 0 and not i:
                suspicious.append((obj_name, 0))
    for obj_name, t in suspicious[:6]:
        if t:
            print("[M2] WARNING: %s uses texture type %d (%s): the client only "
                  "binds that in its own specific case, so on an ordinary "
                  "character it resolves to NOTHING and the mesh is invisible "
                  "(use type 1 for client-composited skin, or type 0 with a "
                  "FileDataID)" % (obj_name, t, _TYPE_NAMES.get(t, "?")),
                  flush=True)
        else:
            print("[M2] WARNING: %s uses a type-0 texture with no FileDataID: "
                  "it will be untextured" % obj_name, flush=True)
    return plan


# ---------------------------------------------------------------------------
def _vertex_bones(mv, g2b):
    pairs = []
    for g in mv.groups:
        bi = g2b.get(g.group)
        if bi is not None and g.weight > 0:
            pairs.append((bi, g.weight))
    pairs.sort(key=lambda p: p[1], reverse=True)
    pairs = pairs[:4]
    w = [0, 0, 0, 0]
    idx = [0, 0, 0, 0]
    tot = sum(p[1] for p in pairs) or 1.0
    for k, (bi, wt) in enumerate(pairs):
        w[k] = max(0, min(255, int(round(wt / tot * 255))))
        idx[k] = bi
    if not pairs:                               # unweighted -> bind fully to bone 0
        w[0] = 255
    # fix rounding so weights sum to 255
    s = sum(w)
    if s and s != 255 and w[0]:
        w[0] += 255 - s
    return tuple(w), tuple(idx)


def _evaluated_mesh(obj):
    """The mesh as modifiers actually produce it, minus armature deformation."""
    if not obj.modifiers:
        return obj.data, None
    disabled = [md for md in obj.modifiers
                if md.type == "ARMATURE" and md.show_viewport]
    for md in disabled:
        md.show_viewport = False
    try:
        dg = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(dg)
        return eval_obj.to_mesh(), eval_obj
    except Exception:  # noqa: BLE001 - fall back to the raw mesh
        return obj.data, None
    finally:
        for md in disabled:
            md.show_viewport = True


def _build_mesh(model, objects, g2b, mirror_x, specs):
    """Vertices, triangles, submeshes and batches."""
    meshes = sorted((o for o in objects
                     if o.type == "MESH" and not is_helper_box(o)),
                    key=lambda o: (_geoset_id_of(o), o.name))
    vertices = []
    skin_verts = []
    triangles = []
    submeshes = []
    batches = []
    skipped = []
    for obj in meshes:
        mesh, owner = _evaluated_mesh(obj)
        mesh.calc_loop_triangles()
        try:
            mesh.calc_normals_split()
        except Exception:  # noqa: BLE001 (Blender 4.1+ computes automatically)
            pass
        mw = obj.matrix_world
        nmat = mw.to_3x3().inverted_safe().transposed()
        uv_layer = mesh.uv_layers.active
        uv2_layer = mesh.uv_layers.get("UVMap2")     # second set for multi-layer materials
        g2b_obj = {}
        for gi, vg in enumerate(obj.vertex_groups):
            j = g2b.get(vg.name)
            if j is not None:
                g2b_obj[gi] = j

        slots = list(mesh.materials) or [None]
        by_slot = {}
        for tri in mesh.loop_triangles:
            by_slot.setdefault(min(tri.material_index, len(slots) - 1),
                               []).append(tri)
        if not by_slot:
            # No faces at all (empty mesh, curve-only, or geometry that exists
            # solely as loose verts/edges). Never drop this quietly.
            skipped.append("%s (no faces)" % obj.name)
            if owner is not None:
                owner.to_mesh_clear()
            continue

        name_id = geoset_id_from_name(obj.name)
        if name_id is None:
            name_id = obj.get("m2_skin_section_id")
        geoset_id = int(name_id) if name_id is not None else 0

        for slot in sorted(by_slot):
            # Merge corners that share vertex + normal + UV into one M2 vertex
            vbase = len(vertices)
            corner_to_local = {}
            istart = len(triangles)

            def local_of(loop_index, vert_index, _cache=corner_to_local):
                mv = mesh.vertices[vert_index]
                try:
                    ln = mesh.loops[loop_index].normal
                    nrm = (nmat @ Vector(ln)).normalized()
                except Exception:  # noqa: BLE001
                    nrm = (nmat @ mv.normal).normalized()
                if uv_layer is not None:
                    u, vv = uv_layer.data[loop_index].uv
                    uv = (u, 1.0 - vv)
                else:
                    uv = (0.0, 0.0)
                if uv2_layer is not None:
                    u2, vv2 = uv2_layer.data[loop_index].uv
                    uv2 = (u2, 1.0 - vv2)
                else:
                    uv2 = (0.0, 0.0)
                key = (vert_index,
                       round(nrm.x, 4), round(nrm.y, 4), round(nrm.z, 4),
                       round(uv[0], 5), round(uv[1], 5),
                       round(uv2[0], 5), round(uv2[1], 5))
                l = _cache.get(key)
                if l is not None:
                    return l
                l = len(vertices)
                _cache[key] = l
                co = mw @ mv.co
                v = M2Vertex()
                v.pos = (-co.x, co.y, co.z) if mirror_x else (co.x, co.y, co.z)
                v.normal = (-nrm.x, nrm.y, nrm.z) if mirror_x else (nrm.x, nrm.y, nrm.z)
                v.uv1 = uv
                v.uv2 = uv2
                v.bone_weights, v.bone_indices = _vertex_bones(mv, g2b_obj)
                vertices.append(v)
                skin_verts.append(len(skin_verts))
                return l

            for tri in by_slot[slot]:
                for li, vidx in zip(tri.loops, tri.vertices):
                    triangles.append(local_of(li, vidx))

            sub = M2SubMesh()
            # Geoset id: the object NAME wins (so it survives editing/joining
            sub.skin_section_id = geoset_id
            sub.center_bone_index = int(obj.get("m2_center_bone", 0))
            sub.vertex_start = vbase
            sub.vertex_count = len(vertices) - vbase
            sub.index_start = istart
            sub.index_count = len(triangles) - istart
            sub_index = len(submeshes)
            submeshes.append(sub)

            spec = specs.get((obj.name, slot))
            if spec is None:
                spec = specs.get((None, 0)) or next(iter(specs.values()))

            b = M2Batch()
            b.flags = int(getattr(spec, "batch_flags", DEFAULT_BATCH_FLAGS)) & 0xFF
            b.submesh_index = sub_index
            b.material_index = spec.material_index
            b.texture_combo_index = spec.combo_index
            b.color_index = int(getattr(spec, "color", -1))
            b.shader_id = spec.shader
            b.material_layer = spec.layer
            b.texture_count = spec.count
            b.texture_coord_combo = spec.combo_index
            b.texture_weight_combo = spec.combo_index
            b.texture_transform_combo = spec.combo_index
            batches.append(b)

        if owner is not None:
            owner.to_mesh_clear()

    # Say exactly which objects made it into the file, so a mesh that silently
    # failed to export is obvious rather than something you find out in-game.
    built = {}
    for b in batches:
        s = submeshes[b.submesh_index]
        built.setdefault(s.skin_section_id, 0)
        built[s.skin_section_id] += 1
    print("   %d objects -> %d submeshes, %d batches, geoset ids %s"
          % (len(meshes), len(submeshes), len(batches), sorted(built)), flush=True)
    if skipped:
        print("[M2] WARNING: %d mesh object(s) exported NOTHING: %s"
              % (len(skipped), "; ".join(skipped)), flush=True)

    if len(skin_verts) > 0xFFFF:
        raise ValueError(
            "Model has %d vertices after merging; the .skin format is limited to "
            "65535. Decimate / reduce the mesh (or split it) before exporting."
            % len(skin_verts))

    skin = M2SkinProfile()
    skin.vertices = skin_verts
    skin.triangles = triangles
    skin.submeshes = submeshes
    skin.batches = batches
    model.skin = skin
    model.vertices = vertices


# ---------------------------------------------------------------------------
def build_model_from_scene(objects, fps=30, mirror_x=False, texture_file_id=0,
                           name="CustomModel", source_model=None):
    """Construct a complete M2Model from a raw Blender scene."""
    model = M2Model()
    model.version = 274
    model.name = name
    # global_flags: retail characters set 0x80, which is what tells the client
    model.global_flags = 0x80 | 0x400

    meta = _find_meta(objects)

    arm = next((o for o in objects if o.type == "ARMATURE"), None)
    g2b = {}
    idx = {}
    if arm is not None:
        ordered, idx = _build_bones(model, arm, mirror_x)
        g2b = {db.name: i for db, i in [(b, idx[b.name]) for b in ordered]}
        # Prefer the tag-based rebuild (imported clips). If there are no tagged
        if not _build_animation_multi(model, arm, ordered, idx, fps, mirror_x, meta):
            if not _build_animation_named(model, arm, ordered, idx, fps, mirror_x):
                _build_animation(model, arm, ordered, fps, mirror_x)
    else:
        # No armature: one static bone at the origin so the skin can bind.
        b = M2Bone(); b.parent = -1; b.flags = 0x200
        b.translation = M2AnimTrack(); b.rotation = M2AnimTrack(); b.scale = M2AnimTrack()
        model.bones.append(b)
        seq = M2Sequence(); seq.id = 0; seq.duration = 1000; seq.end_timestamp = 1000
        model.sequences = [seq]; model.seq_lookup = [0]

    # Textures / renderflags / combo tables come from the Blender materials, so
    # a multi-material model exports with its real blend modes and shaders.
    meshes = [o for o in objects
              if o.type == "MESH" and not is_helper_box(o)]
    geoset_map = geoset_material_map(source_model) if source_model else {}
    if geoset_map:
        print("[M2] source M2 supplies material data for %d geoset id(s)"
              % len(geoset_map), flush=True)
    if source_model is not None:
        # Keep the source's global flags (0x80 is what the glow shaders rely
        # on) but drop every "animation data lives outside the M2" bit: 0x2000
        # (Legion chunked .anim), 0x200000 (upgraded-format chunked .anim) and
        # 0x100000 (skeleton .skel file). Our sequences are embedded; with those
        # bits set the client looks for external tracks and crashes on a null
        # pointer while loading the model (character select, "ERROR #132").
        model.global_flags = (int(source_model.global_flags) & ~EXTERNAL_ANIM_FLAGS) | 0x80
        source_tracks = {"colors": list(source_model.colors), "transforms": list(source_model.transforms)}
    else:
        source_tracks = {"colors": [(anim_data.track_from_json(c), anim_data.track_from_json(a))
                                    for c, a in (meta or {}).get("colors", [])],
                         "transforms": [(anim_data.track_from_json(t), anim_data.track_from_json(r),
                                         anim_data.track_from_json(s))
                                        for t, r, s in (meta or {}).get("transforms", [])]}
    specs = _build_material_tables(model, meshes, texture_file_id, geoset_map, source_tracks)
    _build_mesh(model, objects, g2b, mirror_x, specs)

    nseq = max(1, len(model.sequences))
    weight_values = getattr(model, "_weight_values", None) or [TRANSPARENCY_OPAQUE]
    model.weights = []
    for tv in weight_values:
        wt = M2AnimTrack()
        wt.interpolation_type = 0
        wt.timelines = [[(0, int(tv) & 0xFFFF)] for _ in range(nseq)]
        model.weights.append(wt)
    model.n_texture_weights = len(model.weights)
    _build_attachments(model, objects, arm, idx, mirror_x)
    _build_cameras(model, objects, mirror_x)
    _build_events(model, objects, arm, mirror_x, source_model)

    if source_model is not None:
        model.bounding_min, model.bounding_max = source_model.bounding_min, source_model.bounding_max
        model.bounding_radius = source_model.bounding_radius
        model.collision_min, model.collision_max = source_model.collision_min, source_model.collision_max
        model.collision_radius = source_model.collision_radius
    else:
        meta_b = (_find_meta(objects) or {}).get("bounds") or {}
        if "bbox" in meta_b:
            model.bounding_min, model.bounding_max = tuple(meta_b["bbox"][0]), tuple(meta_b["bbox"][1])
            model.bounding_radius = float(meta_b["bbox"][2])
        if "collision" in meta_b:
            model.collision_min, model.collision_max = tuple(meta_b["collision"][0]), tuple(meta_b["collision"][1])
            model.collision_radius = float(meta_b["collision"][2])
    model.collision_override = bounds_override_from_objects(objects, mirror_x, "m2_collision_box")
    model.bounding_override = bounds_override_from_objects(objects, mirror_x)
    if model.bounding_override:
        print("[M2] using edited bounding box from 'M2_BoundingBox' object",
              flush=True)
    model.aux_chunks = {}
    model.skin_file_ids = []
    print("[M2] from-scratch: %d sequences, %d attachments, %d textures, "
          "%d materials, %d batches, geosets %s"
          % (len(model.sequences), len(model.attachments), len(model.textures),
             len(model.materials),
             len(model.skin.batches) if model.skin else 0,
             sorted({s.skin_section_id for s in model.skin.submeshes})
             if model.skin else []), flush=True)
    return model
