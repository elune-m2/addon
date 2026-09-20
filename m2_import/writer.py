"""Serialize an :class:`M2Model` back to disk."""

import struct
from .model import M2Event

_M2_HEADER_SIZE = 304
_MAX_PALETTE = 256       # GPU bone palette limit per submesh


def _encode_comp_quat(x, y, z, w):
    def e(q):
        s = round(q * 32767 + 32767) if q >= 0 else round(q * 32767 - 32768)
        return s & 0xFFFF
    return struct.pack("<4H", e(x), e(y), e(z), e(w))


# An empty modern M2Track: interp=0, global_seq=none, empty timestamp/value
# arrays. The client reads it and falls back to the channel default.
_EMPTY_TRACK = struct.pack("<HHIIII", 0, 0xFFFF, 0, 0, 0, 0)


class _Buf:
    def __init__(self, reserve):
        self.b = bytearray(reserve)

    def append(self, data):
        while len(self.b) % 4:
            self.b.append(0)
        off = len(self.b)
        self.b += data
        return off


# ---------------------------------------------------------------------------
def build_render_tables(model):
    """Generate boneCombos, the per-submesh palette slices, the skin ``bones``"""
    skin = model.skin
    if skin is None:
        return
    verts = model.vertices
    skin_bones = [(0, 0, 0, 0)] * len(skin.vertices)

    # Pass 1: build the shared palette across every skin vertex.
    palette = []
    local = {}
    max_infl = 1
    for sv, gv in enumerate(skin.vertices):
        v = verts[gv]
        locs = [0, 0, 0, 0]
        n = 0
        for k in range(4):
            if v.bone_weights[k] == 0:
                continue
            n += 1
            bi = v.bone_indices[k]
            li = local.get(bi)
            if li is None:
                li = len(palette)
                local[bi] = li
                palette.append(bi)
            locs[k] = li
        max_infl = max(max_infl, n)
        skin_bones[sv] = tuple(locs)

    if len(palette) <= _MAX_PALETTE:
        skin.bone_indices = skin_bones
        model.bone_lookup = palette
        for sub in skin.submeshes:
            sub.bone_combo_index = 0
            sub.bone_count = len(palette)
            # bone_influences is PER GEOSET (max weighted bones over this
            mi = 1
            start, end = sub.vertex_start, sub.vertex_start + sub.vertex_count
            for sv in range(start, min(end, len(skin.vertices))):
                v = verts[skin.vertices[sv]]
                n = sum(1 for k in range(4) if v.bone_weights[k] != 0)
                if n > mi:
                    mi = n
            sub.bone_influences = min(4, mi)
    else:
        # Too many bones for one palette: per-submesh palettes instead.
        _build_per_submesh_palettes(model)

    # Safety net: never let a batch point past the colour array (a null deref
    # there crashes the client). -1 means "no colour track".
    n_colors = max(len(model.colors), model.n_colors)
    for b in skin.batches:
        if b.color_index >= n_colors:
            b.color_index = -1

    if not model.key_bone_lookup:
        kb = {}
        for i, b in enumerate(model.bones):
            if b.key_bone_id is not None and b.key_bone_id >= 0:
                kb[b.key_bone_id] = i
        if kb:
            model.key_bone_lookup = [kb.get(k, 0xFFFF) for k in range(max(kb) + 1)]

    # Stale bone references crash the client outright: repair before anything
    # else, since the tables below are derived from the bone data.
    sanitize_bone_references(model, log=lambda m: print("[M2] " + m, flush=True))

    # The client culls sections against these, so they must be real values.
    build_section_bounds(model)

    # Every write path funnels through here, and bone_count is only meaningful
    # now, so this is the one place that sees a complete skin.
    validate_skin(skin, log=lambda m: print("[M2] " + m, flush=True))


def _build_per_submesh_palettes(model):
    """Fallback bone palette: give each submesh its own slice of boneCombos."""
    skin = model.skin
    verts = model.vertices
    bone_lookup = []
    skin_bones = [(0, 0, 0, 0)] * len(skin.vertices)
    for sub in skin.submeshes:
        palette, local, infl = [], {}, 1
        start, end = sub.vertex_start, sub.vertex_start + sub.vertex_count
        for sv in range(start, min(end, len(skin.vertices))):
            v = verts[skin.vertices[sv]]
            locs = [0, 0, 0, 0]
            n = 0
            for k in range(4):
                if v.bone_weights[k] == 0:
                    continue
                n += 1
                bi = v.bone_indices[k]
                li = local.get(bi)
                if li is None:
                    li = len(palette)
                    local[bi] = li
                    palette.append(bi)
                locs[k] = li
            infl = max(infl, n)
            skin_bones[sv] = tuple(locs)
        sub.bone_combo_index = len(bone_lookup)
        sub.bone_count = len(palette)
        sub.bone_influences = min(4, infl)
        bone_lookup.extend(palette)
    skin.bone_indices = skin_bones
    model.bone_lookup = bone_lookup
    print("[M2] note: used per-submesh bone palettes (shared exceeded %d)" % _MAX_PALETTE)


# ---------------------------------------------------------------------------
def _pack_vertex(v):
    return struct.pack(
        "<3f4B4B3f2f2f",
        v.pos[0], v.pos[1], v.pos[2],
        *(int(b) & 0xFF for b in v.bone_weights),
        *(int(b) & 0xFF for b in v.bone_indices),
        v.normal[0], v.normal[1], v.normal[2],
        v.uv1[0], v.uv1[1], v.uv2[0], v.uv2[1],
    )


def link_animation_variations(model):
    """Chain each animation id's variations and give them selection weights."""
    seqs = model.sequences
    if not seqs:
        return
    by_id = {}
    for i, s in enumerate(seqs):
        by_id.setdefault(s.id, []).append(i)

    for _id, idxs in by_id.items():
        idxs.sort(key=lambda i: seqs[i].variation_index)
        have_freq = any(seqs[i].frequency for i in idxs)
        # Link the chain (unless the file already provided links).
        if all(seqs[i].variation_next in (-1, 0) for i in idxs):
            for a, b in zip(idxs, idxs[1:]):
                seqs[a].variation_next = b
            seqs[idxs[-1]].variation_next = -1
        if not have_freq:
            n = len(idxs)
            share = 0x7FFF // n
            for i in idxs:
                seqs[i].frequency = share
            seqs[idxs[0]].frequency += 0x7FFF - share * n   # remainder to base


def _pack_sequence(s, bounds):
    flags = (s.flags | 0x20) & 0xFFFFFFFF
    (bx0, by0, bz0), (bx1, by1, bz1), radius = bounds
    freq = int(getattr(s, "frequency", 0)) & 0xFFFF
    vnext = int(getattr(s, "variation_next", -1))
    anext = int(getattr(s, "alias_next", 0)) & 0xFFFF
    blend_in = int(getattr(s, "blend_time_in", 150)) & 0xFFFF
    blend_out = int(getattr(s, "blend_time_out", 0)) & 0xFFFF
    return struct.pack(
        "<HHIfIHHIIHH6ffhH",
        s.id & 0xFFFF, s.variation_index & 0xFFFF, s.duration & 0xFFFFFFFF,
        float(getattr(s, "movespeed", 0.0)), flags, freq, 0, 0, 0,
        blend_in, blend_out,
        bx0, by0, bz0, bx1, by1, bz1, radius,
        vnext, anext,
    )


def _pack_track_value(value, kind):
    if kind == "compquat":
        return _encode_comp_quat(*value)
    if kind == "quat4f":
        return struct.pack("<4f", value[0], value[1], value[2], value[3])
    if kind == "fixed16":
        return struct.pack("<h", int(value))
    if kind == "splinevec3":                 # M2SplineKey<vec3>: value + 2 tangents
        x, y, z = value
        return struct.pack("<9f", x, y, z, 0, 0, 0, 0, 0, 0)
    if kind == "splinefloat":                # M2SplineKey<float>
        return struct.pack("<3f", float(value), 0.0, 0.0)
    return struct.pack("<3f", value[0], value[1], value[2])


def _pack_track_header(fields):
    """Pack the inline 20-byte M2Track header from _write_track's return."""
    interp, gseq, (tc, to), (vc, vo) = fields
    return struct.pack("<HHIIII", interp, gseq, tc, to, vc, vo)


def _render_array(buf, items, kinds, nseq, empty_count):
    """Lay out a colour/weight/transform-style array; return (count, offset)."""
    if items:
        infos = []
        for elem in items:
            tracks = elem if isinstance(elem, (tuple, list)) else (elem,)
            infos.append([_write_track(buf, t, nseq, k) for t, k in zip(tracks, kinds)])
        blob = b"".join(b"".join(_pack_track_header(f) for f in info) for info in infos)
        return len(items), buf.append(blob)
    if empty_count:
        blob = (_EMPTY_TRACK * len(kinds)) * empty_count
        return empty_count, buf.append(blob)
    return 0, 0


def _static_camera_track(value):
    """A one-key camera track carrying a single (spline) value at t=0."""
    from .model import M2AnimTrack
    t = M2AnimTrack()
    t.interpolation_type = 0
    t.global_sequence = -1
    t.timelines = [[(0, value)]]
    return t


def _write_cameras(buf, model, version):
    """Write the M2Camera array; return (count, offset)."""
    if not model.cameras:
        return 0, 0
    fov_is_track = version >= 265
    structs = bytearray()
    for cam in model.cameras:
        pos_t = cam.position or _static_camera_track((0.0, 0.0, 0.0))
        tgt_t = cam.target or _static_camera_track((0.0, 0.0, 0.0))
        roll_t = cam.roll or _static_camera_track(0.0)
        pos_h = _pack_track_header(_write_track(buf, pos_t, 1, "splinevec3"))
        tgt_h = _pack_track_header(_write_track(buf, tgt_t, 1, "splinevec3"))
        roll_h = _pack_track_header(_write_track(buf, roll_t, 1, "splinefloat"))
        if fov_is_track:
            fov_t = cam.fov or _static_camera_track(float(cam.fov_static or 0.0))
            fov_h = _pack_track_header(_write_track(buf, fov_t, 1, "splinefloat"))

        s = struct.pack("<i", cam.type)
        if not fov_is_track:
            s += struct.pack("<f", float(cam.fov_static or 0.0))
        s += struct.pack("<ff", cam.far_clip, cam.near_clip)
        s += pos_h
        s += struct.pack("<3f", *cam.position_base)
        s += tgt_h
        s += struct.pack("<3f", *cam.target_base)
        s += roll_h
        if fov_is_track:
            s += fov_h
        structs += s
    return len(model.cameras), buf.append(bytes(structs))


def _write_track(buf, track, nseq, kind):
    if track is None or (not track.timelines and track.flat is None):
        return (0, 0xFFFF, (0, 0), (0, 0))
    interp = track.interpolation_type & 0xFFFF
    gseq = track.global_sequence if track.global_sequence >= 0 else 0xFFFF
    if track.flat is not None:
        timelines = [list(zip(track.flat[0], track.flat[1]))]
    else:
        timelines = list(track.timelines)
    # Global-sequence tracks live in a SINGLE outer timeline that loops
    if gseq != 0xFFFF:
        # Keep just the first non-empty timeline; that's where import stored
        # the loop's keys (parser puts everything in timelines[0]).
        keys = next((tl for tl in timelines if tl), [])
        outer_len = 1
        timelines_out = [keys]
    else:
        outer_len = nseq
        timelines_out = timelines[:nseq]
        while len(timelines_out) < nseq:
            timelines_out.append([])
    inner_ts, inner_val = [], []
    for keys in timelines_out:
        if keys:
            t_off = buf.append(struct.pack(
                "<%dI" % len(keys), *(int(round(t)) & 0xFFFFFFFF for t, _ in keys)))
            inner_ts.append((len(keys), t_off))
            v_off = buf.append(b"".join(_pack_track_value(v, kind) for _, v in keys))
            inner_val.append((len(keys), v_off))
        else:
            inner_ts.append((0, 0))
            inner_val.append((0, 0))
    ts_outer = buf.append(b"".join(struct.pack("<II", c, o) for c, o in inner_ts))
    val_outer = buf.append(b"".join(struct.pack("<II", c, o) for c, o in inner_val))
    return (interp, gseq, (outer_len, ts_outer), (outer_len, val_outer))


def _pack_bone(b, tracks):
    out = struct.pack("<iIhHI", b.key_bone_id, b.flags & 0xFFFFFFFF, b.parent,
                      getattr(b, "submesh_id", 0) & 0xFFFF,
                      getattr(b, "name_crc", 0) & 0xFFFFFFFF)
    for interp, gseq, (tc, to), (vc, vo) in tracks:
        out += struct.pack("<HHIIII", interp, gseq, tc, to, vc, vo)
    out += struct.pack("<3f", b.pivot[0], b.pivot[1], b.pivot[2])
    return out


DEFAULT_EVENTS = [
    ("$SHL", [(89, 300), (90, 200)], 2),   # sheathe/draw, left hand
    ("$SHR", [(89, 300), (90, 200)], 1),   # sheathe/draw, right hand
]


def ensure_events(model):
    """Synthesise the essential events when a model has none."""
    if getattr(model, "events", None):
        return
    if not model.sequences:
        return

    by_anim = {}
    for i, s in enumerate(model.sequences):
        by_anim.setdefault(s.id, []).append(i)
    att_bone = {a.id: (a.bone, a.position) for a in model.attachments}

    made = []
    for ident, fires, att_id in DEFAULT_EVENTS:
        hits = [(aid, t) for aid, t in fires if aid in by_anim]
        if not hits:
            continue                      # model lacks the sheathe animations
        bone, pos = att_bone.get(att_id, (0, (0.0, 0.0, 0.0)))
        e = M2Event()
        e.identifier = ident
        e.data = 0
        e.bone = bone
        e.position = tuple(pos)
        e.timestamps = [[] for _ in model.sequences]
        for aid, t in hits:
            for si in by_anim[aid]:       # every variation of that animation
                e.timestamps[si] = [int(t)]
        made.append(e)

    if made:
        model.events = made
        detail = ", ".join("%s(bone %d)" % (e.identifier, e.bone) for e in made)
        print("[M2] model had no events - added defaults: %s" % detail, flush=True)
    else:
        print("[M2] no events, and no Sheath/HipSheath animation to attach them "
              "to - weapons will not draw", flush=True)


def _write_events(buf, model, nseq):
    """Lay out the M2Event array; return (count, offset)."""
    events = getattr(model, "events", None) or []
    if not events:
        return 0, 0

    # Inner timestamp arrays first, so the outer array can point at them.
    inner = []
    for e in events:
        rows = []
        times = e.timestamps or []
        for si in range(nseq):
            ts = times[si] if si < len(times) else []
            if ts:
                off = buf.append(struct.pack("<%dI" % len(ts), *(int(t) & 0xFFFFFFFF for t in ts)))
                rows.append((len(ts), off))
            else:
                rows.append((0, 0))
        inner.append(rows)

    outer = []
    for rows in inner:
        outer.append(buf.append(b"".join(struct.pack("<II", c, o) for c, o in rows)))

    blob = bytearray()
    for e, outer_off in zip(events, outer):
        ident = (e.identifier or "    ").encode("ascii", "replace")[:4].ljust(4, b" ")
        blob += ident
        blob += struct.pack("<II", e.data & 0xFFFFFFFF, e.bone & 0xFFFFFFFF)
        blob += struct.pack("<3f", *e.position)
        blob += struct.pack("<HH", 0, 0xFFFF)          # interp, global_sequence
        blob += struct.pack("<II", nseq, outer_off)    # timestamps M2Array
    return len(events), buf.append(bytes(blob))


def scale_model(model, k):
    """Uniformly resize a finished model by ``k``, keeping animation intact."""
    if not k or abs(k - 1.0) < 1e-9:
        return

    def s3(v):
        return (v[0] * k, v[1] * k, v[2] * k)

    def scale_track(track):
        """Scale a vec3 translation track in place (modern and legacy layouts)."""
        if track is None:
            return
        if track.timelines:
            track.timelines = [[(t, s3(v)) for t, v in tl] for tl in track.timelines]
        if track.flat is not None:
            times, values = track.flat
            track.flat = (times, [s3(v) for v in values])

    for v in model.vertices:
        v.pos = s3(v.pos)
    for b in model.bones:
        b.pivot = s3(b.pivot)
        scale_track(b.translation)          # rotation/scale deliberately untouched
    for a in model.attachments:
        a.position = s3(a.position)
    for c in getattr(model, "cameras", []) or []:
        c.position_base = s3(c.position_base)
        c.target_base = s3(c.target_base)
        c.near_clip *= k
        c.far_clip *= k
        scale_track(c.position)
        scale_track(c.target)
    ov = getattr(model, "bounding_override", None)
    if ov:
        model.bounding_override = (s3(ov[0]), s3(ov[1]))
    print("[M2] scaled model by %.4f" % k, flush=True)


def _model_bounds(model):
    """Bounds to write: an explicit override (edited box) wins, else vertices."""
    ov = getattr(model, "bounding_override", None)
    if ov:
        mn, mx = ov
        cx, cy, cz = ((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2)
        radius = (((mx[0] - cx) ** 2 + (mx[1] - cy) ** 2
                   + (mx[2] - cz) ** 2) ** 0.5)
        return tuple(mn), tuple(mx), radius
    return _bounds(model.vertices)


def _bounds(vertices):
    if not vertices:
        return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1.0
    xs = [v.pos[0] for v in vertices]
    ys = [v.pos[1] for v in vertices]
    zs = [v.pos[2] for v in vertices]
    mn = (min(xs), min(ys), min(zs))
    mx = (max(xs), max(ys), max(zs))
    cx, cy, cz = ((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2)
    radius = max(((v.pos[0] - cx) ** 2 + (v.pos[1] - cy) ** 2
                  + (v.pos[2] - cz) ** 2) ** 0.5 for v in vertices)
    return mn, mx, radius


# ---------------------------------------------------------------------------
def write_m2(model, version=264, track_nseq=None):
    """Return the MD20 data block (also the payload of an MD21 chunk)."""
    link_animation_variations(model)
    ensure_events(model)
    buf = _Buf(_M2_HEADER_SIZE)
    seqs = model.sequences
    nseq = len(seqs)
    rnseq = track_nseq if track_nseq is not None else nseq
    bounds = _model_bounds(model)

    name = (model.name or "").encode("utf-8") + b"\x00"
    name_off = buf.append(name)
    gl = model.global_loops
    gl_off = buf.append(struct.pack("<%dI" % len(gl), *gl)) if gl else 0
    seq_off = buf.append(b"".join(_pack_sequence(s, bounds) for s in seqs)) if seqs else 0
    sl = model.seq_lookup
    sl_off = buf.append(struct.pack("<%dH" % len(sl), *(v & 0xFFFF for v in sl))) \
        if sl else 0

    track_info = []
    for b in model.bones:
        track_info.append((
            _write_track(buf, b.translation, rnseq, "vec3"),
            _write_track(buf, b.rotation, rnseq, "compquat"),
            _write_track(buf, b.scale, rnseq, "vec3"),
        ))
    bones_off = buf.append(b"".join(
        _pack_bone(b, tr) for b, tr in zip(model.bones, track_info))) if model.bones else 0

    kbl = model.key_bone_lookup
    kbl_off = buf.append(struct.pack("<%dH" % len(kbl), *kbl)) if kbl else 0

    vert_off = buf.append(b"".join(_pack_vertex(v) for v in model.vertices)) \
        if model.vertices else 0

    tex_structs = bytearray()
    for t in model.textures:
        fn = (t.filename or "").encode("utf-8") + b"\x00"
        f_off = buf.append(fn)
        tex_structs += struct.pack("<IIII", t.type & 0xFFFFFFFF,
                                   t.flags & 0xFFFFFFFF, len(fn), f_off)
    tex_off = buf.append(bytes(tex_structs)) if model.textures else 0

    mat_off = buf.append(b"".join(
        struct.pack("<HH", m.flags & 0xFFFF, m.blend_mode & 0xFFFF)
        for m in model.materials)) if model.materials else 0

    # Render arrays the batches index into: emitted with their real animated
    colors_count, colors_off = _render_array(
        buf, model.colors, ("vec3", "fixed16"), rnseq, model.n_colors)
    tw_count, tw_off = _render_array(
        buf, model.weights, ("fixed16",), rnseq, model.n_texture_weights)
    tt_count, tt_off = _render_array(
        buf, model.transforms, ("vec3", "quat4f", "vec3"), rnseq,
        model.n_texture_transforms)
    tii = model.texture_indices_by_id
    tii_off = buf.append(struct.pack("<%dH" % len(tii), *tii)) if tii else 0
    cc = model.tex_coord_combos
    cc_off = buf.append(struct.pack("<%dH" % len(cc), *cc)) if cc else 0
    wc = model.tex_weight_combos
    wc_off = buf.append(struct.pack("<%dH" % len(wc), *wc)) if wc else 0
    tc = model.tex_transform_combos
    tc_off = buf.append(struct.pack("<%dH" % len(tc), *tc)) if tc else 0

    bc = model.bone_lookup
    bc_off = buf.append(struct.pack("<%dH" % len(bc), *bc)) if bc else 0
    tl = model.texture_lookup
    tl_off = buf.append(struct.pack("<%dH" % len(tl), *tl)) if tl else 0

    att_bytes = bytearray()
    for a in model.attachments:
        att_bytes += struct.pack("<IHH3f", a.id & 0xFFFFFFFF, a.bone & 0xFFFF, 0,
                                 a.position[0], a.position[1], a.position[2])
        att_bytes += struct.pack("<HHIIII", 0, 0xFFFF, 0, 0, 0, 0)
    att_off = buf.append(bytes(att_bytes)) if model.attachments else 0
    ev_count, ev_off = _write_events(buf, model, rnseq)
    al = model.attachment_lookup
    al_off = buf.append(struct.pack("<%dH" % len(al), *(v & 0xFFFF for v in al))) \
        if al else 0

    cam_count, cam_off = _write_cameras(buf, model, version)
    cl = model.camera_lookup or ([0] if model.cameras else [])
    cl_off = buf.append(struct.pack("<%dH" % len(cl), *(v & 0xFFFF for v in cl))) \
        if cl else 0

    bbmin, bbmax, radius = bounds
    h = bytearray()
    h += b"MD20"
    h += struct.pack("<I", version)
    h += struct.pack("<II", len(name), name_off)
    h += struct.pack("<I", int(getattr(model, "global_flags", 0)) & 0xFFFFFFFF)
    h += struct.pack("<II", len(gl), gl_off)
    h += struct.pack("<II", nseq, seq_off)
    h += struct.pack("<II", len(sl), sl_off)            # sequenceIdxHashById
    h += struct.pack("<II", len(model.bones), bones_off)
    h += struct.pack("<II", len(kbl), kbl_off)          # boneIndicesById
    h += struct.pack("<II", len(model.vertices), vert_off)
    h += struct.pack("<I", 1)                            # num_skin_profiles
    h += struct.pack("<II", colors_count, colors_off)
    h += struct.pack("<II", len(model.textures), tex_off)
    h += struct.pack("<II", tw_count, tw_off)
    h += struct.pack("<II", tt_count, tt_off)
    h += struct.pack("<II", len(tii), tii_off)          # textureIndicesById
    h += struct.pack("<II", len(model.materials), mat_off)
    h += struct.pack("<II", len(bc), bc_off)            # boneCombos
    h += struct.pack("<II", len(tl), tl_off)            # textureCombos
    h += struct.pack("<II", len(cc), cc_off)            # textureCoordCombos
    h += struct.pack("<II", len(wc), wc_off)            # textureWeightCombos
    h += struct.pack("<II", len(tc), tc_off)            # textureTransformCombos
    h += struct.pack("<6f", *bbmin, *bbmax)
    h += struct.pack("<f", radius)
    h += struct.pack("<6f", *bbmin, *bbmax)
    h += struct.pack("<f", radius)
    h += struct.pack("<II", 0, 0)                       # collisionIndices
    h += struct.pack("<II", 0, 0)                       # collisionPositions
    h += struct.pack("<II", 0, 0)                       # collisionFaceNormals
    h += struct.pack("<II", len(model.attachments), att_off)
    h += struct.pack("<II", len(al), al_off)            # attachmentIndicesById
    h += struct.pack("<II", ev_count, ev_off)           # events
    h += struct.pack("<II", 0, 0)                       # lights
    h += struct.pack("<II", cam_count, cam_off)         # cameras
    h += struct.pack("<II", len(cl), cl_off)            # cameraIndicesById
    h += struct.pack("<II", 0, 0)                       # ribbon_emitters
    h += struct.pack("<II", 0, 0)                       # particle_emitters
    assert len(h) == _M2_HEADER_SIZE, len(h)
    buf.b[0:_M2_HEADER_SIZE] = h
    return bytes(buf.b)


# ---------------------------------------------------------------------------
def validate_skin(skin, log=None):
    """Report skin sections the client will fail to draw."""
    problems = []
    if skin is None:
        return problems
    n_tris = len(skin.triangles)
    covered = {b.submesh_index for b in skin.batches}
    for i, s in enumerate(skin.submeshes):
        gid = s.skin_section_id
        if s.vertex_count > 0xFFFF:
            problems.append(
                "geoset %d (section %d): %d vertices exceeds the 65535 per-section "
                "limit: split this mesh" % (gid, i, s.vertex_count))
        if s.vertex_start > 0xFFFF:
            problems.append(
                "geoset %d (section %d): vertex_start %d exceeds 65535: the model "
                "has too many vertices before this section" % (gid, i, s.vertex_start))
        if s.index_count > 0xFFFF:
            problems.append(
                "geoset %d (section %d): %d indices (%d triangles) exceeds the 65535 "
                "per-section limit: split this mesh into smaller geosets"
                % (gid, i, s.index_count, s.index_count // 3))
        if s.index_start + s.index_count > n_tris:
            problems.append(
                "geoset %d (section %d): index range %d..%d runs past the %d "
                "triangle indices in the skin"
                % (gid, i, s.index_start, s.index_start + s.index_count, n_tris))
        if s.bone_count > _MAX_PALETTE:
            problems.append(
                "geoset %d (section %d): needs %d bones, over the %d-bone GPU "
                "palette limit: reduce the number of bones influencing it"
                % (gid, i, s.bone_count, _MAX_PALETTE))
        if i not in covered:
            problems.append(
                "geoset %d (section %d): has no batch, so it is never drawn"
                % (gid, i))
        if s.vertex_count and not getattr(s, "sort_radius", 0.0):
            problems.append(
                "geoset %d (section %d): sortRadius is 0, so the client culls it "
                "as a zero-size object and never draws it" % (gid, i))
    if problems and log is not None:
        log("WARNING: %d geoset problem(s) that cause meshes to NOT DISPLAY "
            "in-game:" % len(problems))
        for p in problems:
            log("   " + p)
    return problems


def sanitize_bone_references(model, log=None):
    """Drop or clamp anything pointing at a bone that doesn't exist."""
    fixed = []
    n = len(model.bones)
    if n == 0:
        return fixed

    # Bone parents
    bad_parent = 0
    for b in model.bones:
        if b.parent is not None and b.parent >= n:
            b.parent = -1
            bad_parent += 1
    if bad_parent:
        fixed.append("%d bone parent link(s) pointed past the end" % bad_parent)

    # Attachments
    if model.attachments:
        keep = [a for a in model.attachments if 0 <= a.bone < n]
        dropped = len(model.attachments) - len(keep)
        if dropped:
            model.attachments = keep
            lookup = {}
            for i, a in enumerate(keep):
                lookup.setdefault(a.id, i)
            model.attachment_lookup = ([lookup.get(k, 0xFFFF)
                                        for k in range(max(lookup) + 1)]
                                       if lookup else [])
            fixed.append("dropped %d attachment(s) bound to bones >= %d "
                         "(would crash the client on load)" % (dropped, n))

    # Key-bone lookup
    if model.key_bone_lookup:
        bad = sum(1 for k in model.key_bone_lookup if k != 0xFFFF and k >= n)
        if bad:
            model.key_bone_lookup = [k if (k == 0xFFFF or k < n) else 0xFFFF
                                     for k in model.key_bone_lookup]
            fixed.append("blanked %d key-bone lookup entry(ies) past the end" % bad)

    # Per-section centre bone
    if model.skin is not None:
        bad = 0
        for s in model.skin.submeshes:
            if s.center_bone_index >= n:
                s.center_bone_index = 0
                bad += 1
        if bad:
            fixed.append("reset %d section centre-bone index(es)" % bad)

    # Per-vertex bone indices
    bad_v = 0
    for v in model.vertices:
        if any(bi >= n for bi in v.bone_indices):
            v.bone_indices = tuple(bi if bi < n else 0 for bi in v.bone_indices)
            bad_v += 1
    if bad_v:
        fixed.append("clamped bone indices on %d vertex(es)" % bad_v)

    if fixed and log is not None:
        log("repaired %d bone-reference problem(s) that would crash the client:"
            % len(fixed))
        for f in fixed:
            log("   " + f)
    return fixed


def build_section_bounds(model):
    """Fill in each skin section's centre and sort radius."""
    skin = model.skin
    if skin is None:
        return
    verts = model.vertices
    sv = skin.vertices
    for s in skin.submeshes:
        start = s.vertex_start
        end = min(start + s.vertex_count, len(sv))
        pts = []
        for i in range(start, end):
            gi = sv[i]
            if 0 <= gi < len(verts):
                pts.append(verts[gi].pos)
        if not pts:
            s.center = s.sort_center = (0.0, 0.0, 0.0)
            s.sort_radius = 0.0
            continue
        n = float(len(pts))
        s.center = (sum(p[0] for p in pts) / n,
                    sum(p[1] for p in pts) / n,
                    sum(p[2] for p in pts) / n)
        lo = [min(p[k] for p in pts) for k in range(3)]
        hi = [max(p[k] for p in pts) for k in range(3)]
        s.sort_center = tuple((lo[k] + hi[k]) * 0.5 for k in range(3))
        s.sort_radius = max(
            ((p[0] - s.sort_center[0]) ** 2 +
             (p[1] - s.sort_center[1]) ** 2 +
             (p[2] - s.sort_center[2]) ** 2) ** 0.5 for p in pts)


def _pack_skin_section(s):
    cx, cy, cz = getattr(s, "center", (0.0, 0.0, 0.0))
    sx, sy, sz = getattr(s, "sort_center", (0.0, 0.0, 0.0))
    return struct.pack(
        "<10H3f3ff",
        s.skin_section_id & 0xFFFF, (s.index_start >> 16) & 0xFFFF,
        s.vertex_start & 0xFFFF, s.vertex_count & 0xFFFF,
        s.index_start & 0xFFFF, s.index_count & 0xFFFF,
        s.bone_count & 0xFFFF, s.bone_combo_index & 0xFFFF,
        s.bone_influences & 0xFFFF, s.center_bone_index & 0xFFFF,
        cx, cy, cz, sx, sy, sz, float(getattr(s, "sort_radius", 0.0)),
    )


def _pack_batch(b):
    return struct.pack(
        "<BbHHHhHHHHHHH",
        b.flags & 0xFF, 0, b.shader_id & 0xFFFF,
        b.submesh_index & 0xFFFF, getattr(b, "geoset_index", 0) & 0xFFFF,
        b.color_index, b.material_index & 0xFFFF, b.material_layer & 0xFFFF,
        b.texture_count & 0xFFFF, b.texture_combo_index & 0xFFFF,
        b.texture_coord_combo & 0xFFFF, b.texture_weight_combo & 0xFFFF,
        b.texture_transform_combo & 0xFFFF,
    )


def write_skin(skin, modern=False):
    """Serialize a skin profile (``modern`` adds the SKIN header fields for Legion+)."""
    reserve = 64 if modern else 44
    buf = _Buf(reserve)
    nverts = len(skin.vertices)
    v_off = buf.append(struct.pack("<%dH" % nverts, *skin.vertices)) if nverts else 0
    t_off = buf.append(struct.pack("<%dH" % len(skin.triangles), *skin.triangles)) \
        if skin.triangles else 0
    # per-vertex palette-local bone indices (4 ubytes each)
    if skin.bone_indices and len(skin.bone_indices) == nverts:
        bone_bytes = b"".join(struct.pack("<4B", *(i & 0xFF for i in q))
                              for q in skin.bone_indices)
    else:
        bone_bytes = b"\x00\x00\x00\x00" * nverts
    b_off = buf.append(bone_bytes) if nverts else 0
    sub_off = buf.append(b"".join(_pack_skin_section(s) for s in skin.submeshes)) \
        if skin.submeshes else 0
    bat_off = buf.append(b"".join(_pack_batch(b) for b in skin.batches)) \
        if skin.batches else 0

    h = bytearray()
    h += struct.pack("<II", nverts, v_off)
    h += struct.pack("<II", len(skin.triangles), t_off)
    h += struct.pack("<II", nverts, b_off)             # bones (per-vertex)
    h += struct.pack("<II", len(skin.submeshes), sub_off)
    h += struct.pack("<II", len(skin.batches), bat_off)
    h += struct.pack("<I", max((s.bone_count for s in skin.submeshes), default=0))
    if modern:
        h += struct.pack("<II", 0, 0)                  # shadow_batches (empty)
        h += struct.pack("<II", 0, 0)                  # trailing array (empty)
    base = 0
    if modern:
        buf.b[0:4] = b"SKIN"
        base = 4
    buf.b[base:base + len(h)] = h
    return bytes(buf.b)


# ---------------------------------------------------------------------------
def _chunk(name, data):
    return name + struct.pack("<I", len(data)) + data


def write_m2_chunked(model, version=274, skin_file_ids=None, texture_file_ids=None):
    """Return a chunked MD21 ``.m2`` plus the modern ``.skin`` bytes."""
    skel_based = "SKID" in (model.aux_chunks or {})
    orig_nseq = len(model.sequences)
    saved = None
    if skel_based:
        saved = (model.bones, model.sequences, model.seq_lookup,
                 model.key_bone_lookup, model.attachments,
                 model.attachment_lookup, model.global_loops)
        model.bones = []
        model.sequences = []
        model.seq_lookup = []
        model.key_bone_lookup = []
        model.attachments = []
        model.attachment_lookup = []
        model.global_loops = []
    try:
        build_render_tables(model)
        block = write_m2(model, version=version,
                         track_nseq=(orig_nseq if skel_based else None))
    finally:
        if saved is not None:
            (model.bones, model.sequences, model.seq_lookup,
             model.key_bone_lookup, model.attachments,
             model.attachment_lookup, model.global_loops) = saved

    out = bytearray()
    out += _chunk(b"MD21", block)

    if skin_file_ids:
        out += _chunk(b"SFID", struct.pack("<%dI" % len(skin_file_ids),
                                           *skin_file_ids))
    if texture_file_ids is None:
        texture_file_ids = [t.file_data_id for t in model.textures]
    if texture_file_ids:
        out += _chunk(b"TXID", struct.pack("<%dI" % len(texture_file_ids),
                                           *(int(x) & 0xFFFFFFFF for x in texture_file_ids)))

    # We always regenerate MD21/SFID/TXID. For a skeleton-based model we KEEP
    skip = {"MD21", "SFID", "TXID"}
    if not skel_based:
        skip |= {"SKID", "AFID", "BFID"}
    for name, raw in (model.aux_chunks or {}).items():
        if name in skip:
            continue
        out += _chunk(name.encode("ascii")[:4].ljust(4, b" "), raw)

    return bytes(out), write_skin(model.skin, modern=True) if model.skin else None


def write_custom_character(model, skel_file_id, skin_file_ids,
                           texture_file_ids=None, version=274):
    """Mesh-only, single-LOD MD21 that animates from an external ``.skel``."""
    build_render_tables(model)           # boneCombos + skin + centers (+ key_bone_lookup)
    orig_nseq = len(model.sequences)
    saved = (model.bones, model.sequences, model.seq_lookup,
             model.key_bone_lookup, model.attachments, model.attachment_lookup,
             model.global_loops)
    model.bones = []
    model.sequences = []
    model.seq_lookup = []
    model.key_bone_lookup = []
    model.attachments = []
    model.attachment_lookup = []
    model.global_loops = []
    try:
        # 0 sequences in the header (they live in the .skel), but the render
        block = write_m2(model, version=version, track_nseq=orig_nseq)
    finally:
        (model.bones, model.sequences, model.seq_lookup, model.key_bone_lookup,
         model.attachments, model.attachment_lookup, model.global_loops) = saved

    out = bytearray()
    out += _chunk(b"MD21", block)
    if skin_file_ids:
        out += _chunk(b"SFID", struct.pack("<%dI" % len(skin_file_ids),
                                           *(int(x) & 0xFFFFFFFF for x in skin_file_ids)))
    if texture_file_ids is None:
        texture_file_ids = [t.file_data_id for t in model.textures]
    if texture_file_ids:
        out += _chunk(b"TXID", struct.pack("<%dI" % len(texture_file_ids),
                                           *(int(x) & 0xFFFFFFFF for x in texture_file_ids)))
    if skel_file_id:
        out += _chunk(b"SKID", struct.pack("<I", int(skel_file_id) & 0xFFFFFFFF))
    skin_bytes = write_skin(model.skin, modern=True) if model.skin else None
    return bytes(out), skin_bytes


# ---------------------------------------------------------------------------
_OFF_VERTICES = 0x3C
_OFF_BONECOMBOS = 0x78


def sfid_count(original_bytes):
    """Number of skin LODs declared by an M2's SFID chunk (1 if none)."""
    d = original_bytes
    pos = 0
    while pos + 8 <= len(d):
        name = d[pos:pos + 4]
        size = int.from_bytes(d[pos + 4:pos + 8], "little")
        if name == b"SFID":
            return max(1, size // 4)
        pos = pos + 8 + size
    return 1


def _skel_chunk(name, header_len, build):
    """Build one .skel chunk; ``build(buf)`` appends data and returns the fixed"""
    buf = _Buf(header_len)
    header = build(buf)
    assert len(header) == header_len, (name, len(header))
    buf.b[0:header_len] = header
    return name + struct.pack("<I", len(buf.b)) + bytes(buf.b)


def write_skel(model):
    """Serialize a ``.skel`` carrying the skeleton + animations (Legion+ form)."""
    sanitize_bone_references(model, log=lambda m: print("[M2] skel: " + m, flush=True))
    link_animation_variations(model)
    ensure_events(model)
    nseq = len(model.sequences)
    bounds = _model_bounds(model) if (model.vertices or getattr(model, "bounding_override", None)) else ((0, 0, 0), (0, 0, 0), 0.0)

    def b_skl1(buf):
        nm = (model.name or "").encode("utf-8") + b"\x00"
        off = buf.append(nm)
        return struct.pack("<I", 0x100) + struct.pack("<II", len(nm), off) + struct.pack("<I", 0)

    def b_sks1(buf):
        gl = model.global_loops
        gl_off = buf.append(struct.pack("<%dI" % len(gl), *gl)) if gl else 0
        seq_off = buf.append(b"".join(_pack_sequence(s, bounds) for s in model.sequences)) \
            if model.sequences else 0
        sl = model.seq_lookup
        sl_off = buf.append(struct.pack("<%dH" % len(sl), *(v & 0xFFFF for v in sl))) if sl else 0
        h = struct.pack("<II", len(gl), gl_off)
        h += struct.pack("<II", len(model.sequences), seq_off)
        h += struct.pack("<II", len(sl), sl_off)
        h += struct.pack("<II", 0, 0)            # _0x18[8]
        return h

    def b_skb1(buf):
        bones = model.bones
        track_info = []
        for b in bones:
            track_info.append((
                _write_track(buf, b.translation, nseq, "vec3"),
                _write_track(buf, b.rotation, nseq, "compquat"),
                _write_track(buf, b.scale, nseq, "vec3"),
            ))
        bones_off = buf.append(b"".join(
            _pack_bone(b, tr) for b, tr in zip(bones, track_info))) if bones else 0
        kbl = model.key_bone_lookup
        kbl_off = buf.append(struct.pack("<%dH" % len(kbl), *kbl)) if kbl else 0
        return struct.pack("<II", len(bones), bones_off) + struct.pack("<II", len(kbl), kbl_off)

    def b_ska1(buf):
        att = model.attachments
        ab = bytearray()
        for a in att:
            ab += struct.pack("<IHH3f", a.id & 0xFFFFFFFF, a.bone & 0xFFFF, 0,
                              a.position[0], a.position[1], a.position[2])
            ab += struct.pack("<HHIIII", 0, 0xFFFF, 0, 0, 0, 0)   # empty animate track
        att_off = buf.append(bytes(ab)) if att else 0
        al = model.attachment_lookup
        al_off = buf.append(struct.pack("<%dH" % len(al), *(v & 0xFFFF for v in al))) if al else 0
        return struct.pack("<II", len(att), att_off) + struct.pack("<II", len(al), al_off)

    return (_skel_chunk(b"SKL1", 16, b_skl1)
            + _skel_chunk(b"SKS1", 32, b_sks1)
            + _skel_chunk(b"SKB1", 16, b_skb1)
            + _skel_chunk(b"SKA1", 16, b_ska1))


def hardcode_skin_texture(original_bytes, blp_id):
    """Rewrite every runtime-composited skin texture (type 1) to a hardcoded type-0 texture pointing at ``blp_id``."""
    buf = bytearray(original_bytes)
    md = 0
    if buf[:4] != b"MD20":
        pos = 0
        while pos + 8 <= len(buf):
            if buf[pos:pos + 4] == b"MD21":
                md = pos + 8
                break
            pos = pos + 8 + int.from_bytes(buf[pos + 4:pos + 8], "little")
    tcount, toff = struct.unpack_from("<II", buf, md + 0x50)   # textures M2Array
    abs_toff = md + toff
    changed = []
    for i in range(tcount):
        tb = abs_toff + i * 16                                  # M2Texture = 16 bytes
        if struct.unpack_from("<I", buf, tb)[0] == 1:          # type == 1 (skin)
            struct.pack_into("<I", buf, tb, 0)                 # -> type 0 (hardcoded)
            changed.append(i)
    if changed:
        pos = 0
        while pos + 8 <= len(buf):
            name = buf[pos:pos + 4]
            size = int.from_bytes(buf[pos + 4:pos + 8], "little")
            if name == b"TXID":
                base = pos + 8
                n = size // 4
                for i in changed:
                    if i < n:
                        struct.pack_into("<I", buf, base + i * 4, int(blp_id) & 0xFFFFFFFF)
                break
            pos = pos + 8 + size
    return bytes(buf), changed


def vertex_count(original_bytes):
    """The M2's vertex count (from the header's vertices M2Array)."""
    buf = original_bytes
    md21_off = 0
    if buf[:4] != b"MD20":
        pos = 0
        while pos + 8 <= len(buf):
            name = buf[pos:pos + 4]
            size = int.from_bytes(buf[pos + 4:pos + 8], "little")
            if name == b"MD21":
                md21_off = pos + 8
                break
            pos = pos + 8 + size
    return struct.unpack_from("<I", buf, md21_off + _OFF_VERTICES)[0]


def patch_m2_vertices(original_bytes, updates):
    """Overwrite only the vertex pos/normal/uv of an existing M2, in place."""
    buf = bytearray(original_bytes)
    chunked = buf[:4] != b"MD20"
    md21_off = 0
    if chunked:
        pos = 0
        while pos + 8 <= len(buf):
            name = buf[pos:pos + 4]
            size = int.from_bytes(buf[pos + 4:pos + 8], "little")
            if name == b"MD21":
                md21_off = pos + 8
                break
            pos = pos + 8 + size
        else:
            raise ValueError("patch_m2_vertices: no MD21 chunk")
    vcount, voff = struct.unpack_from("<II", buf, md21_off + _OFF_VERTICES)
    abs_voff = md21_off + voff
    applied = 0
    for gidx, (pos3, nrm3, uv2) in updates.items():
        if not (0 <= gidx < vcount):
            continue
        vb = abs_voff + gidx * 48          # M2Vertex stride
        struct.pack_into("<3f", buf, vb + 0, *pos3)        # position
        struct.pack_into("<3f", buf, vb + 20, *nrm3)       # normal
        struct.pack_into("<2f", buf, vb + 32, *uv2)        # uv1
        applied += 1
    return bytes(buf), applied, vcount


def patch_m2(original_bytes, model, skin_file_ids=None, skid=None, drop=()):
    """Splice ``model``'s geometry into ``original_bytes``, keeping everything"""
    drop = {d.encode("ascii") if isinstance(d, str) else d for d in drop}
    orig = original_bytes
    chunked = orig[:4] != b"MD20"
    md21_off, md21_size = 0, len(orig)
    if chunked:
        pos = 0
        while pos + 8 <= len(orig):
            name = orig[pos:pos + 4]
            size = int.from_bytes(orig[pos + 4:pos + 8], "little")
            if name == b"MD21":
                md21_off, md21_size = pos + 8, size
                break
            pos = pos + 8 + size
        else:
            raise ValueError("patch_m2: no MD21 chunk in base M2")

    block = bytearray(orig[md21_off:md21_off + md21_size])

    def align4():
        while len(block) % 4:
            block.append(0)

    align4()
    vert_off = len(block)
    block += b"".join(_pack_vertex(v) for v in model.vertices)
    struct.pack_into("<II", block, _OFF_VERTICES, len(model.vertices), vert_off)

    build_render_tables(model)
    align4()
    bc_off = len(block)
    if model.bone_lookup:
        block += struct.pack("<%dH" % len(model.bone_lookup), *model.bone_lookup)
    struct.pack_into("<II", block, _OFF_BONECOMBOS, len(model.bone_lookup), bc_off)

    skin_bytes = write_skin(model.skin, modern=True) if model.skin else None
    if not chunked:
        return bytes(block), skin_bytes

    out = bytearray()
    seen_skid = False
    pos = 0
    while pos + 8 <= len(orig):
        name = orig[pos:pos + 4]
        size = int.from_bytes(orig[pos + 4:pos + 8], "little")
        end = pos + 8 + size
        if name == b"MD21":
            out += b"MD21" + struct.pack("<I", len(block)) + block
        elif name == b"SFID" and skin_file_ids:
            sf = struct.pack("<%dI" % len(skin_file_ids),
                             *(int(x) & 0xFFFFFFFF for x in skin_file_ids))
            out += b"SFID" + struct.pack("<I", len(sf)) + sf
        elif name == b"SKID" and skid is not None:
            out += b"SKID" + struct.pack("<II", 4, int(skid) & 0xFFFFFFFF)
            seen_skid = True
        elif name in drop:
            pass                      # remove this chunk
        else:
            out += orig[pos:end]
        pos = end
    if skid is not None and not seen_skid:
        out += b"SKID" + struct.pack("<II", 4, int(skid) & 0xFFFFFFFF)
    return bytes(out), skin_bytes


# ---------------------------------------------------------------------------
def skin_lod_path(base, lod):
    """Game skin-file naming: LOD 0 is ``<base>00.skin``; LODs 1+ are"""
    if lod == 0:
        return "%s00.skin" % base
    return "%s_lod%02d.skin" % (base, lod)


def write_skin_lods(base, skin_bytes, lod_count):
    """Write ``lod_count`` identical skin files under the game's LOD names."""
    paths = []
    for lod in range(max(1, lod_count)):
        p = skin_lod_path(base, lod)
        with open(p, "wb") as f:
            f.write(skin_bytes)
        paths.append(p)
    return paths


def write_model(model, m2_path, chunked=False, version=None,
                skin_file_ids=None, texture_file_ids=None, lod_count=1):
    """Write ``model`` to ``m2_path`` (+ its ``.skin`` LOD files)."""
    base = m2_path[:-3] if m2_path.lower().endswith(".m2") else m2_path
    if chunked:
        ver = version or 274
        m2_bytes, skin_bytes = write_m2_chunked(
            model, version=ver, skin_file_ids=skin_file_ids,
            texture_file_ids=texture_file_ids)
    else:
        ver = version or 264
        build_render_tables(model)
        m2_bytes = write_m2(model, version=ver)
        skin_bytes = write_skin(model.skin, modern=False) if model.skin else None

    with open(m2_path, "wb") as f:
        f.write(m2_bytes)
    if skin_bytes is not None:
        write_skin_lods(base, skin_bytes, lod_count)
    return m2_path
