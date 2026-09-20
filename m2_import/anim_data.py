"""Animation track decoding, shared by the in-``.m2`` and ``.skel`` paths."""

import os

from .reader import BinaryReader, decompress_quat, scan_chunks
from .model import M2AnimTrack

# Sequence flag: when set the animation data is stored inside the model.
SEQ_RESIDENT = 0x20

# Chunk magics that wrap bone keyframe data inside a chunked ``.anim`` file.
_ANIM_CHUNK_MAGICS = (b"AFM2", b"AFSA", b"AFSB", b"MAOF")


def make_value_readers(legacy):
    """Return ``(vec3_reader, quat_reader)`` for the given track generation."""
    vec3 = lambda r: r.vec3()
    if legacy:
        quat = lambda r: r.quat()                      # C4Quaternion (x,y,z,w)
    else:
        quat = lambda r: decompress_quat(              # M2CompQuat
            r.i16(), r.i16(), r.i16(), r.i16())
    return vec3, quat


def read_track(reader, legacy, read_value, sequences, anim_loader):
    """Read one M2Track at ``reader``'s cursor into a normalised M2AnimTrack."""
    t = M2AnimTrack()
    t.interpolation_type = reader.u16()
    gseq = reader.u16()
    t.global_sequence = -1 if gseq == 0xFFFF else gseq

    if legacy:
        reader.m2array()                       # interpolation_ranges (unused)
        a_times = reader.m2array()
        a_values = reader.m2array()
        times = a_times.read_u32(reader)
        values = a_values.read(reader, read_value)
        n = min(len(times), len(values))
        t.flat = (times[:n], values[:n])
        return t

    a_times = reader.m2array()                 # M2Array<M2Array<u32>>
    a_values = reader.m2array()                # M2Array<M2Array<T>>
    outer_times = a_times.read_arrays(reader)
    outer_values = a_values.read_arrays(reader)
    timelines = []
    for i in range(min(len(outer_times), len(outer_values))):
        seq = sequences[i] if i < len(sequences) else None
        external = seq is not None and not (seq.flags & SEQ_RESIDENT)
        src = reader
        if external:
            src = anim_loader(seq) if anim_loader is not None else None
        if src is None:
            timelines.append([])               # missing .anim -> no keyframes
            continue
        times = outer_times[i].read_u32(src)
        values = outer_values[i].read(src, read_value)
        n = min(len(times), len(values))
        timelines.append(list(zip(times[:n], values[:n])))
    t.timelines = timelines
    return t


def track_to_json(track):
    """Compact JSON-able form of an M2AnimTrack (for export metadata)."""
    if track is None:
        return None
    d = {"i": track.interpolation_type, "g": track.global_sequence}
    if track.flat is not None:
        times, values = track.flat
        d["f"] = [list(times), [list(v) if isinstance(v, (tuple, list)) else v
                                for v in values]]
    else:
        d["t"] = [[[t, list(v) if isinstance(v, (tuple, list)) else v]
                   for t, v in tl] for tl in track.timelines]
    return d


def track_from_json(d):
    from .model import M2AnimTrack
    if d is None:
        return None
    t = M2AnimTrack()
    t.interpolation_type = d.get("i", 0)
    t.global_sequence = d.get("g", -1)
    if "f" in d:
        times, values = d["f"]
        t.flat = (times, [tuple(v) if isinstance(v, list) else v for v in values])
    else:
        t.timelines = [[(tm, tuple(v) if isinstance(v, list) else v)
                        for tm, v in tl] for tl in d.get("t", [])]
    return t


class AnimLoader:
    """Resolve a non-resident sequence to a reader over its ``.anim`` bone data."""

    def __init__(self, name_base):
        self.name_base = name_base
        self.cache = {}

    def __call__(self, seq):
        key = (seq.id, seq.variation_index)
        if key in self.cache:
            return self.cache[key]
        reader = self._load(seq)
        self.cache[key] = reader
        return reader

    def _load(self, seq):
        path = "%s%04d-%02d.anim" % (self.name_base, seq.id, seq.variation_index)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            return None
        base = 0
        if data[:4] in _ANIM_CHUNK_MAGICS:
            ch = scan_chunks(data)
            if b"AFSB" in ch:        # bone data (skel-based models)
                base = ch[b"AFSB"][0]
            elif b"AFM2" in ch:      # old converted: bones inside AFM2
                base = ch[b"AFM2"][0]
        return BinaryReader(data, base=base)
