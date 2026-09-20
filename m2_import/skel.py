"""Parsing of external ``.skel`` skeleton files (Legion+)."""

import os

from .reader import BinaryReader, scan_chunks
from .model import M2Bone, M2Sequence, M2Attachment, repair_zero_blend_times
from . import anim_data

# Modern struct sizes (the .skel format is Legion+, always modern).
_SEQUENCE_SIZE = 64
_BONE_TRACK_SIZE = 20


def find_skel_file(m2_path):
    """Return ``<model>.skel`` next to the ``.m2`` if it exists, else None."""
    base = m2_path[:-3] if m2_path.lower().endswith(".m2") else m2_path
    candidate = base + ".skel"
    return candidate if os.path.isfile(candidate) else None


def _chunk_reader(data, chunks, name):
    """A reader whose base is the named chunk's data start, or None."""
    if name not in chunks:
        return None
    off, _size = chunks[name]
    r = BinaryReader(data, base=off)
    r.seek(off)                  # header arrays sit at the chunk data start
    return r


def _read_sequences(reader, arr, data_len):
    if arr.count == 0:
        return []
    if arr.base_offset(reader) + arr.count * _SEQUENCE_SIZE > data_len:
        print("[M2] warning: .skel sequence table overruns file.")
        return []

    def one(r):
        start = r.tell()
        s = M2Sequence()
        s.id = r.u16()
        s.variation_index = r.u16()
        s.duration = r.u32()
        s.end_timestamp = s.duration
        s.movespeed = r.f32()
        s.flags = r.u32()
        s.frequency = r.i16()
        r.u16()                  # padding
        r.u32(); r.u32()         # replay range
        s.blend_time_in = r.u16()
        s.blend_time_out = r.u16()
        r.seek(start + _SEQUENCE_SIZE)
        return s

    return repair_zero_blend_times(arr.read(reader, one))


def _read_attachments(reader, arr):
    if arr.count == 0:
        return []

    def one(r):
        a = M2Attachment()
        a.id = r.u32()
        a.bone = r.u16()
        r.u16()                  # unknown
        a.position = r.vec3()
        r.skip(_BONE_TRACK_SIZE)  # animate_attached M2Track header
        return a

    return arr.read(reader, one)


def _read_bones(reader, arr, sequences, loader):
    vec3, quat = anim_data.make_value_readers(False)

    def one(r):
        b = M2Bone()
        b.key_bone_id = r.i32()
        b.flags = r.u32()
        b.parent = r.i16()
        b.submesh_id = r.u16()
        b.name_crc = r.u32()     # boneNameCRC (always present in skel)
        b.translation = anim_data.read_track(r, False, vec3, sequences, loader)
        b.rotation = anim_data.read_track(r, False, quat, sequences, loader)
        b.scale = anim_data.read_track(r, False, vec3, sequences, loader)
        b.pivot = r.vec3()
        return b

    return arr.read(reader, one)


def load_skel(skel_path, model, name_base):
    """Populate ``model`` bones/sequences/attachments/global loops from a skel."""
    with open(skel_path, "rb") as f:
        data = f.read()
    chunks = scan_chunks(data)
    loader = anim_data.AnimLoader(name_base)

    # SKS1 first: bone track decoding needs the sequence flags to tell resident
    # animations from external ones.
    r = _chunk_reader(data, chunks, b"SKS1")
    if r is not None:
        a_global = r.m2array()
        a_seq = r.m2array()
        a_lookup = r.m2array()  # sequence_lookups (animation lookup table)
        model.global_loops = a_global.read_u32(r)
        model.sequences = _read_sequences(r, a_seq, len(data))
        model.seq_lookup = a_lookup.read_u16(r)

    r = _chunk_reader(data, chunks, b"SKB1")
    if r is not None:
        a_bones = r.m2array()
        a_kbl = r.m2array()      # key_bone_lookup - REQUIRED for character bake
        model.bones = _read_bones(r, a_bones, model.sequences, loader)
        # Populate key_bone_lookup from .skel. Without this, the character
        model.key_bone_lookup = a_kbl.read_u16(r)

    r = _chunk_reader(data, chunks, b"SKA1")
    if r is not None:
        a_att = r.m2array()
        a_att_lookup = r.m2array()   # attachment_lookup_table
        model.attachments = _read_attachments(r, a_att)
        model.attachment_lookup = a_att_lookup.read_u16(r)

    print("[M2] skeleton: %d bones, %d sequences, %d attachments"
          % (len(model.bones), len(model.sequences), len(model.attachments)))
