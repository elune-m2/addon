"""Skin profile (LOD view) parsing."""

from .reader import BinaryReader
from .model import M2SkinProfile, M2SubMesh, M2Batch


def _read_skin_section(reader: BinaryReader, has_sort: bool) -> M2SubMesh:
    s = M2SubMesh()
    s.skin_section_id = reader.u16()
    s.level = reader.u16()
    s.vertex_start = reader.u16()
    s.vertex_count = reader.u16()
    s.index_start = reader.u16()
    s.index_count = reader.u16()
    s.bone_count = reader.u16()
    s.bone_combo_index = reader.u16()
    s.bone_influences = reader.u16()   # max weighted bones per vertex
    s.center_bone_index = reader.u16()
    # Keep the bounds: sortRadius is the section's culling sphere, so a tool
    # that reads a model back needs the real value, not a default.
    s.center = reader.vec3()
    if has_sort:
        s.sort_center = reader.vec3()
        s.sort_radius = reader.f32()
    # indexStart is only 16-bit; the real start folds in the level field.
    s.index_start = s.index_start + (s.level << 16)
    return s


def _read_batch_modern(reader: BinaryReader) -> M2Batch:
    """Wrath and later: documented M2Batch (24 bytes)."""
    b = M2Batch()
    b.flags = reader.u8()
    reader.i8()                     # priorityPlane
    b.shader_id = reader.u16()
    b.submesh_index = reader.u16()
    b.geoset_index = reader.u16()   # 0 in every retail model; keep it verbatim
    b.color_index = reader.i16()
    b.material_index = reader.u16()
    b.material_layer = reader.u16()
    b.texture_count = reader.u16()
    b.texture_combo_index = reader.u16()
    b.texture_coord_combo = reader.u16()
    b.texture_weight_combo = reader.u16()
    b.texture_transform_combo = reader.u16()
    return b


def _read_batch_old(reader: BinaryReader) -> M2Batch:
    """Classic / TBC texture unit (24 bytes, different field order)."""
    b = M2Batch()
    b.flags = reader.u16()
    b.shader_id = reader.u16()
    b.submesh_index = reader.u16()
    reader.u16()                    # submesh index copy
    b.color_index = reader.i16()
    b.material_index = reader.u16()  # "render flags" index -> materials
    b.material_layer = reader.u16()
    reader.u16()                    # mode
    b.texture_combo_index = reader.u16()
    reader.u16()                    # texture unit number 2
    reader.u16()                    # transparency lookup
    reader.u16()                    # texture anim lookup
    return b


def parse_skin(reader: BinaryReader, has_sort: bool, modern_batch: bool) -> M2SkinProfile:
    """Parse a skin profile header at the reader's current position."""
    # Optional SKIN magic on Cata+ .skin files (a bare 4-byte tag immediately
    # followed by the profile header, not a sized chunk).
    magic = reader.data[reader.tell():reader.tell() + 4]
    if magic == b"SKIN":
        reader.skip(4)
    elif magic[:1].isalpha() and magic not in (b"",):
        # Anything else with a chunk-like ASCII tag (e.g. AFM2 from a mangled
        raise ValueError("unrecognised skin header %r" % magic)

    a_vertices = reader.m2array()
    a_indices = reader.m2array()
    a_bones = reader.m2array()
    a_submeshes = reader.m2array()
    a_batches = reader.m2array()
    # boneCountMax (u32) follows, plus shadow batches on Legion+; not needed.

    # Sanity-check the arrays point inside the buffer before we trust them; a
    # truncated or wrong-format file would otherwise read past the end.
    n = len(reader.data)
    for arr, stride in ((a_vertices, 2), (a_indices, 2),
                        (a_submeshes, 48), (a_batches, 24)):
        if arr.base_offset(reader) + arr.count * stride > n:
            raise ValueError("skin arrays overrun the file (malformed skin)")

    profile = M2SkinProfile()
    profile.vertices = a_vertices.read_u16(reader)
    profile.triangles = a_indices.read_u16(reader)
    profile.submeshes = a_submeshes.read(reader, lambda r: _read_skin_section(r, has_sort))
    batch_reader = _read_batch_modern if modern_batch else _read_batch_old
    profile.batches = a_batches.read(reader, batch_reader)
    return profile


def find_skin_file(m2_path: str):
    """Return the path of the LOD-0 skin next to an .m2, or None."""
    import os
    base = m2_path[:-3] if m2_path.lower().endswith(".m2") else m2_path
    for suffix in ("00.skin", "01.skin", ".skin", "0.skin"):
        candidate = base + suffix
        if os.path.isfile(candidate):
            return candidate
    return None
