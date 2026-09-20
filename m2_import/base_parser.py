"""Shared parsing logic."""

import os

from .reader import BinaryReader
from .model import (
    M2Model, M2Vertex, M2Texture, M2Material, M2Bone,
    M2Sequence, M2Attachment, M2Event, repair_zero_blend_times,
)
from . import skin as skin_mod
from . import anim_data
from . import skel as skel_mod


class BaseM2Parser:
    """Subclasses set these class attributes and implement parse_header()."""

    expansion = ""
    # M2Track byte size for skipping bone animation tracks:
    #   28 for pre-Wrath, 20 for Wrath and later.
    bone_track_size = 20
    # Whether the bone struct carries the 4-byte boneNameCRC union (>= TBC).
    bone_has_crc = True
    # M2SkinSection includes a sort centre + radius from TBC onward.
    skin_has_sort = True
    # Texture-unit (batch) layout: modern (Wrath+) vs old (Classic/TBC).
    skin_modern_batch = True
    # Are skin profiles embedded in the .m2 (Classic/TBC) or external?
    skin_embedded = False
    # M2Camera fov: Cata+ stores it as an animation track at the end of the
    # struct; pre-Cata stores a single static float after ``type``.
    camera_fov_track = True

    # Whether the animation data uses the legacy single-timeline layout
    @property
    def legacy_anim(self):
        return self.bone_track_size == 28

    def __init__(self, data: bytes, base: int = 0):
        self.data = data
        self.base = base
        self.reader = BinaryReader(data, base)
        self.model = M2Model()
        self.model.expansion = self.expansion
        # Header array refs captured during the walk for later parsing.
        self._a_sequences = None
        self._a_global_loops = None
        self._a_seq_lookup = None
        self._a_attachments = None
        self._a_attachment_lookup = None
        self._a_events = None
        self._a_bone_combos = None
        self._a_key_bone_lookup = None
        self._a_colors = None
        self._a_texture_weights = None
        self._a_texture_transforms = None
        self._a_coord_combos = None
        self._a_weight_combos = None
        self._a_transform_combos = None
        self._a_texture_indices_by_id = None
        self._anim_loader = None

    # -- entry point ------------------------------------------------------
    def parse(self, m2_path: str = "") -> M2Model:
        self.model.source_path = m2_path
        self.parse_header()
        # Attachments and (when the rig is external) the .skel are best-effort:
        # a bad field size in an unusual file must never sink geometry import.
        try:
            self.read_attachments()
        except Exception as exc:  # noqa: BLE001
            print("[M2] warning: attachment parse failed: %r" % exc)
        try:
            self._maybe_load_skel(m2_path)
        except Exception as exc:  # noqa: BLE001
            print("[M2] warning: .skel load failed: %r" % exc)
        if self.model.skin is None:
            self._load_external_skin(m2_path)
        return self.model

    def _maybe_load_skel(self, m2_path):
        # Only when the model carries no bones of its own (rig lives in .skel).
        if self.model.bones or not m2_path:
            return
        skel_path = skel_mod.find_skel_file(m2_path)
        if not skel_path:
            return
        name_base = m2_path[:-3] if m2_path.lower().endswith(".m2") else m2_path
        print("[M2] loading external skeleton %s" % os.path.basename(skel_path))
        skel_mod.load_skel(skel_path, self.model, name_base)

    def parse_header(self):
        raise NotImplementedError

    # -- shared record readers -------------------------------------------
    def read_vertices(self, arr):
        def one(r: BinaryReader):
            v = M2Vertex()
            v.pos = r.vec3()
            v.bone_weights = (r.u8(), r.u8(), r.u8(), r.u8())
            v.bone_indices = (r.u8(), r.u8(), r.u8(), r.u8())
            v.normal = r.vec3()
            v.uv1 = r.vec2()
            v.uv2 = r.vec2()
            return v
        self.model.vertices = arr.read(self.reader, one)

    def read_textures(self, arr):
        def one(r: BinaryReader):
            tex_type = r.u32()
            flags = r.u32()
            name_arr = r.m2array()
            filename = name_arr.read_string(r)
            return M2Texture(tex_type, flags, filename)
        self.model.textures = arr.read(self.reader, one)

    def read_materials(self, arr):
        def one(r: BinaryReader):
            return M2Material(r.u16(), r.u16())
        self.model.materials = arr.read(self.reader, one)

    def read_texture_lookup(self, arr):
        self.model.texture_lookup = arr.read_u16(self.reader)

    def read_events(self, arr):
        """Read the M2Event array (36 bytes each)."""
        if arr is None or arr.count == 0:
            return
        r = self.reader

        def one(rd):
            e = M2Event()
            e.identifier = rd.fourcc()
            e.data = rd.u32()
            e.bone = rd.u32()
            e.position = rd.vec3()
            rd.u16()                      # interpolation_type (unused)
            rd.u16()                      # global_sequence
            outer = rd.m2array()          # M2Array<M2Array<uint32>>
            saved = rd.tell()
            times = []
            if outer.count:
                rd.seek(outer.base_offset(rd))
                inner = [rd.m2array() for _ in range(outer.count)]
                for a in inner:
                    times.append(a.read_u32(rd) if a.count else [])
            rd.seek(saved)
            e.timestamps = times
            return e

        self.model.events = arr.read(r, one)

    def read_cameras(self, arr):
        """Read the M2Camera array. Camera tracks store M2SplineKey values"""
        from .model import M2Camera
        legacy = self.legacy_anim
        sequences = self.model.sequences
        loader = self._anim_loader
        fov_is_track = getattr(self, "camera_fov_track", True)

        def spline_vec3(r):
            v = r.vec3(); r.vec3(); r.vec3()   # value, in-tan, out-tan
            return v

        def spline_float(r):
            v = r.f32(); r.f32(); r.f32()
            return v

        def one(r):
            c = M2Camera()
            c.type = r.i32()
            if not fov_is_track:
                c.fov_static = r.f32()          # pre-Cata static fov
            c.far_clip = r.f32()
            c.near_clip = r.f32()
            c.position = anim_data.read_track(r, legacy, spline_vec3, sequences, loader)
            c.position_base = r.vec3()
            c.target = anim_data.read_track(r, legacy, spline_vec3, sequences, loader)
            c.target_base = r.vec3()
            c.roll = anim_data.read_track(r, legacy, spline_float, sequences, loader)
            if fov_is_track:
                c.fov = anim_data.read_track(r, legacy, spline_float, sequences, loader)
            return c

        self.model.cameras = arr.read(self.reader, one)

    def read_bones(self, arr):
        has_crc = self.bone_has_crc
        legacy = self.legacy_anim
        vec3, quat = anim_data.make_value_readers(legacy)
        sequences = self.model.sequences
        loader = self._anim_loader

        def one(r: BinaryReader):
            b = M2Bone()
            b.key_bone_id = r.i32()
            b.flags = r.u32()
            b.parent = r.i16()
            b.submesh_id = r.u16()  # submesh_id / distance-to-parent
            if has_crc:
                b.name_crc = r.u32()  # boneNameCRC union
            b.translation = anim_data.read_track(r, legacy, vec3, sequences, loader)
            b.rotation = anim_data.read_track(r, legacy, quat, sequences, loader)
            b.scale = anim_data.read_track(r, legacy, vec3, sequences, loader)
            b.pivot = r.vec3()
            return b
        self.model.bones = arr.read(self.reader, one)

    # -- header walks -----------------------------------------------------
    def _walk_common_prefix(self, r: BinaryReader):
        r.seek(self.base)
        r.fourcc()                       # magic ("MD20")
        self.model.version = r.u32()
        name_arr = r.m2array()
        self.model.name = name_arr.read_string(r)
        self.model.global_flags = r.u32()
        self._a_global_loops = r.m2array()   # global_loops
        self._a_sequences = r.m2array()      # sequences
        self._a_seq_lookup = r.m2array()     # sequenceIdxHashById (anim lookup)

    def walk_old_header(self):
        """Classic / TBC header layout (embedded skins, extra lookups)."""
        r = self.reader
        self._walk_common_prefix(r)
        r.m2array()                      # playable_animation_lookup (<= TBC)
        bones = r.m2array()
        self._a_key_bone_lookup = r.m2array()   # boneIndicesById
        vertices = r.m2array()
        skin_profiles = r.m2array()      # embedded (<= TBC)
        self._a_colors = r.m2array()     # colors
        textures = r.m2array()
        self._a_texture_weights = r.m2array()
        r.m2array()                      # texture_flipbooks (<= TBC)
        self._a_texture_transforms = r.m2array()
        self._a_texture_indices_by_id = r.m2array()   # textureIndicesById
        materials = r.m2array()
        self._a_bone_combos = r.m2array()   # boneCombos
        texture_lookup = r.m2array()     # textureCombos
        self._walk_tail_to_attachments(r)
        return {
            "bones": bones, "vertices": vertices, "textures": textures,
            "materials": materials, "texture_lookup": texture_lookup,
            "skin_profiles": skin_profiles,
        }

    def walk_modern_header(self):
        """Wrath onward header layout (external skins)."""
        r = self.reader
        self._walk_common_prefix(r)
        bones = r.m2array()
        self._a_key_bone_lookup = r.m2array()   # boneIndicesById
        vertices = r.m2array()
        r.u32()                          # num_skin_profiles
        self._a_colors = r.m2array()     # colors
        textures = r.m2array()
        self._a_texture_weights = r.m2array()
        self._a_texture_transforms = r.m2array()
        self._a_texture_indices_by_id = r.m2array()   # textureIndicesById
        materials = r.m2array()
        self._a_bone_combos = r.m2array()   # boneCombos
        texture_lookup = r.m2array()     # textureCombos
        self._walk_tail_to_attachments(r)
        return {
            "bones": bones, "vertices": vertices, "textures": textures,
            "materials": materials, "texture_lookup": texture_lookup,
        }

    def _walk_tail_to_attachments(self, r: BinaryReader):
        """From textureCombos, walk the shared header tail to ``attachments``."""
        try:
            self._a_coord_combos = r.m2array()      # textureCoordCombos
            self._a_weight_combos = r.m2array()     # textureWeightCombos
            self._a_transform_combos = r.m2array()  # textureTransformCombos
            self.model.bounding_min = (r.f32(), r.f32(), r.f32())  # bounding_box min
            self.model.bounding_max = (r.f32(), r.f32(), r.f32())  # bounding_box max
            self.model.bounding_radius = r.f32()   # bounding_sphere_radius
            r.skip(24)                   # collision_box (CAaBox)
            r.f32()                      # collision_sphere_radius
            r.m2array()                  # collisionIndices
            r.m2array()                  # collisionPositions
            r.m2array()                  # collisionFaceNormals
            self._a_attachments = r.m2array()   # attachments
            self._a_attachment_lookup = r.m2array()  # attachmentIndicesById
        except Exception as exc:  # noqa: BLE001
            print("[M2] warning: header tail walk failed: %r" % exc)
            self._a_attachments = None
            self._a_cameras = None
            return
        # Continue past attachments to the cameras. Separate try/except so a
        # camera-layout surprise never costs the attachments we just read.
        try:
            self._a_events = r.m2array()           # events
            r.m2array()                  # lights
            self._a_cameras = r.m2array()          # cameras
            self._a_camera_lookup = r.m2array()    # cameraIndicesById
        except Exception as exc:  # noqa: BLE001
            print("[M2] warning: camera header walk failed: %r" % exc)
            self._a_cameras = None
            self._a_events = None

    def populate_common(self, arrays):
        """Read the records every era shares, from collected array refs."""
        self.read_vertices(arrays["vertices"])
        self.read_textures(arrays["textures"])
        self.read_materials(arrays["materials"])
        self.read_texture_lookup(arrays["texture_lookup"])
        if self._a_bone_combos is not None:
            self.model.bone_lookup = self._a_bone_combos.read_u16(self.reader)
        if self._a_key_bone_lookup is not None:
            self.model.key_bone_lookup = self._a_key_bone_lookup.read_u16(self.reader)
        # Render-array counts + combo lookups (needed so a retail re-export
        # resolves the batch indices instead of crashing the client).
        if self._a_colors is not None:
            self.model.n_colors = self._a_colors.count
        if self._a_texture_weights is not None:
            self.model.n_texture_weights = self._a_texture_weights.count
        if self._a_texture_transforms is not None:
            self.model.n_texture_transforms = self._a_texture_transforms.count
        if self._a_coord_combos is not None:
            self.model.tex_coord_combos = self._a_coord_combos.read_u16(self.reader)
        if self._a_weight_combos is not None:
            self.model.tex_weight_combos = self._a_weight_combos.read_u16(self.reader)
        if self._a_transform_combos is not None:
            self.model.tex_transform_combos = self._a_transform_combos.read_u16(self.reader)
        if self._a_texture_indices_by_id is not None:
            self.model.texture_indices_by_id = \
                self._a_texture_indices_by_id.read_u16(self.reader)
        if self._a_attachment_lookup is not None:
            self.model.attachment_lookup = self._a_attachment_lookup.read_u16(self.reader)
        if getattr(self, "_a_events", None) is not None:
            try:
                self.read_events(self._a_events)
            except Exception as exc:  # noqa: BLE001
                print("[M2] warning: event read failed: %r" % exc)
        if getattr(self, "_a_cameras", None) is not None:
            try:
                self.read_cameras(self._a_cameras)
                if getattr(self, "_a_camera_lookup", None) is not None:
                    self.model.camera_lookup = \
                        self._a_camera_lookup.read_u16(self.reader)
            except Exception as exc:  # noqa: BLE001
                print("[M2] warning: camera read failed: %r" % exc)
                self.model.cameras = []
        try:
            self.read_global_loops()
            self.read_sequences()
        except Exception as exc:  # noqa: BLE001
            print("[M2] warning: sequence parse failed: %r" % exc)
        path = self.model.source_path
        if path:
            name_base = path[:-3] if path.lower().endswith(".m2") else path
            self._anim_loader = anim_data.AnimLoader(name_base)
        self.read_bones(arrays["bones"])
        try:
            self.read_render_arrays()
        except Exception as exc:  # noqa: BLE001
            print("[M2] warning: colour/weight/transform parse failed: %r" % exc)

    def read_render_arrays(self):
        """Read the colour / texture-weight / texture-transform animation arrays."""
        legacy = self.legacy_anim
        seqs = self.model.sequences
        loader = self._anim_loader
        vec3 = lambda r: r.vec3()
        fixed16 = lambda r: r.i16()
        quat4 = lambda r: r.quat()

        a = self._a_colors
        if a is not None and a.count:
            def one_color(r):
                return (anim_data.read_track(r, legacy, vec3, seqs, loader),
                        anim_data.read_track(r, legacy, fixed16, seqs, loader))
            self.model.colors = a.read(self.reader, one_color)

        a = self._a_texture_weights
        if a is not None and a.count:
            self.model.weights = a.read(
                self.reader,
                lambda r: anim_data.read_track(r, legacy, fixed16, seqs, loader))

        a = self._a_texture_transforms
        if a is not None and a.count:
            def one_xform(r):
                return (anim_data.read_track(r, legacy, vec3, seqs, loader),
                        anim_data.read_track(r, legacy, quat4, seqs, loader),
                        anim_data.read_track(r, legacy, vec3, seqs, loader))
            self.model.transforms = a.read(self.reader, one_xform)

    # -- sequences / global loops / attachments --------------------------
    def read_global_loops(self):
        arr = self._a_global_loops
        if arr is not None:
            self.model.global_loops = arr.read_u32(self.reader)
        if self._a_seq_lookup is not None:
            self.model.seq_lookup = self._a_seq_lookup.read_u16(self.reader)

    def read_sequences(self):
        """Read the animation sequence table (best effort)."""
        arr = self._a_sequences
        if arr is None or arr.count == 0:
            return
        size = 68 if self.legacy_anim else 64
        end = arr.base_offset(self.reader) + arr.count * size
        if end > len(self.data):
            print("[M2] warning: sequence table overruns file; "
                  "skipping animation names.")
            return
        legacy = self.legacy_anim

        def one(r: BinaryReader):
            start_pos = r.tell()
            s = M2Sequence()
            s.id = r.u16()
            s.variation_index = r.u16()
            if legacy:
                s.start_timestamp = r.u32()
                s.end_timestamp = r.u32()
                s.duration = max(0, s.end_timestamp - s.start_timestamp)
            else:
                s.duration = r.u32()
                s.start_timestamp = 0
                s.end_timestamp = s.duration
            s.movespeed = r.f32()
            s.flags = r.u32()
            s.frequency = r.i16()        # selection weight among variations
            r.u16()                      # padding
            r.u32(); r.u32()             # replay range
            if legacy:
                s.blend_time_in = s.blend_time_out = r.u32() & 0xFFFF
            else:
                s.blend_time_in = r.u16()
                s.blend_time_out = r.u16()
            # variationNext (the linked list to the next variation) sits near
            if not legacy:
                r.seek(start_pos + size - 4)
                s.variation_next = r.i16()
                s.alias_next = r.u16()
            r.seek(start_pos + size)     # skip to the next entry
            return s

        self.model.sequences = repair_zero_blend_times(arr.read(self.reader, one))

    def read_attachments(self):
        arr = self._a_attachments
        if arr is None or arr.count == 0:
            return
        track_size = self.bone_track_size

        def one(r: BinaryReader):
            a = M2Attachment()
            a.id = r.u32()
            a.bone = r.u16()
            r.u16()                      # unknown
            a.position = r.vec3()
            r.skip(track_size)           # animate_attached M2Track header
            return a

        self.model.attachments = arr.read(self.reader, one)

    # -- skin loading -----------------------------------------------------
    def read_embedded_skin(self, arr):
        """Classic / TBC: first skin profile lives inside the .m2."""
        profiles = arr.read(
            self.reader,
            lambda r: skin_mod.parse_skin(
                r, self.skin_has_sort, self.skin_modern_batch),
        )
        if profiles:
            self.model.skin = profiles[0]

    def _load_external_skin(self, m2_path: str):
        if self.skin_embedded or not m2_path:
            return
        skin_path = skin_mod.find_skin_file(m2_path)
        if not skin_path:
            print("[M2] no .skin file found next to %s: mesh will be empty."
                  % m2_path)
            return
        try:
            with open(skin_path, "rb") as f:
                skin_data = f.read()
            skin_reader = BinaryReader(skin_data, base=0)
            self.model.skin = skin_mod.parse_skin(
                skin_reader, self.skin_has_sort, self.skin_modern_batch)
        except Exception as exc:  # noqa: BLE001
            # A bad/unsupported skin must not sink skeleton + animation import.
            print("[M2] warning: failed to read skin %r: %r" % (skin_path, exc))
            self.model.skin = None
