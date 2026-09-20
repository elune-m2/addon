"""Version-independent intermediate representation."""


class M2Vertex:
    __slots__ = ("pos", "bone_weights", "bone_indices", "normal", "uv1", "uv2")

    def __init__(self):
        self.pos = (0.0, 0.0, 0.0)
        self.bone_weights = (0, 0, 0, 0)
        self.bone_indices = (0, 0, 0, 0)
        self.normal = (0.0, 0.0, 1.0)
        self.uv1 = (0.0, 0.0)
        self.uv2 = (0.0, 0.0)


class M2Texture:
    __slots__ = ("type", "flags", "filename", "file_data_id")

    def __init__(self, tex_type=0, flags=0, filename="", file_data_id=0):
        self.type = tex_type
        self.flags = flags
        self.filename = filename
        self.file_data_id = file_data_id


class M2Material:
    __slots__ = ("flags", "blend_mode")

    def __init__(self, flags=0, blend_mode=0):
        self.flags = flags
        self.blend_mode = blend_mode


class M2AnimTrack:
    """A single animated channel (translation, rotation or scale)."""
    __slots__ = ("interpolation_type", "global_sequence", "timelines", "flat")

    def __init__(self):
        self.interpolation_type = 0
        self.global_sequence = -1
        self.timelines = []   # modern: list[ list[(time_ms, value)] ]
        self.flat = None      # legacy: (list[time_ms], list[value])

    def is_empty(self):
        if self.flat is not None:
            return not self.flat[0]
        return not any(self.timelines)


class M2Bone:
    __slots__ = (
        "key_bone_id", "flags", "parent", "pivot", "name",
        "translation", "rotation", "scale", "submesh_id", "name_crc",
    )

    def __init__(self):
        self.key_bone_id = -1
        self.flags = 0
        self.parent = -1
        self.pivot = (0.0, 0.0, 0.0)
        self.name = ""
        self.translation = None   # M2AnimTrack or None
        self.rotation = None      # M2AnimTrack or None
        self.scale = None         # M2AnimTrack or None
        self.submesh_id = 0       # M2CompBone.submesh_id
        self.name_crc = 0         # boneNameCRC (client uses it to identify bones)

    def is_animated(self):
        for t in (self.translation, self.rotation, self.scale):
            if t is not None and not t.is_empty():
                return True
        return False


class M2Sequence:
    """One animation in the model (an entry in AnimationData.dbc)."""
    __slots__ = (
        "id", "variation_index", "duration", "flags",
        "start_timestamp", "end_timestamp",
        "frequency", "variation_next", "alias_next",
        "movespeed", "blend_time_in", "blend_time_out",
    )

    def __init__(self):
        self.id = 0
        self.variation_index = 0
        self.duration = 0        # milliseconds
        self.flags = 0
        # Legacy single-timeline window (== 0/duration for modern files).
        self.start_timestamp = 0
        self.end_timestamp = 0
        self.frequency = 0
        # Index (into the sequence array) of the next variation with the same
        # id, or -1 to end the chain. This links Stand -> Stand_1 -> ...
        self.variation_next = -1
        self.alias_next = 0
        self.movespeed = 0.0
        # Cross-fade (ms) into / out of this animation. 0 makes the client snap
        # between animations; retail uses 150 almost everywhere.
        # Retail stores (in=150, out=0): keep out at 0 so the pair also reads
        # as a plain uint32 of 150 on clients that treat it as one field.
        self.blend_time_in = 150
        self.blend_time_out = 0


def repair_zero_blend_times(sequences):
    """Restore default blend times on a table whose blends are ALL zero."""
    # Retail never ships that (most sequences blend over 150 ms); it is the
    # signature of older exports from this addon, which wrote 0 everywhere and
    # made the client snap between animations.
    if len(sequences) > 1 and not any(
            s.blend_time_in or s.blend_time_out for s in sequences):
        for s in sequences:
            s.blend_time_in = 150
            s.blend_time_out = 0
        print("[M2] all blend times were 0; restored the 150 ms default.")
    return sequences


class M2Attachment:
    """An attachment point: where weapons / effects / etc. are mounted."""
    __slots__ = ("id", "bone", "position")

    def __init__(self):
        self.id = 0
        self.bone = -1
        self.position = (0.0, 0.0, 0.0)


class M2Event:
    """A timed trigger inside an animation."""
    __slots__ = ("identifier", "data", "bone", "position", "timestamps")

    def __init__(self):
        self.identifier = ""        # 4 chars, e.g. "$SHR"
        self.data = 0
        self.bone = 0
        self.position = (0.0, 0.0, 0.0)
        self.timestamps = []        # list[ list[int] ] -- per sequence


class M2Camera:
    """A camera: type (portrait/character), clip planes, and animated position /"""
    __slots__ = ("type", "far_clip", "near_clip", "fov_static",
                 "position_base", "target_base",
                 "position", "target", "roll", "fov")

    def __init__(self):
        self.type = -1                       # -1 free, 0 portrait, 1 char info
        self.far_clip = 0.0
        self.near_clip = 0.0
        self.fov_static = 0.0                # radians; older files (no fov track)
        self.position_base = (0.0, 0.0, 0.0)
        self.target_base = (0.0, 0.0, 0.0)
        self.position = None                 # M2AnimTrack of vec3 (spline value)
        self.target = None                   # M2AnimTrack of vec3
        self.roll = None                     # M2AnimTrack of float
        self.fov = None                      # M2AnimTrack of float (Cata+)


class M2SubMesh:
    """One M2SkinSection: a contiguous range of triangles drawn together."""
    __slots__ = (
        "skin_section_id", "level", "vertex_start", "vertex_count",
        "index_start", "index_count", "bone_count", "bone_combo_index",
        "center_bone_index", "bone_influences",
        "center", "sort_center", "sort_radius",
    )

    def __init__(self):
        self.skin_section_id = 0
        self.level = 0
        self.vertex_start = 0
        self.vertex_count = 0
        self.index_start = 0
        self.index_count = 0
        self.bone_count = 0
        self.bone_combo_index = 0
        self.center_bone_index = 0
        self.bone_influences = 4
        self.center = (0.0, 0.0, 0.0)
        self.sort_center = (0.0, 0.0, 0.0)
        self.sort_radius = 0.0


class M2Batch:
    """One M2Batch / texture unit: ties a submesh to a material + texture."""
    __slots__ = (
        "flags", "shader_id", "submesh_index", "material_index",
        "texture_combo_index", "color_index", "material_layer",
        "texture_count", "texture_coord_combo", "texture_weight_combo",
        "texture_transform_combo", "geoset_index",
    )

    def __init__(self):
        self.flags = 0
        self.shader_id = 0
        self.submesh_index = 0
        # The uint16 after skinSectionIndex. Retail writes 0 in every batch of
        # every model checked; it is NOT a second copy of the submesh index.
        self.geoset_index = 0
        self.material_index = 0
        self.texture_combo_index = 0
        self.color_index = -1
        self.material_layer = 0
        self.texture_count = 1
        # Combo (lookup) indices the client dereferences; preserving them and
        # the arrays they point at is required or the game crashes on load.
        self.texture_coord_combo = 0
        self.texture_weight_combo = 0
        self.texture_transform_combo = 0


class M2SkinProfile:
    """A LOD view: which global vertices and triangles form the mesh."""
    __slots__ = ("vertices", "triangles", "submeshes", "batches", "bone_indices")

    def __init__(self):
        self.vertices = []   # local index -> global vertex index
        self.triangles = []  # indices into ``vertices``
        self.submeshes = []  # list[M2SubMesh]
        self.batches = []    # list[M2Batch]
        self.bone_indices = []  # per skin-vertex: 4 palette-local bone indices


class M2Model:
    """Everything the Blender builder needs."""

    def __init__(self):
        self.name = ""
        self.version = 0
        self.expansion = ""
        self.global_flags = 0
        self.vertices = []          # list[M2Vertex]
        self.textures = []          # list[M2Texture]
        self.materials = []         # list[M2Material]
        self.texture_lookup = []    # list[int] -> index into textures
        self.bone_lookup = []       # boneCombos (bone palette table)
        self.key_bone_lookup = []   # boneIndicesById (key-bone -> bone index)
        self.bones = []             # list[M2Bone]
        self.sequences = []         # list[M2Sequence]
        self.seq_lookup = []        # sequenceIdxHashById (animation lookup table)
        self.global_loops = []      # list[int] global sequence durations (ms)
        self.attachments = []       # list[M2Attachment]
        self.attachment_lookup = []  # attachmentIndicesById
        self.cameras = []           # list[M2Camera]
        self.camera_lookup = []     # cameraIndicesById
        self.events = []            # list[M2Event] -- sheathe/effect triggers
        # Header bounding box (render bounds the client frames the view camera
        # around). Read on import; None until then.
        self.bounding_min = None    # (x, y, z) in WoW space
        self.bounding_max = None    # (x, y, z)
        self.bounding_radius = 0.0
        # Set by export when the scene carries an editable bounding-box object;
        # (min, max) tuple in WoW space. Overrides the vertex-derived bounds.
        self.bounding_override = None
        self.skin = None            # M2SkinProfile (LOD 0)
        self.source_path = ""
        # Retail export extras (preserved from the source where available).
        self.skin_file_ids = []     # SFID FileDataIDs
        self.aux_chunks = {}        # name(str) -> raw bytes (non-MD21 chunks)
        # Render arrays the batches index into. We keep the counts + the combo
        self.n_colors = 0
        self.n_texture_weights = 0
        self.n_texture_transforms = 0
        self.tex_coord_combos = []
        self.tex_weight_combos = []
        self.tex_transform_combos = []
        self.texture_indices_by_id = []   # replaceable-texture lookup
        # Full animated content (preserved for faithful re-export).
        self.colors = []        # list[(color_track C3Vec, alpha_track fixed16)]
        self.weights = []       # list[weight_track fixed16]
        self.transforms = []    # list[(trans C3Vec, rot C4Quat, scale C3Vec)]
