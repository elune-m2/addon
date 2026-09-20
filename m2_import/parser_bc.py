"""The Burning Crusade (v260-263)."""

from .base_parser import BaseM2Parser
from . import versions


class TBCParser(BaseM2Parser):
    expansion = versions.TBC
    bone_track_size = 28
    camera_fov_track = False  # pre-Cata static fov
    bone_has_crc = True
    skin_has_sort = True
    skin_modern_batch = False
    skin_embedded = True

    def parse_header(self):
        arrays = self.walk_old_header()
        self.populate_common(arrays)
        self.read_embedded_skin(arrays["skin_profiles"])
