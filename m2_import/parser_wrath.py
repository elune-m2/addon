"""Wrath of the Lich King (v264)."""

from .base_parser import BaseM2Parser
from . import versions


class WrathParser(BaseM2Parser):
    expansion = versions.WRATH
    bone_track_size = 20
    camera_fov_track = False  # pre-Cata static fov
    bone_has_crc = True
    skin_has_sort = True
    skin_modern_batch = True
    skin_embedded = False

    def parse_header(self):
        arrays = self.walk_modern_header()
        self.populate_common(arrays)
