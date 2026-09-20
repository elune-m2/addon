"""Mists of Pandaria / Warlords of Draenor (v272)."""

from .base_parser import BaseM2Parser
from . import versions


class MopWodParser(BaseM2Parser):
    expansion = versions.MOP_WOD
    bone_track_size = 20
    bone_has_crc = True
    skin_has_sort = True
    skin_modern_batch = True
    skin_embedded = False

    def parse_header(self):
        arrays = self.walk_modern_header()
        self.populate_common(arrays)
