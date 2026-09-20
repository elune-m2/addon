"""Legion / Battle for Azeroth / Shadowlands / Dragonflight (chunked)."""

from .base_parser import BaseM2Parser
from .reader import BinaryReader
from . import versions


class LegionParser(BaseM2Parser):
    expansion = versions.LEGION
    bone_track_size = 20
    bone_has_crc = True
    skin_has_sort = True
    skin_modern_batch = True
    skin_embedded = False

    def __init__(self, data: bytes, base: int = 0, chunks=None):
        super().__init__(data, base)
        # chunks: {fourcc: (data_offset, size)} absolute into ``data``.
        self.chunks = chunks or {}

    def parse_header(self):
        arrays = self.walk_modern_header()
        self.populate_common(arrays)
        self._apply_txid()
        self._capture_chunks()

    def _capture_chunks(self):
        """Preserve auxiliary chunks + SFID so a retail export can reproduce them."""
        for name, (offset, size) in self.chunks.items():
            if name == "MD21":
                continue
            raw = self.data[offset:offset + size]
            self.model.aux_chunks[name] = raw
            if name == "SFID":
                self.model.skin_file_ids = [
                    int.from_bytes(raw[i:i + 4], "little")
                    for i in range(0, size - 3, 4)
                ]

    def _apply_txid(self):
        info = self.chunks.get("TXID")
        if not info:
            return
        offset, size = info
        r = BinaryReader(self.data, base=0)
        r.seek(offset)
        ids = [r.u32() for _ in range(size // 4)]
        for i, fid in enumerate(ids):
            if i < len(self.model.textures):
                self.model.textures[i].file_data_id = fid
                if not self.model.textures[i].filename:
                    self.model.textures[i].filename = f"FileDataID_{fid}"
