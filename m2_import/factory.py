"""Chooses and constructs the right parser for a given .m2 buffer."""

from .reader import BinaryReader
from . import versions
from .parser_vanilla import VanillaParser
from .parser_bc import TBCParser
from .parser_wrath import WrathParser
from .parser_cata import CataParser
from .parser_mop_wod import MopWodParser
from .parser_legion import LegionParser

PARSERS = {
    versions.VANILLA: VanillaParser,
    versions.TBC: TBCParser,
    versions.WRATH: WrathParser,
    versions.CATA: CataParser,
    versions.MOP_WOD: MopWodParser,
    versions.LEGION: LegionParser,
}


def _scan_chunks(data: bytes):
    """Return {fourcc: (data_offset, size)} for a chunked M2."""
    chunks = {}
    pos = 0
    n = len(data)
    while pos + 8 <= n:
        name = data[pos:pos + 4].decode("ascii", "replace")
        size = int.from_bytes(data[pos + 4:pos + 8], "little")
        data_off = pos + 8
        chunks[name] = (data_off, size)
        pos = data_off + size
    return chunks


def detect(data: bytes):
    """Return (is_chunked, base, version, chunks)."""
    magic = data[0:4]
    if magic == b"MD20":
        version = int.from_bytes(data[4:8], "little")
        return False, 0, version, {}
    # Chunked: find MD21.
    chunks = _scan_chunks(data)
    if "MD21" not in chunks:
        raise ValueError(
            "Not a recognisable M2 file: no MD20 magic and no MD21 chunk.")
    base, _size = chunks["MD21"]
    # Inner data begins with its own MD20 magic + version.
    version = int.from_bytes(data[base + 4:base + 8], "little")
    return True, base, version, chunks


def make_parser(data: bytes, expansion: str = "AUTO"):
    """Build the appropriate parser instance."""
    is_chunked, base, version, chunks = detect(data)

    if expansion == "AUTO":
        if is_chunked:
            expansion = versions.LEGION
        else:
            expansion = versions.expansion_from_version(version)

    if expansion == versions.LEGION:
        # Legion parser works for chunked files; if a user forces LEGION on a
        # flat file, base 0 is correct and chunks is empty.
        return LegionParser(data, base=base, chunks=chunks)

    parser_cls = PARSERS.get(expansion)
    if parser_cls is None:
        raise ValueError(f"Unknown expansion '{expansion}'.")
    return parser_cls(data, base=base)


def load_model(m2_path: str, expansion: str = "AUTO"):
    with open(m2_path, "rb") as f:
        data = f.read()
    parser = make_parser(data, expansion)
    return parser.parse(m2_path)
