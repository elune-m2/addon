"""Low-level binary reading helpers shared by every parser."""

import struct


class BinaryReader:
    """A cursor over a bytes buffer with little-endian helpers."""

    def __init__(self, data: bytes, base: int = 0):
        self.data = data
        self.base = base
        self.pos = 0

    # -- cursor management ------------------------------------------------
    def seek(self, pos: int):
        self.pos = pos

    def tell(self) -> int:
        return self.pos

    def skip(self, n: int):
        self.pos += n

    # -- scalar reads -----------------------------------------------------
    def _read(self, fmt: str):
        size = struct.calcsize(fmt)
        value = struct.unpack_from("<" + fmt, self.data, self.pos)
        self.pos += size
        return value

    def u8(self):
        return self._read("B")[0]

    def i8(self):
        return self._read("b")[0]

    def u16(self):
        return self._read("H")[0]

    def i16(self):
        return self._read("h")[0]

    def u32(self):
        return self._read("I")[0]

    def i32(self):
        return self._read("i")[0]

    def f32(self):
        return self._read("f")[0]

    def vec2(self):
        return self._read("2f")

    def vec3(self):
        return self._read("3f")

    def quat(self):
        return self._read("4f")

    def fourcc(self):
        raw = self.data[self.pos:self.pos + 4]
        self.pos += 4
        return raw.decode("ascii", "replace")

    # -- M2Array: (count uint32, offset uint32) ---------------------------
    def m2array(self):
        count = self.u32()
        offset = self.u32()
        return M2Array(count, offset)


class M2Array:
    """A (count, offset) pair. Resolve it against a reader to read records."""

    __slots__ = ("count", "offset")

    def __init__(self, count: int, offset: int):
        self.count = count
        self.offset = offset

    def __repr__(self):
        return f"M2Array(count={self.count}, offset=0x{self.offset:X})"

    def read(self, reader: BinaryReader, read_one):
        """Read ``count`` records via ``read_one(reader)``."""
        if self.count == 0:
            return []
        saved = reader.tell()
        out = []
        reader.seek(self.base_offset(reader))
        for _ in range(self.count):
            out.append(read_one(reader))
        reader.seek(saved)
        return out

    def base_offset(self, reader: BinaryReader) -> int:
        return reader.base + self.offset

    def read_string(self, reader: BinaryReader) -> str:
        if self.count == 0:
            return ""
        start = self.base_offset(reader)
        raw = reader.data[start:start + self.count]
        return raw.split(b"\x00", 1)[0].decode("utf-8", "replace")

    def read_u16(self, reader: BinaryReader):
        return self.read(reader, lambda r: r.u16())

    def read_u32(self, reader: BinaryReader):
        return self.read(reader, lambda r: r.u32())

    def read_arrays(self, reader: BinaryReader):
        """Read ``count`` nested M2Arrays (for the modern per-anim track shape)."""
        return self.read(reader, lambda r: r.m2array())


def scan_chunks(data: bytes, start: int = 0):
    """Return ``{b'NAME': (data_offset, size)}`` for an IFF-style chunk file."""
    chunks = {}
    pos = start
    n = len(data)
    while pos + 8 <= n:
        name = data[pos:pos + 4]
        size = int.from_bytes(data[pos + 4:pos + 8], "little")
        chunks[name] = (pos + 8, size)
        nxt = pos + 8 + size
        if nxt <= pos:           # guard against zero/negative strides
            break
        pos = nxt
    return chunks


def decompress_quat(x, y, z, w):
    """Decode an M2CompQuat (four int16) to a float (x, y, z, w) quaternion."""
    def f(v):
        return ((v + 32768) if v < 0 else (v - 32767)) / 32767.0
    return (f(x), f(y), f(z), f(w))


def read_caabox(reader: BinaryReader):
    """Axis-aligned bounding box: min C3Vector, max C3Vector."""
    return (reader.vec3(), reader.vec3())
