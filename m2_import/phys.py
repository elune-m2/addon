"""WoW .phys reader/writer (versions 0-6).

Format reference: wowdev.wiki/PHYS, checked against retail files. A .phys is a
chunk list; every chunk magic is stored REVERSED on disk (PHYS -> "SYHP"). The
same bytes also appear inside modern M2s as the PFDC chunk.

One body attaches to one bone, a body owns 0+ shapes, bodies are linked by
joints, and exactly one body is the root (type 0).

Conventions established from retail data:
  * mat3x4 is 12 floats: the X, Y and Z basis vectors, THEN the translation.
  * Body frames are axis-aligned model space; ``position`` is model space.
  * A joint frame's Z axis is its primary axis (shoulder twist / cone axis).
  * Twist and cone angles are in degrees.
  * From version 2 on the client reads SHOJ entries as 116 bytes.

Every record keeps the fields we don't understand, so read -> write is
byte-identical for an unedited file.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


SHAPE_BOX = 0
SHAPE_CAPSULE = 1
SHAPE_SPHERE = 2
SHAPE_POLYTOPE = 3          # v3+, preserved raw

JOINT_SPHERICAL = 0
JOINT_SHOULDER = 1
JOINT_WELD = 2
JOINT_REVOLUTE = 3          # v2+
JOINT_PRISMATIC = 4         # v2+
JOINT_DISTANCE = 5          # v2+

BODY_ROOT = 0               # the single anchor body
BODY_DYNAMIC = 1            # simulated
BODY_KINEMATIC = 2          # follows its bone, pushes dynamic bodies

# Version written for rigs authored in Blender: what every retail character
# and mount rig uses (BDY4, SHP2, WLJ3, SHJ2).
DEFAULT_VERSION = 6

Vec3 = Tuple[float, float, float]
Mat3x4 = Tuple[float, ...]   # x axis, y axis, z axis, translation

IDENTITY_AXES = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def make_mat3x4(axes=IDENTITY_AXES, pos: Vec3 = (0.0, 0.0, 0.0)) -> Mat3x4:
    return tuple(float(a) for a in axes) + (float(pos[0]), float(pos[1]), float(pos[2]))


def identity_mat3x4(pos: Vec3 = (0.0, 0.0, 0.0)) -> Mat3x4:
    return make_mat3x4(IDENTITY_AXES, pos)


# --- records ---------------------------------------------------------------

@dataclass
class BoxShape:
    frame: Mat3x4 = field(default_factory=identity_mat3x4)
    half_extents: Vec3 = (0.05, 0.05, 0.05)


@dataclass
class CapsuleShape:
    p1: Vec3 = (0.0, 0.0, 0.0)
    p2: Vec3 = (0.0, 0.0, 0.1)
    radius: float = 0.03


@dataclass
class SphereShape:
    center: Vec3 = (0.0, 0.0, 0.0)
    radius: float = 0.05


@dataclass
class Shape:
    shape_type: int = SHAPE_CAPSULE
    shape_index: int = 0              # into BOXS / CAPS / SPHS / PLYT
    friction: float = 0.7
    restitution: float = 0.1
    density: float = 1000.0
    unk: bytes = b"\x00\x00\x00\x00"
    # SHP2 (v2+)
    x14: int = 0
    x18: float = 1.0
    x1c: int = 0
    x1e: int = 0


@dataclass
class Body:
    type: int = BODY_DYNAMIC
    position: Vec3 = (0.0, 0.0, 0.0)
    bone_index: int = 0
    shapes_base: int = 0
    shapes_count: int = 0
    x1c: float = 1.0                  # BDY2+
    unk0: float = 1.0                 # BDY3+
    drag: float = 0.0                 # BDY3+
    unk1: float = 0.0                 # BDY3+
    x28: float = 0.89999998           # BDY3+
    x2c: bytes = b"\x00\x00\x00\x00"  # BDY4
    pad_a: bytes = b"\x00\x00"        # BODY/BDY2 padding, kept for exact rewrite
    pad_b: bytes = b"\x00\x00"


@dataclass
class WeldJoint:
    frame_a: Mat3x4 = field(default_factory=identity_mat3x4)
    frame_b: Mat3x4 = field(default_factory=identity_mat3x4)
    angular_frequency_hz: float = 0.0
    angular_damping_ratio: float = 1.0
    linear_frequency_hz: float = 0.0   # WLJ2+
    linear_damping_ratio: float = 0.0  # WLJ2+
    unk70: float = 0.0                 # WLJ3


@dataclass
class SphericalJoint:
    anchor_a: Vec3 = (0.0, 0.0, 0.0)
    anchor_b: Vec3 = (0.0, 0.0, 0.0)
    friction_torque: float = 0.0


@dataclass
class ShoulderJoint:
    frame_a: Mat3x4 = field(default_factory=identity_mat3x4)
    frame_b: Mat3x4 = field(default_factory=identity_mat3x4)
    lower_twist: float = -10.0         # degrees
    upper_twist: float = 10.0
    cone_angle: float = 20.0
    max_motor_torque: float = 0.0      # 116-byte SHOJ / SHJ2
    motor_mode: int = 0
    motor_frequency_hz: float = 0.0    # SHJ2
    motor_damping_ratio: float = 0.0   # SHJ2


@dataclass
class RevoluteJoint:
    frame_a: Mat3x4 = field(default_factory=identity_mat3x4)
    frame_b: Mat3x4 = field(default_factory=identity_mat3x4)
    lower_angle: float = 0.0
    upper_angle: float = 0.0
    max_motor_torque: float = 0.0
    motor_mode: int = 0
    motor_frequency_hz: float = 0.0    # REV2
    motor_damping_ratio: float = 0.0   # REV2


@dataclass
class PrismaticJoint:
    frame_a: Mat3x4 = field(default_factory=identity_mat3x4)
    frame_b: Mat3x4 = field(default_factory=identity_mat3x4)
    lower_limit: float = 0.0
    upper_limit: float = 0.0
    x68: float = 0.0
    max_motor_force: float = 0.0
    x70: float = 0.0
    motor_mode: int = 0
    motor_frequency_hz: float = 0.0    # PRS2
    motor_damping_ratio: float = 0.0   # PRS2


@dataclass
class DistanceJoint:
    anchor_a: Vec3 = (0.0, 0.0, 0.0)
    anchor_b: Vec3 = (0.0, 0.0, 0.0)
    distance_factor: float = 1.0


@dataclass
class Joint:
    body_a: int = 0
    body_b: int = 0
    joint_type: int = JOINT_WELD
    joint_id: int = 0                  # index into that type's chunk
    unk: bytes = b"\x00\x00\x00\x00"


@dataclass
class PhysDoc:
    version: int = DEFAULT_VERSION
    phyt: Optional[int] = None         # v1+; None = chunk absent
    bodies: List[Body] = field(default_factory=list)
    shapes: List[Shape] = field(default_factory=list)
    boxes: List[BoxShape] = field(default_factory=list)
    capsules: List[CapsuleShape] = field(default_factory=list)
    spheres: List[SphereShape] = field(default_factory=list)
    joints: List[Joint] = field(default_factory=list)
    weld_joints: List[WeldJoint] = field(default_factory=list)
    spherical_joints: List[SphericalJoint] = field(default_factory=list)
    shoulder_joints: List[ShoulderJoint] = field(default_factory=list)
    revolute_joints: List[RevoluteJoint] = field(default_factory=list)
    prismatic_joints: List[PrismaticJoint] = field(default_factory=list)
    distance_joints: List[DistanceJoint] = field(default_factory=list)
    # Chunks we carry through untouched: PLYT, PHYV and anything unknown.
    raw_chunks: Dict[str, bytes] = field(default_factory=dict)
    # How the file was laid out, so an unedited file rewrites identically.
    chunk_order: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)   # kind -> chunk magic
    shoulder_size: int = 0             # 108 / 116 / 124; 0 = pick from version

    def polytope_vertices(self) -> List[List[Vec3]]:
        """Best-effort hull vertices of each PLYT entry (preview only)."""
        return _plyt_vertices(self.raw_chunks.get("PLYT", b""))


# --- primitives ------------------------------------------------------------

def _f3(v) -> bytes:
    return struct.pack("<3f", float(v[0]), float(v[1]), float(v[2]))


def _m34(m) -> bytes:
    return struct.pack("<12f", *[float(x) for x in m])


def _b(raw, n) -> bytes:
    raw = bytes(raw or b"")
    return raw[:n].ljust(n, b"\x00")


def _chunk(magic: str, payload: bytes) -> bytes:
    return magic[::-1].encode("ascii") + struct.pack("<I", len(payload)) + payload


def _iter_chunks(data: bytes):
    off = 0
    while off + 8 <= len(data):
        magic = data[off:off + 4][::-1].decode("ascii", "replace")
        size = struct.unpack_from("<I", data, off + 4)[0]
        yield magic, data[off + 8:off + 8 + size]
        off += 8 + size


def _records(payload: bytes, size: int):
    for i in range(0, len(payload) - size + 1, size):
        yield payload[i:i + size]


# --- readers ---------------------------------------------------------------

def _read_bodies(tag: str, payload: bytes) -> List[Body]:
    out = []
    if tag in ("BODY", "BDY2"):
        size = 28 if tag == "BODY" else 32
        for r in _records(payload, size):
            b = Body()
            b.type = struct.unpack_from("<H", r, 0)[0]
            b.pad_a = r[2:4]
            b.position = struct.unpack_from("<3f", r, 4)
            b.bone_index = struct.unpack_from("<H", r, 16)[0]
            b.pad_b = r[18:20]
            b.shapes_base, b.shapes_count = struct.unpack_from("<ii", r, 20)
            if tag == "BDY2":
                b.x1c = struct.unpack_from("<f", r, 28)[0]
            out.append(b)
        return out
    size = 44 if tag == "BDY3" else 48
    for r in _records(payload, size):
        b = Body()
        b.type, b.bone_index = struct.unpack_from("<HH", r, 0)
        b.position = struct.unpack_from("<3f", r, 4)
        b.shapes_base = struct.unpack_from("<H", r, 16)[0]
        b.pad_b = r[18:20]
        b.shapes_count = struct.unpack_from("<i", r, 20)[0]
        b.unk0, b.x1c, b.drag, b.unk1, b.x28 = struct.unpack_from("<5f", r, 24)
        if tag == "BDY4":
            b.x2c = r[44:48]
        out.append(b)
    return out


def _read_shapes(tag: str, payload: bytes) -> List[Shape]:
    out = []
    for r in _records(payload, 20 if tag == "SHAP" else 32):
        s = Shape()
        s.shape_type, s.shape_index = struct.unpack_from("<hh", r, 0)
        s.unk = r[4:8]
        s.friction, s.restitution, s.density = struct.unpack_from("<3f", r, 8)
        if tag == "SHP2":
            s.x14 = struct.unpack_from("<I", r, 20)[0]
            s.x18 = struct.unpack_from("<f", r, 24)[0]
            s.x1c, s.x1e = struct.unpack_from("<HH", r, 28)
        out.append(s)
    return out


def _read_weld(tag: str, payload: bytes) -> List[WeldJoint]:
    size = {"WELJ": 104, "WLJ2": 112, "WLJ3": 116}[tag]
    out = []
    for r in _records(payload, size):
        w = WeldJoint(struct.unpack_from("<12f", r, 0), struct.unpack_from("<12f", r, 48))
        w.angular_frequency_hz, w.angular_damping_ratio = struct.unpack_from("<2f", r, 96)
        if size >= 112:
            w.linear_frequency_hz, w.linear_damping_ratio = struct.unpack_from("<2f", r, 104)
        if size >= 116:
            w.unk70 = struct.unpack_from("<f", r, 112)[0]
        out.append(w)
    return out


def _shoulder_size(tag: str, version: int, payload_len: int) -> int:
    if tag == "SHJ2":
        return 124
    want = 108 if version < 2 else 116
    if payload_len % want == 0:
        return want
    for alt in (116, 108, 124):
        if payload_len % alt == 0:
            return alt
    return want


def _read_shoulder(payload: bytes, size: int) -> List[ShoulderJoint]:
    out = []
    for r in _records(payload, size):
        s = ShoulderJoint(struct.unpack_from("<12f", r, 0), struct.unpack_from("<12f", r, 48))
        s.lower_twist, s.upper_twist, s.cone_angle = struct.unpack_from("<3f", r, 96)
        if size >= 116:
            s.max_motor_torque = struct.unpack_from("<f", r, 108)[0]
            s.motor_mode = struct.unpack_from("<I", r, 112)[0]
        if size >= 124:
            s.motor_frequency_hz, s.motor_damping_ratio = struct.unpack_from("<2f", r, 116)
        out.append(s)
    return out


def _read_revolute(tag: str, payload: bytes) -> List[RevoluteJoint]:
    size = 112 if tag == "REVJ" else 120
    out = []
    for r in _records(payload, size):
        j = RevoluteJoint(struct.unpack_from("<12f", r, 0), struct.unpack_from("<12f", r, 48))
        j.lower_angle, j.upper_angle, j.max_motor_torque = struct.unpack_from("<3f", r, 96)
        j.motor_mode = struct.unpack_from("<I", r, 108)[0]
        if size == 120:
            j.motor_frequency_hz, j.motor_damping_ratio = struct.unpack_from("<2f", r, 112)
        out.append(j)
    return out


def _read_prismatic(tag: str, payload: bytes) -> List[PrismaticJoint]:
    size = 120 if tag == "PRSJ" else 128
    out = []
    for r in _records(payload, size):
        j = PrismaticJoint(struct.unpack_from("<12f", r, 0), struct.unpack_from("<12f", r, 48))
        (j.lower_limit, j.upper_limit, j.x68,
         j.max_motor_force, j.x70) = struct.unpack_from("<5f", r, 96)
        j.motor_mode = struct.unpack_from("<I", r, 116)[0]
        if size == 128:
            j.motor_frequency_hz, j.motor_damping_ratio = struct.unpack_from("<2f", r, 120)
        out.append(j)
    return out


def _plyt_vertices(payload: bytes) -> List[List[Vec3]]:
    """Hull vertices per polytope. The data block's alignment isn't documented,
    so this is only trusted for drawing a preview; the chunk itself is rewritten
    from the raw bytes."""
    try:
        if len(payload) < 4:
            return []
        count = struct.unpack_from("<I", payload, 0)[0]
        if count <= 0 or count > 4096:
            return []
        headers = []
        off = 4
        for _ in range(count):
            nverts = struct.unpack_from("<I", payload, off)[0]
            n10 = struct.unpack_from("<I", payload, off + 0x10)[0]
            nnodes = struct.unpack_from("<I", payload, off + 0x28)[0]
            headers.append((nverts, n10, nnodes))
            off += 0x50
        out = []
        for nverts, n10, nnodes in headers:
            if nverts > 1024 or off + nverts * 12 > len(payload):
                return out
            out.append([struct.unpack_from("<3f", payload, off + i * 12)
                        for i in range(nverts)])
            off += nverts * 12 + n10 * 16 + n10 + nnodes * 4
        return out
    except struct.error:
        return []


_BODY_TAGS = ("BODY", "BDY2", "BDY3", "BDY4")
_SHAPE_TAGS = ("SHAP", "SHP2")
_WELD_TAGS = ("WELJ", "WLJ2", "WLJ3")
_SHOULDER_TAGS = ("SHOJ", "SHJ2")
_REVOLUTE_TAGS = ("REVJ", "REV2")
_PRISMATIC_TAGS = ("PRSJ", "PRS2")


def read(data: bytes) -> PhysDoc:
    doc = PhysDoc(version=0)
    chunks = list(_iter_chunks(data))
    for magic, payload in chunks:          # version first: SHOJ size depends on it
        if magic == "PHYS" and len(payload) >= 2:
            doc.version = struct.unpack_from("<h", payload, 0)[0]
    for magic, payload in chunks:
        doc.chunk_order.append(magic)
        if magic == "PHYS":
            continue
        if magic == "PHYT":
            doc.phyt = struct.unpack_from("<I", payload, 0)[0] if len(payload) >= 4 else 0
        elif magic in _BODY_TAGS:
            doc.bodies = _read_bodies(magic, payload)
            doc.tags["body"] = magic
        elif magic in _SHAPE_TAGS:
            doc.shapes = _read_shapes(magic, payload)
            doc.tags["shape"] = magic
        elif magic == "BOXS":
            doc.boxes = [BoxShape(struct.unpack_from("<12f", r, 0),
                                  struct.unpack_from("<3f", r, 48))
                         for r in _records(payload, 60)]
        elif magic == "CAPS":
            doc.capsules = [CapsuleShape(struct.unpack_from("<3f", r, 0),
                                         struct.unpack_from("<3f", r, 12),
                                         struct.unpack_from("<f", r, 24)[0])
                            for r in _records(payload, 28)]
        elif magic == "SPHS":
            doc.spheres = [SphereShape(struct.unpack_from("<3f", r, 0),
                                       struct.unpack_from("<f", r, 12)[0])
                           for r in _records(payload, 16)]
        elif magic in _WELD_TAGS:
            doc.weld_joints = _read_weld(magic, payload)
            doc.tags["weld"] = magic
        elif magic == "SPHJ":
            doc.spherical_joints = [
                SphericalJoint(struct.unpack_from("<3f", r, 0),
                               struct.unpack_from("<3f", r, 12),
                               struct.unpack_from("<f", r, 24)[0])
                for r in _records(payload, 28)]
        elif magic in _SHOULDER_TAGS:
            doc.shoulder_size = _shoulder_size(magic, doc.version, len(payload))
            doc.shoulder_joints = _read_shoulder(payload, doc.shoulder_size)
            doc.tags["shoulder"] = magic
        elif magic in _REVOLUTE_TAGS:
            doc.revolute_joints = _read_revolute(magic, payload)
            doc.tags["revolute"] = magic
        elif magic in _PRISMATIC_TAGS:
            doc.prismatic_joints = _read_prismatic(magic, payload)
            doc.tags["prismatic"] = magic
        elif magic == "DSTJ":
            doc.distance_joints = [
                DistanceJoint(struct.unpack_from("<3f", r, 0),
                              struct.unpack_from("<3f", r, 12),
                              struct.unpack_from("<f", r, 24)[0])
                for r in _records(payload, 28)]
        elif magic == "JOIN":
            for r in _records(payload, 16):
                a, b = struct.unpack_from("<II", r, 0)
                jt, jid = struct.unpack_from("<hh", r, 12)
                doc.joints.append(Joint(a, b, jt, jid, r[8:12]))
        else:                               # PLYT, PHYV, anything newer
            doc.raw_chunks[magic] = payload
    return doc


# --- writers ---------------------------------------------------------------

def _body_tag(doc: PhysDoc) -> str:
    if "body" in doc.tags:
        return doc.tags["body"]
    v = doc.version
    return "BODY" if v < 2 else "BDY2" if v == 2 else "BDY3" if v == 3 else "BDY4"


def _write_bodies(tag: str, bodies) -> bytes:
    out = bytearray()
    for b in bodies:
        if tag in ("BODY", "BDY2"):
            out += struct.pack("<H", b.type & 0xFFFF) + _b(b.pad_a, 2) + _f3(b.position)
            out += struct.pack("<H", b.bone_index & 0xFFFF) + _b(b.pad_b, 2)
            out += struct.pack("<ii", b.shapes_base, b.shapes_count)
            if tag == "BDY2":
                out += struct.pack("<f", b.x1c)
        else:
            out += struct.pack("<HH", b.type & 0xFFFF, b.bone_index & 0xFFFF) + _f3(b.position)
            out += struct.pack("<H", b.shapes_base & 0xFFFF) + _b(b.pad_b, 2)
            out += struct.pack("<i", b.shapes_count)
            out += struct.pack("<5f", b.unk0, b.x1c, b.drag, b.unk1, b.x28)
            if tag == "BDY4":
                out += _b(b.x2c, 4)
    return bytes(out)


def _write_shapes(tag: str, shapes) -> bytes:
    out = bytearray()
    for s in shapes:
        out += struct.pack("<hh", s.shape_type, s.shape_index) + _b(s.unk, 4)
        out += struct.pack("<3f", s.friction, s.restitution, s.density)
        if tag == "SHP2":
            out += struct.pack("<IfHH", s.x14 & 0xFFFFFFFF, s.x18,
                               s.x1c & 0xFFFF, s.x1e & 0xFFFF)
    return bytes(out)


def _write_weld(tag: str, joints) -> bytes:
    out = bytearray()
    for w in joints:
        out += _m34(w.frame_a) + _m34(w.frame_b)
        out += struct.pack("<2f", w.angular_frequency_hz, w.angular_damping_ratio)
        if tag in ("WLJ2", "WLJ3"):
            out += struct.pack("<2f", w.linear_frequency_hz, w.linear_damping_ratio)
        if tag == "WLJ3":
            out += struct.pack("<f", w.unk70)
    return bytes(out)


def _write_shoulder(size: int, joints) -> bytes:
    out = bytearray()
    for s in joints:
        out += _m34(s.frame_a) + _m34(s.frame_b)
        out += struct.pack("<3f", s.lower_twist, s.upper_twist, s.cone_angle)
        if size >= 116:
            out += struct.pack("<fI", s.max_motor_torque, s.motor_mode & 0xFFFFFFFF)
        if size >= 124:
            out += struct.pack("<2f", s.motor_frequency_hz, s.motor_damping_ratio)
    return bytes(out)


def _write_revolute(tag: str, joints) -> bytes:
    out = bytearray()
    for j in joints:
        out += _m34(j.frame_a) + _m34(j.frame_b)
        out += struct.pack("<3fI", j.lower_angle, j.upper_angle,
                           j.max_motor_torque, j.motor_mode & 0xFFFFFFFF)
        if tag == "REV2":
            out += struct.pack("<2f", j.motor_frequency_hz, j.motor_damping_ratio)
    return bytes(out)


def _write_prismatic(tag: str, joints) -> bytes:
    out = bytearray()
    for j in joints:
        out += _m34(j.frame_a) + _m34(j.frame_b)
        out += struct.pack("<5fI", j.lower_limit, j.upper_limit, j.x68,
                           j.max_motor_force, j.x70, j.motor_mode & 0xFFFFFFFF)
        if tag == "PRS2":
            out += struct.pack("<2f", j.motor_frequency_hz, j.motor_damping_ratio)
    return bytes(out)


# Layout of the retail v5 sample; chunks a file lacks are simply skipped.
_DEFAULT_ORDER = ["PHYS", "PHYT", "PHYV", "BOXS", "CAPS", "SPHS", "PLYT",
                  "shape", "body", "weld", "SPHJ", "shoulder", "revolute",
                  "prismatic", "DSTJ", "JOIN"]
_KIND_OF = {}
for _kind, _tags in (("body", _BODY_TAGS), ("shape", _SHAPE_TAGS),
                     ("weld", _WELD_TAGS), ("shoulder", _SHOULDER_TAGS),
                     ("revolute", _REVOLUTE_TAGS), ("prismatic", _PRISMATIC_TAGS)):
    for _t in _tags:
        _KIND_OF[_t] = _kind


def write(doc: PhysDoc) -> bytes:
    """Serialize ``doc``. An unedited document rewrites byte-identically."""
    v = doc.version
    modern = v >= 2
    tags = {
        "body": _body_tag(doc),
        "shape": doc.tags.get("shape", "SHP2" if modern else "SHAP"),
        "weld": doc.tags.get("weld", "WELJ" if v < 2 else "WLJ2" if v == 2 else "WLJ3"),
        "shoulder": doc.tags.get("shoulder", "SHJ2" if v >= 6 else "SHOJ"),
        "revolute": doc.tags.get("revolute", "REVJ"),
        "prismatic": doc.tags.get("prismatic", "PRSJ"),
    }
    sh_size = doc.shoulder_size or (124 if tags["shoulder"] == "SHJ2"
                                    else 116 if modern else 108)
    phyt = doc.phyt
    if phyt is None and v >= 1 and not doc.chunk_order:
        phyt = 4                       # authored file: what retail mounts/belts carry

    payloads = {
        "PHYS": struct.pack("<h", v),
        "PHYT": None if phyt is None else struct.pack("<I", phyt & 0xFFFFFFFF),
        "BOXS": b"".join(_m34(b.frame) + _f3(b.half_extents) for b in doc.boxes),
        "CAPS": b"".join(_f3(c.p1) + _f3(c.p2) + struct.pack("<f", c.radius)
                         for c in doc.capsules),
        "SPHS": b"".join(_f3(s.center) + struct.pack("<f", s.radius) for s in doc.spheres),
        "shape": _write_shapes(tags["shape"], doc.shapes),
        "body": _write_bodies(tags["body"], doc.bodies),
        "weld": _write_weld(tags["weld"], doc.weld_joints),
        "SPHJ": b"".join(_f3(s.anchor_a) + _f3(s.anchor_b)
                         + struct.pack("<f", s.friction_torque)
                         for s in doc.spherical_joints),
        "shoulder": _write_shoulder(sh_size, doc.shoulder_joints),
        "revolute": _write_revolute(tags["revolute"], doc.revolute_joints),
        "prismatic": _write_prismatic(tags["prismatic"], doc.prismatic_joints),
        "DSTJ": b"".join(_f3(d.anchor_a) + _f3(d.anchor_b)
                         + struct.pack("<f", d.distance_factor)
                         for d in doc.distance_joints),
        "JOIN": b"".join(struct.pack("<II", j.body_a & 0xFFFFFFFF, j.body_b & 0xFFFFFFFF)
                         + _b(j.unk, 4) + struct.pack("<hh", j.joint_type, j.joint_id)
                         for j in doc.joints),
    }
    for magic, raw in doc.raw_chunks.items():
        payloads[magic] = raw

    # Original order first, then anything new in the default order.
    order = [_KIND_OF.get(m, m) for m in doc.chunk_order]
    for key in _DEFAULT_ORDER + list(doc.raw_chunks):
        if key not in order:
            order.append(key)
    if "PHYS" in order:
        order.remove("PHYS")
    order.insert(0, "PHYS")

    out = bytearray()
    seen = set()
    for key in order:
        if key in seen:
            continue
        seen.add(key)
        payload = payloads.get(key)
        if payload is None:
            continue
        was_present = key in [_KIND_OF.get(m, m) for m in doc.chunk_order]
        if not payload and not was_present and key != "PHYS":
            continue
        out += _chunk(tags.get(key, key), payload)
    return bytes(out)


# --- helpers shared by the Blender side ------------------------------------

def shape_volume(doc: PhysDoc, shape: Shape) -> float:
    import math
    t, i = shape.shape_type, shape.shape_index
    if t == SHAPE_CAPSULE and 0 <= i < len(doc.capsules):
        c = doc.capsules[i]
        length = math.dist(c.p1, c.p2)
        return math.pi * c.radius ** 2 * length + 4.0 / 3.0 * math.pi * c.radius ** 3
    if t == SHAPE_SPHERE and 0 <= i < len(doc.spheres):
        return 4.0 / 3.0 * math.pi * doc.spheres[i].radius ** 3
    if t == SHAPE_BOX and 0 <= i < len(doc.boxes):
        hx, hy, hz = doc.boxes[i].half_extents
        return 8.0 * hx * hy * hz
    return 0.0


def summary(doc: PhysDoc) -> str:
    return ("v%d: %d bodies, %d shapes (%d box / %d capsule / %d sphere), %d joints"
            % (doc.version, len(doc.bodies), len(doc.shapes), len(doc.boxes),
               len(doc.capsules), len(doc.spheres), len(doc.joints)))
