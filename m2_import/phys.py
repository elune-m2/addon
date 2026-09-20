"""WoW .phys file writer/reader.

Format reference: wowdev.wiki/PHYS. .phys files are chunked; every chunk
magic is stored REVERSED on disk (e.g. PHYS → SYHP). One body attaches to
one bone; bodies are connected by joints; a body owns 1+ shapes.

This module targets the widest-compatibility version (0) by default. The
writer emits: PHYS, BODY, SHAP, BOXS/CAPS/SPHS (as needed), JOIN,
WELJ/SPHJ/SHOJ (as needed). Version 2+ chunk variants and PLYT are not
emitted here — Blender's rigid-body world doesn't provide the extra
fields cleanly, and version 0 is what the client happily loads for
cloth/buckle-style rigs.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# --- shape / joint kinds ---------------------------------------------------

SHAPE_BOX = 0
SHAPE_CAPSULE = 1
SHAPE_SPHERE = 2
SHAPE_POLYTOPE = 3  # v3+, not emitted

JOINT_SPHERICAL = 0
JOINT_SHOULDER = 1
JOINT_WELD = 2
JOINT_REVOLUTE = 3   # v2+
JOINT_PRISMATIC = 4  # v2+
JOINT_DISTANCE = 5   # v2+

# BODY.type: 0 = root, 1 = dynamic (dm_dynamicBody), other = kinematic-ish.
BODY_ROOT = 0
BODY_DYNAMIC = 1
BODY_KINEMATIC = 2


# --- data classes ----------------------------------------------------------

Vec3 = Tuple[float, float, float]
# mat3x4: 12 floats (row-major 3×4 -> orientation basis + position).
Mat3x4 = Tuple[float, ...]


def identity_mat3x4(pos: Vec3 = (0.0, 0.0, 0.0)) -> Mat3x4:
    """3x4 = 3 rows of [xx xy xz t] (identity rotation, given translation)."""
    return (
        1.0, 0.0, 0.0, pos[0],
        0.0, 1.0, 0.0, pos[1],
        0.0, 0.0, 1.0, pos[2],
    )


@dataclass
class BoxShape:
    frame: Mat3x4                     # mat3x4 a
    half_extents: Vec3                # vec3 c


@dataclass
class CapsuleShape:
    p1: Vec3
    p2: Vec3
    radius: float


@dataclass
class SphereShape:
    center: Vec3
    radius: float


@dataclass
class Shape:
    shape_type: int                   # SHAPE_*
    shape_index: int                  # index into the corresponding shape chunk
    friction: float = 0.5
    restitution: float = 0.0
    density: float = 1.0
    unk: bytes = b"\x00\x00\x00\x00"  # 4 bytes


@dataclass
class Body:
    type: int                         # BODY_ROOT / _DYNAMIC / _KINEMATIC
    position: Vec3
    bone_index: int
    shapes_base: int                  # first shape in Shape[] this body owns
    shapes_count: int                 # how many shapes this body owns


@dataclass
class WeldJoint:
    frame_a: Mat3x4
    frame_b: Mat3x4
    angular_frequency_hz: float = 0.0
    angular_damping_ratio: float = 1.0


@dataclass
class SphericalJoint:
    anchor_a: Vec3
    anchor_b: Vec3
    friction_torque: float = 0.0


@dataclass
class ShoulderJoint:
    frame_a: Mat3x4
    frame_b: Mat3x4
    lower_twist: float = 0.0
    upper_twist: float = 0.0
    cone_angle: float = 0.0


@dataclass
class Joint:
    body_a: int
    body_b: int
    joint_type: int                   # JOINT_*
    joint_id: int                     # index into WELJ/SPHJ/SHOJ etc.
    unk: bytes = b"\x00\x00\x00\x00"


@dataclass
class PhysDoc:
    version: int = 0
    bodies: List[Body] = field(default_factory=list)
    shapes: List[Shape] = field(default_factory=list)
    boxes: List[BoxShape] = field(default_factory=list)
    capsules: List[CapsuleShape] = field(default_factory=list)
    spheres: List[SphereShape] = field(default_factory=list)
    joints: List[Joint] = field(default_factory=list)
    weld_joints: List[WeldJoint] = field(default_factory=list)
    spherical_joints: List[SphericalJoint] = field(default_factory=list)
    shoulder_joints: List[ShoulderJoint] = field(default_factory=list)


# --- writer ----------------------------------------------------------------

def _rev(magic: str) -> bytes:
    """PHYS chunks are stored reversed on disk."""
    return magic[::-1].encode("ascii")


def _chunk(magic: str, body: bytes) -> bytes:
    return _rev(magic) + struct.pack("<I", len(body)) + body


def _f3(v: Vec3) -> bytes:
    return struct.pack("<fff", float(v[0]), float(v[1]), float(v[2]))


def _mat3x4(m: Mat3x4) -> bytes:
    return struct.pack("<12f", *[float(x) for x in m])


def _write_body(b: Body) -> bytes:
    # BODY v0/v1 layout (28 bytes):
    # ushort type; char pad[2]; vec3 pos; ushort bone_index; char pad[2];
    # int shapes_base; int shapes_count;
    return (
        struct.pack("<H2x", b.type & 0xFFFF)
        + _f3(b.position)
        + struct.pack("<H2xii", b.bone_index & 0xFFFF, b.shapes_base, b.shapes_count)
    )


def _write_shape(s: Shape) -> bytes:
    # SHAP v0/v1 layout (20 bytes):
    # short type; short index; char unk[4]; float friction; float rest; float density;
    unk = s.unk if len(s.unk) == 4 else s.unk[:4].ljust(4, b"\x00")
    return (
        struct.pack("<hh", s.shape_type, s.shape_index)
        + unk
        + struct.pack("<fff", s.friction, s.restitution, s.density)
    )


def _write_box(bx: BoxShape) -> bytes:
    return _mat3x4(bx.frame) + _f3(bx.half_extents)


def _write_capsule(c: CapsuleShape) -> bytes:
    return _f3(c.p1) + _f3(c.p2) + struct.pack("<f", c.radius)


def _write_sphere(sp: SphereShape) -> bytes:
    return _f3(sp.center) + struct.pack("<f", sp.radius)


def _write_join(j: Joint) -> bytes:
    unk = j.unk if len(j.unk) == 4 else j.unk[:4].ljust(4, b"\x00")
    return (
        struct.pack("<II", j.body_a & 0xFFFFFFFF, j.body_b & 0xFFFFFFFF)
        + unk
        + struct.pack("<hh", j.joint_type, j.joint_id)
    )


def _write_welj(w: WeldJoint) -> bytes:
    return _mat3x4(w.frame_a) + _mat3x4(w.frame_b) + struct.pack(
        "<ff", w.angular_frequency_hz, w.angular_damping_ratio,
    )


def _write_sphj(s: SphericalJoint) -> bytes:
    return _f3(s.anchor_a) + _f3(s.anchor_b) + struct.pack("<f", s.friction_torque)


def _write_shoj(s: ShoulderJoint) -> bytes:
    return _mat3x4(s.frame_a) + _mat3x4(s.frame_b) + struct.pack(
        "<fff", s.lower_twist, s.upper_twist, s.cone_angle,
    )


def write(doc: PhysDoc) -> bytes:
    """Serialize the phys document to bytes."""
    out = bytearray()
    # PHYS is 2-byte version (rest of file assumes v0 payload here).
    out += _chunk("PHYS", struct.pack("<h", doc.version & 0xFFFF))

    # Shapes must come before BODY in some parsers' bookkeeping, but the wiki
    # says chunks after PHYS are unordered. Match the sample file's ordering
    # to be safe: shape data → SHAP → BODY → joint data → JOIN.
    if doc.boxes:
        out += _chunk("BOXS", b"".join(_write_box(b) for b in doc.boxes))
    if doc.capsules:
        out += _chunk("CAPS", b"".join(_write_capsule(c) for c in doc.capsules))
    if doc.spheres:
        out += _chunk("SPHS", b"".join(_write_sphere(s) for s in doc.spheres))
    if doc.shapes:
        out += _chunk("SHAP", b"".join(_write_shape(s) for s in doc.shapes))
    if doc.bodies:
        out += _chunk("BODY", b"".join(_write_body(b) for b in doc.bodies))
    if doc.weld_joints:
        out += _chunk("WELJ", b"".join(_write_welj(w) for w in doc.weld_joints))
    if doc.spherical_joints:
        out += _chunk("SPHJ", b"".join(_write_sphj(s) for s in doc.spherical_joints))
    if doc.shoulder_joints:
        out += _chunk("SHOJ", b"".join(_write_shoj(s) for s in doc.shoulder_joints))
    if doc.joints:
        out += _chunk("JOIN", b"".join(_write_join(j) for j in doc.joints))
    return bytes(out)


# --- reader (round-trip verification helper, not used by the addon) --------

def _iter_chunks(data: bytes):
    off = 0
    while off + 8 <= len(data):
        magic = data[off:off+4][::-1].decode("ascii", "replace")
        size = struct.unpack("<I", data[off+4:off+8])[0]
        yield magic, data[off+8:off+8+size]
        off += 8 + size


def _read_body(data: bytes) -> List[Body]:
    out = []
    for i in range(0, len(data), 28):
        b = data[i:i+28]
        typ, = struct.unpack("<H", b[0:2])
        px, py, pz = struct.unpack("<fff", b[4:16])
        bone, = struct.unpack("<H", b[16:18])
        sb, sc = struct.unpack("<ii", b[20:28])
        out.append(Body(typ, (px, py, pz), bone, sb, sc))
    return out


def _read_shape(data: bytes) -> List[Shape]:
    out = []
    for i in range(0, len(data), 20):
        b = data[i:i+20]
        st, si = struct.unpack("<hh", b[0:4])
        unk = b[4:8]
        fr, rs, dn = struct.unpack("<fff", b[8:20])
        out.append(Shape(st, si, fr, rs, dn, unk))
    return out


def _read_capsule(data: bytes) -> List[CapsuleShape]:
    out = []
    for i in range(0, len(data), 28):
        b = data[i:i+28]
        p1 = struct.unpack("<fff", b[0:12])
        p2 = struct.unpack("<fff", b[12:24])
        r, = struct.unpack("<f", b[24:28])
        out.append(CapsuleShape(p1, p2, r))
    return out


def _read_sphere(data: bytes) -> List[SphereShape]:
    out = []
    for i in range(0, len(data), 16):
        b = data[i:i+16]
        c = struct.unpack("<fff", b[0:12])
        r, = struct.unpack("<f", b[12:16])
        out.append(SphereShape(c, r))
    return out


def _read_box(data: bytes) -> List[BoxShape]:
    out = []
    for i in range(0, len(data), 60):
        b = data[i:i+60]
        m = struct.unpack("<12f", b[0:48])
        c = struct.unpack("<fff", b[48:60])
        out.append(BoxShape(m, c))
    return out


def _read_welj(data: bytes) -> List[WeldJoint]:
    out = []
    for i in range(0, len(data), 104):
        b = data[i:i+104]
        fa = struct.unpack("<12f", b[0:48])
        fb = struct.unpack("<12f", b[48:96])
        af, ad = struct.unpack("<ff", b[96:104])
        out.append(WeldJoint(fa, fb, af, ad))
    return out


def _read_sphj(data: bytes) -> List[SphericalJoint]:
    out = []
    for i in range(0, len(data), 28):
        b = data[i:i+28]
        aa = struct.unpack("<fff", b[0:12])
        ab = struct.unpack("<fff", b[12:24])
        ft, = struct.unpack("<f", b[24:28])
        out.append(SphericalJoint(aa, ab, ft))
    return out


def _read_shoj(data: bytes) -> List[ShoulderJoint]:
    out = []
    for i in range(0, len(data), 108):
        b = data[i:i+108]
        fa = struct.unpack("<12f", b[0:48])
        fb = struct.unpack("<12f", b[48:96])
        lt, ut, ca = struct.unpack("<fff", b[96:108])
        out.append(ShoulderJoint(fa, fb, lt, ut, ca))
    return out


def _read_join(data: bytes) -> List[Joint]:
    out = []
    for i in range(0, len(data), 16):
        b = data[i:i+16]
        ba, bb = struct.unpack("<II", b[0:8])
        unk = b[8:12]
        jt, jid = struct.unpack("<hh", b[12:16])
        out.append(Joint(ba, bb, jt, jid, unk))
    return out


def read(data: bytes) -> PhysDoc:
    doc = PhysDoc()
    for magic, payload in _iter_chunks(data):
        if magic == "PHYS":
            doc.version, = struct.unpack("<h", payload[:2])
        elif magic == "BODY":
            doc.bodies = _read_body(payload)
        elif magic == "SHAP":
            doc.shapes = _read_shape(payload)
        elif magic == "CAPS":
            doc.capsules = _read_capsule(payload)
        elif magic == "SPHS":
            doc.spheres = _read_sphere(payload)
        elif magic == "BOXS":
            doc.boxes = _read_box(payload)
        elif magic == "WELJ":
            doc.weld_joints = _read_welj(payload)
        elif magic == "SPHJ":
            doc.spherical_joints = _read_sphj(payload)
        elif magic == "SHOJ":
            doc.shoulder_joints = _read_shoj(payload)
        elif magic == "JOIN":
            doc.joints = _read_join(payload)
        # PHYT/PHYV/BDY2+/SHP2/PLYT/WLJ2+/SHJ2/PRSJ/REVJ/DSTJ: not needed here.
    return doc
