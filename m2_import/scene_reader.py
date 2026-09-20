"""Reconstruct an :class:`M2Model` from the imported Blender scene."""

import os
import json
import base64
import re

import bpy
from mathutils import Vector

from . import meta as meta_mod

# Geoset id embedded in the object name, e.g. "nightelffemale_hd_geoset_3202"
# (Blender's ".001" duplicate suffix is ignored since \d+ stops at the dot).
_GEOSET_NAME_RE = re.compile(r"geoset[_-]?(\d+)", re.IGNORECASE)


def geoset_id_from_name(name):
    """'nightelffemale_hd_geoset_3202' -> 3202, or None."""
    m = _GEOSET_NAME_RE.search(name or "")
    return int(m.group(1)) if m else None


# Back-compat alias; from_scene imports the public name.
_geoset_id_from_name = geoset_id_from_name

from .model import (
    M2Model, M2Vertex, M2Texture, M2Material, M2Bone,
    M2Sequence, M2Attachment, M2AnimTrack, M2SkinProfile, M2SubMesh, M2Batch,
)
from . import anim_data
from .anim_builder import BONE_LENGTH


class ExportError(Exception):
    pass


def read_vertex_updates(objects, mirror_x=False):
    """Collect ``{m2_global_index: (pos, normal, uv1)}`` from imported geosets."""
    updates = {}
    geosets = [o for o in objects
               if o.type == "MESH" and "m2_vidx" in o.data.attributes]
    for obj in geosets:
        mesh = obj.data
        mw = obj.matrix_world
        nmat = mw.to_3x3()
        n = len(mesh.vertices)
        vidx = [0] * n
        mesh.attributes["m2_vidx"].data.foreach_get("value", vidx)
        uv_layer = mesh.uv_layers.active
        vert_loop = [-1] * n
        for loop in mesh.loops:
            if vert_loop[loop.vertex_index] < 0:
                vert_loop[loop.vertex_index] = loop.index
        corner_normals = _corner_normals(mesh)
        for vi, mv in enumerate(mesh.vertices):
            gidx = vidx[vi]
            co = mw @ mv.co
            pos = (-co.x, co.y, co.z) if mirror_x else (co.x, co.y, co.z)
            ln = vert_loop[vi]
            if corner_normals is not None and ln >= 0:
                nv = nmat @ Vector(corner_normals[ln])
                nv.normalize()
                nrm = (-nv.x, nv.y, nv.z) if mirror_x else (nv.x, nv.y, nv.z)
            else:
                nv = nmat @ mv.normal
                nrm = (nv.x, nv.y, nv.z)
            uv = (0.0, 0.0)
            if uv_layer is not None and ln >= 0:
                u, vv = uv_layer.data[ln].uv
                uv = (u, 1.0 - vv)
            updates[gidx] = (pos, nrm, uv)
    return updates


def _bone_index(name):
    """'bone_007' -> 7, or None."""
    if name.startswith("bone_"):
        try:
            return int(name[5:])
        except ValueError:
            return None
    return None


def _find_meta_root(objects):
    for o in objects:
        if "m2_meta" in o.keys():
            return o
    # Fall back to a scene-wide search: the marker lives on the imported root
    # (armature / parent empty), which the user may not have selected.
    for o in bpy.data.objects:
        if "m2_meta" in o.keys():
            return o
    return None


def _meta_from_source(source_m2, fps):
    """Re-derive the metadata by re-reading the original .m2."""
    from . import factory
    model = factory.load_model(source_m2)
    out = meta_mod.build(model, fps)
    out["source_path"] = source_m2
    return out


def read_scene(objects, mirror_x=False, source_m2="", fps=30.0, tried=()):
    """Build an M2Model from the given Blender objects (e.g. a collection)."""
    objects = list(objects)
    meta = None
    if source_m2 and os.path.isfile(source_m2):
        try:
            meta = _meta_from_source(source_m2, fps)
            print("[M2] metadata re-read from %s (no stored data used)"
                  % os.path.basename(source_m2), flush=True)
        except Exception as exc:  # noqa: BLE001 - fall back to the stored blob
            print("[M2] could not read %s (%s); falling back to stored metadata"
                  % (source_m2, exc), flush=True)
    if meta is None:
        root = _find_meta_root(objects)
        if root is None:
            detail = ("  Tried: " + "; ".join(tried) + ".") if tried else \
                "  No source .m2 path was given."
            raise ExportError(
                "Can't find the M2 tables (textures/materials/batches). These "
                "have no Blender equivalent, so they must come either from the "
                "original .m2 or from an import done with this add-on.\n"
                + detail +
                "\n  Fix: set 'Source M2 (read tables from)' in the export "
                "panel to the .m2 this scene came from.")
        meta = json.loads(root["m2_meta"])
    fps = float(meta.get("fps", 30)) or 30.0

    model = M2Model()
    model.name = meta.get("name", "")
    model.version = int(meta.get("version", 274))
    model.expansion = "LEGION"
    model.source_path = meta.get("source_path", "") or ""
    model.global_loops = list(meta.get("global_loops", []))
    model.seq_lookup = list(meta.get("seq_lookup", []))
    model.skin_file_ids = list(meta.get("skin_file_ids", []))
    model.aux_chunks = {name: base64.b64decode(b64)
                        for name, b64 in meta.get("aux_chunks", {}).items()}
    tex_fids = meta.get("texture_file_ids", [])
    model.textures = [M2Texture(t["type"], t["flags"], t["filename"])
                      for t in meta.get("textures", [])]
    for i, tx in enumerate(model.textures):
        if i < len(tex_fids):
            tx.file_data_id = tex_fids[i]
    model.materials = [M2Material(m["flags"], m["blend"])
                       for m in meta.get("materials", [])]
    model.texture_lookup = list(meta.get("texture_lookup", []))
    model.n_colors = meta.get("n_colors", 0)
    model.n_texture_weights = meta.get("n_texture_weights", 0)
    model.n_texture_transforms = meta.get("n_texture_transforms", 0)
    model.tex_coord_combos = list(meta.get("tex_coord_combos", []))
    model.tex_weight_combos = list(meta.get("tex_weight_combos", []))
    model.tex_transform_combos = list(meta.get("tex_transform_combos", []))
    model.texture_indices_by_id = list(meta.get("texture_indices_by_id", []))
    model.colors = [(anim_data.track_from_json(c), anim_data.track_from_json(a))
                    for c, a in meta.get("colors", [])]
    model.weights = [anim_data.track_from_json(w) for w in meta.get("weights", [])]
    model.transforms = [(anim_data.track_from_json(t), anim_data.track_from_json(r),
                         anim_data.track_from_json(s))
                        for t, r, s in meta.get("transforms", [])]

    sequences = []
    for s in meta.get("sequences", []):
        seq = M2Sequence()
        seq.id = s["id"]; seq.variation_index = s["var"]
        seq.duration = s["dur"]; seq.flags = s["flags"]
        seq.end_timestamp = s["dur"]
        seq.movespeed = float(s.get("movespeed", 0.0))
        seq.blend_time_in = int(s.get("blend_in", 150))
        seq.blend_time_out = int(s.get("blend_out", 0))
        sequences.append(seq)
    model.sequences = sequences
    nseq = len(sequences)

    arm = next((o for o in objects if o.type == "ARMATURE"), None)
    _read_bones(model, arm, meta)
    _read_animations(model, objects, nseq, fps, mirror_x)
    _read_attachments(model, meta)
    _read_geometry(model, objects, meta, mirror_x)
    from .from_scene import bounds_override_from_objects
    model.bounding_override = bounds_override_from_objects(objects, mirror_x)
    if model.bounding_override:
        print("[M2] using edited bounding box from 'M2_BoundingBox' object",
              flush=True)
    return model


# ---------------------------------------------------------------------------
def _read_bones(model, arm, meta):
    if arm is None:
        return
    data_bones = arm.data.bones
    indexed = {}
    for b in data_bones:
        i = _bone_index(b.name)
        if i is not None:
            indexed[i] = b
    if not indexed:
        return
    n = max(indexed) + 1
    meta_bones = meta.get("bones", [])
    bones = []
    for i in range(n):
        mb = M2Bone()
        db = indexed.get(i)
        if db is not None:
            h = db.head_local
            mb.pivot = (h.x, h.y, h.z)
            parent = db.parent
            mb.parent = _bone_index(parent.name) if parent else -1
            if mb.parent is None:
                mb.parent = -1
        if i < len(meta_bones):
            mb.key_bone_id = meta_bones[i]["key_bone_id"]
            mb.flags = meta_bones[i]["flags"]
            mb.submesh_id = meta_bones[i].get("submesh_id", 0)
            mb.name_crc = meta_bones[i].get("name_crc", 0)
        bones.append(mb)
    model.bones = bones


# ---------------------------------------------------------------------------
def _conv_pos_inv(v, mirror_x):
    return (-v[0], v[1], v[2]) if mirror_x else (v[0], v[1], v[2])


def _conv_quat_inv(wxyz, mirror_x):
    # Blender (w,x,y,z) -> M2 (x,y,z,w), undoing the X-mirror if applied.
    w, x, y, z = wxyz
    if mirror_x:
        return (x, -y, -z, w)
    return (x, y, z, w)


def _read_animations(model, objects, nseq, fps, mirror_x):
    if not model.bones or nseq == 0:
        return
    # One track triple per bone, each a per-sequence timeline list.
    tracks = {}
    for i in range(len(model.bones)):
        tr = (M2AnimTrack(), M2AnimTrack(), M2AnimTrack())
        for t in tr:
            t.timelines = [[] for _ in range(nseq)]
        tracks[i] = tr

    scale = 1000.0 / fps
    seen = False
    for action in bpy.data.actions:
        if "m2_seq_index" not in action.keys():
            continue
        si = int(action["m2_seq_index"])
        if not (0 <= si < nseq):
            continue
        seen = True
        for bi, comps in _iter_action_bone_channels(action):
            if bi not in tracks:
                continue
            tr_t, tr_r, tr_s = tracks[bi]
            if "location" in comps:
                tr_t.timelines[si] = [
                    (round(f * scale), _conv_pos_inv(v, mirror_x))
                    for f, v in _zip_components(comps["location"], 3)]
                tr_t.interpolation_type = comps["location"][0][1]
            if "rotation_quaternion" in comps:
                tr_r.timelines[si] = [
                    (round(f * scale), _conv_quat_inv(v, mirror_x))
                    for f, v in _zip_components(comps["rotation_quaternion"], 4)]
                tr_r.interpolation_type = comps["rotation_quaternion"][0][1]
            if "scale" in comps:
                tr_s.timelines[si] = [
                    (round(f * scale), v)
                    for f, v in _zip_components(comps["scale"], 3)]
                tr_s.interpolation_type = comps["scale"][0][1]

    # Global-sequence clips: each carries a single looping timeline that is
    # independent of the regular animations.
    for action in bpy.data.actions:
        if "m2_global_sequence" not in action.keys():
            continue
        gs = int(action["m2_global_sequence"])
        seen = True
        for bi, comps in _iter_action_bone_channels(action):
            if bi not in tracks:
                continue
            tr_t, tr_r, tr_s = tracks[bi]
            for prop, tr, n in (("location", tr_t, 3),
                                ("rotation_quaternion", tr_r, 4),
                                ("scale", tr_s, 3)):
                if prop not in comps:
                    continue
                conv = (lambda v: _conv_pos_inv(v, mirror_x)) if prop == "location" \
                    else (lambda v: _conv_quat_inv(v, mirror_x)) if n == 4 \
                    else (lambda v: v)
                keys = [(round(f * scale), conv(v))
                        for f, v in _zip_components(comps[prop], n)]
                tr.global_sequence = gs
                tr.timelines = [keys]            # single global timeline
                tr.interpolation_type = comps[prop][0][1]

    if not seen:
        return
    for i, b in enumerate(model.bones):
        tr_t, tr_r, tr_s = tracks[i]
        b.translation = tr_t if (any(tr_t.timelines) or tr_t.global_sequence >= 0) else None
        b.rotation = tr_r if (any(tr_r.timelines) or tr_r.global_sequence >= 0) else None
        b.scale = tr_s if (any(tr_s.timelines) or tr_s.global_sequence >= 0) else None


def _iter_action_bone_channels(action):
    """Yield ``(bone_index, {prop: [(fcurve, interp_code), ...]})`` per bone."""
    fcurves = _action_fcurves(action)
    by_bone = {}
    for fc in fcurves:
        dp = fc.data_path
        if not dp.startswith('pose.bones["'):
            continue
        try:
            bname = dp.split('"')[1]
            prop = dp.rsplit(".", 1)[1]
        except IndexError:
            continue
        bi = _bone_index(bname)
        if bi is None or prop not in ("location", "rotation_quaternion", "scale"):
            continue
        interp = 0 if (len(fc.keyframe_points)
                       and fc.keyframe_points[0].interpolation == "CONSTANT") else 1
        by_bone.setdefault(bi, {}).setdefault(prop, []).append((fc, interp))
    for bi, props in by_bone.items():
        # order component fcurves by array_index
        ordered = {}
        for prop, lst in props.items():
            lst.sort(key=lambda fi: fi[0].array_index)
            ordered[prop] = lst
        yield bi, ordered


def _zip_components(fcurve_list, n_comp):
    """Yield ``(frame, value_tuple)`` from a property's component F-Curves."""
    if not fcurve_list:
        return
    fcs = [fi[0] for fi in fcurve_list]
    # ensure we have n_comp curves indexed by array_index
    by_idx = {fc.array_index: fc for fc in fcs}
    base = by_idx.get(0) or fcs[0]
    for kp in base.keyframe_points:
        frame = kp.co[0]
        vals = []
        for c in range(n_comp):
            fc = by_idx.get(c)
            if fc is None:
                vals.append(0.0)
            elif fc is base:
                vals.append(kp.co[1])
            else:
                vals.append(fc.evaluate(frame))
        yield frame, tuple(vals)


def _action_fcurves(action):
    """Return an action's F-Curves across the legacy and layered APIs."""
    try:
        return list(action.fcurves)
    except Exception:  # noqa: BLE001 - layered (4.4+/5.0) actions
        out = []
        for layer in action.layers:
            for strip in layer.strips:
                for cbag in strip.channelbags:
                    out.extend(cbag.fcurves)
        return out


# ---------------------------------------------------------------------------
def _read_attachments(model, meta):
    out = []
    for a in meta.get("attachments", []):
        att = M2Attachment()
        att.id = a["id"]; att.bone = a["bone"]
        att.position = tuple(a["pos"])
        out.append(att)
    model.attachments = out
    model.attachment_lookup = list(meta.get("attachment_lookup", []))


# ---------------------------------------------------------------------------
def _read_geometry(model, objects, meta, mirror_x):
    # EVERY mesh is exported (filtering on m2_order, as this used to, silently
    def _order_key(o):
        gid = geoset_id_from_name(o.name)
        if gid is None:
            gid = o.get("m2_skin_section_id")
        gid = int(gid) if gid is not None else 0
        original = int(o["m2_order"]) if "m2_order" in o.keys() else 1 << 30
        return (gid, original, o.name)

    geosets = sorted((o for o in objects
                      if o.type == "MESH" and not o.get("m2_bounding_box")),
                     key=_order_key)

    vertices = []        # global M2Vertex list
    skin_verts = []      # identity map (== range(len(vertices)))
    triangles = []
    submeshes = []
    # original submesh index -> [new indices]. One-to-MANY: duplicating a geoset
    # in Blender copies its m2_order, so several objects can share one origin.
    order_to_new = {}

    bone_groups = {}     # per object cache of group-index -> bone-index

    for new_idx, obj in enumerate(geosets):
        mesh = obj.data
        mw = obj.matrix_world
        nmat = mw.to_3x3()
        vbase = len(vertices)

        # group index -> bone index for this object
        g2b = {}
        for gi, vg in enumerate(obj.vertex_groups):
            bi = _bone_index(vg.name)
            if bi is not None:
                g2b[gi] = bi
        bone_groups[obj] = g2b

        # per-vertex UV and normal (take the first loop touching each vertex)
        uv_layer = mesh.uv_layers.active
        vert_loop = [-1] * len(mesh.vertices)
        for loop in mesh.loops:
            if vert_loop[loop.vertex_index] < 0:
                vert_loop[loop.vertex_index] = loop.index
        corner_normals = _corner_normals(mesh)

        for vi, mv in enumerate(mesh.vertices):
            co = mw @ mv.co
            v = M2Vertex()
            v.pos = (co.x, co.y, co.z)
            ln = vert_loop[vi]
            if uv_layer is not None and ln >= 0:
                u, vv = uv_layer.data[ln].uv
                v.uv1 = (u, 1.0 - vv)
            if corner_normals is not None and ln >= 0:
                n = nmat @ Vector(corner_normals[ln])
                n.normalize()
                v.normal = (n.x, n.y, n.z)
            else:
                n = nmat @ mv.normal
                v.normal = (n.x, n.y, n.z)
            v.bone_weights, v.bone_indices = _vertex_weights(mv, g2b)
            vertices.append(v)
            skin_verts.append(len(skin_verts))

        # triangles (mesh is already triangulated on import)
        istart = len(triangles)
        for poly in mesh.polygons:
            vs = poly.vertices
            if len(vs) != 3:
                continue
            triangles.extend(vbase + vs[k] for k in range(3))

        sub = M2SubMesh()
        # Geoset id: the object NAME wins (so you can set/fix it by renaming, and
        # it survives editing/joining), then the stored property, then index.
        name_id = _geoset_id_from_name(obj.name)
        if name_id is not None:
            sub.skin_section_id = name_id
        else:
            sub.skin_section_id = int(obj.get("m2_skin_section_id", new_idx))
        sub.center_bone_index = int(obj.get("m2_center_bone", 0))
        sub.vertex_start = vbase
        sub.vertex_count = len(vertices) - vbase
        sub.index_start = istart
        sub.index_count = len(triangles) - istart
        submeshes.append(sub)
        order_to_new.setdefault(int(obj.get("m2_order", new_idx)), []).append(new_idx)

    if len(skin_verts) > 0xFFFF:
        raise ExportError(
            "Model has %d skin vertices; the .skin format is limited to 65535. "
            "Reduce/merge geometry before export." % len(skin_verts))

    # Rebuild batches from metadata, remapping submesh indices to the new order.
    def _batch_from(mb, submesh_index):
        b = M2Batch()
        b.flags = mb["flags"]; b.shader_id = mb["shader_id"]
        b.submesh_index = submesh_index
        b.geoset_index = mb.get("geoset_index", 0)
        b.material_index = mb["material_index"]
        b.texture_combo_index = mb["texture_combo_index"]
        b.color_index = mb["color_index"]
        b.material_layer = mb["material_layer"]
        b.texture_count = mb["texture_count"]
        b.texture_coord_combo = mb.get("coord_combo", 0)
        b.texture_weight_combo = mb.get("weight_combo", 0)
        b.texture_transform_combo = mb.get("transform_combo", 0)
        return b

    batches = []
    meta_batches = meta.get("batches", [])
    for mb in meta_batches:
        # One source batch can map to several submeshes: duplicating a geoset in
        # Blender copies its m2_order, so both copies legitimately share it.
        for new_idx in order_to_new.get(mb["submesh_index"], ()):
            batches.append(_batch_from(mb, new_idx))

    # Safety net: give any uncovered submesh a batch, cloned from one belonging
    covered = {b.submesh_index for b in batches}
    missing = [i for i in range(len(submeshes)) if i not in covered]
    if missing and meta_batches:
        by_geoset = {}
        for b in batches:
            by_geoset.setdefault(submeshes[b.submesh_index].skin_section_id, b)
        for i in missing:
            if not batches:
                break
            template = by_geoset.get(submeshes[i].skin_section_id) or batches[0]
            nb = M2Batch()
            for slot in M2Batch.__slots__:
                setattr(nb, slot, getattr(template, slot))
            nb.submesh_index = i
            batches.append(nb)
        names = ", ".join(geosets[i].name for i in missing if i < len(geosets))
        print("[M2] WARNING: %d submesh(es) had no batch and would have been "
              "INVISIBLE in-game; generated one each: %s" % (len(missing), names),
              flush=True)
    elif missing:
        print("[M2] WARNING: %d submesh(es) have no batch and will be invisible "
              "in-game (no batch metadata to clone from)." % len(missing), flush=True)

    batches.sort(key=lambda b: (b.submesh_index, b.material_layer))

    _apply_material_overrides(model, geosets, batches)

    skin = M2SkinProfile()
    skin.vertices = skin_verts
    skin.triangles = triangles
    skin.submeshes = submeshes
    skin.batches = batches
    model.skin = skin
    model.vertices = vertices

    # Log the geoset id assigned to each object so it can be verified in the
    # System Console (Window > Toggle System Console).
    print("[M2] geoset id assignment (object name -> skin_section_id):", flush=True)
    for o, s in zip(geosets, submeshes):
        src = "name" if _geoset_id_from_name(o.name) is not None else "prop/index"
        print("   %-40s -> %d  (from %s)" % (o.name, s.skin_section_id, src), flush=True)


def _apply_material_overrides(model, geosets, batches):
    """Let a configured Blender material override the stored texture binding."""
    from .from_scene import _material_is_configured, _resolve_material

    by_sub = {}
    for b in batches:
        by_sub.setdefault(b.submesh_index, []).append(b)

    changed = []
    for new_idx, obj in enumerate(geosets):
        mats = [m for m in obj.data.materials if m is not None]
        bmat = mats[0] if mats else None
        if not _material_is_configured(bmat):
            continue
        targets = by_sub.get(new_idx)
        if not targets:
            continue
        spec = _resolve_material(bmat, 0)
        if spec.count != 1:
            continue

        fid, ttype = spec.tex_ids[0], spec.tex_types[0]
        ti = next((k for k, t in enumerate(model.textures)
                   if t.type == ttype and t.file_data_id == fid), None)
        if ti is None:
            model.textures.append(M2Texture(ttype, 0, "", fid))
            model.textures[-1].file_data_id = fid
            ti = len(model.textures) - 1
        mi = next((k for k, m in enumerate(model.materials)
                   if m.flags == spec.flags and m.blend_mode == spec.blend), None)
        if mi is None:
            model.materials.append(M2Material(spec.flags, spec.blend))
            mi = len(model.materials) - 1

        combo = len(model.texture_lookup)
        model.texture_lookup.append(ti)
        model.tex_coord_combos.append(0)
        # Point the weight/transform slices at whatever this batch already used,
        # so they stay inside arrays sized by the source model.
        model.tex_weight_combos.append(
            targets[0].texture_weight_combo
            if targets[0].texture_weight_combo < len(model.tex_weight_combos) else 0)
        model.tex_transform_combos.append(0xFFFF)

        for b in targets:
            b.texture_combo_index = combo
            b.texture_coord_combo = combo
            b.texture_count = 1
            b.material_index = mi
            b.shader_id = spec.shader
        changed.append((obj.name, ttype, fid, spec.blend))

    if changed:
        print("[M2] applied %d Blender material override(s):" % len(changed),
              flush=True)
        for name, tt, fid, bl in changed[:8]:
            print("   %-38s -> texture type=%d id=%d blend=%d"
                  % (name, tt, fid, bl), flush=True)
        if len(changed) > 8:
            print("   ... and %d more" % (len(changed) - 8), flush=True)
    return changed


def _corner_normals(mesh):
    try:
        cn = mesh.corner_normals
        return [tuple(c.vector) for c in cn]
    except Exception:  # noqa: BLE001 - not available / no custom normals
        return None


def _vertex_weights(mv, g2b):
    pairs = []
    for g in mv.groups:
        bi = g2b.get(g.group)
        if bi is None or g.weight <= 0.0:
            continue
        pairs.append((bi, g.weight))
    pairs.sort(key=lambda p: p[1], reverse=True)
    pairs = pairs[:4]
    weights = [0, 0, 0, 0]
    indices = [0, 0, 0, 0]
    for k, (bi, w) in enumerate(pairs):
        weights[k] = max(0, min(255, int(round(w * 255))))
        indices[k] = bi
    return tuple(weights), tuple(indices)
