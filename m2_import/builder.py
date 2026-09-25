"""Turn a parsed :class:`M2Model` into Blender data."""

import os
import re
import math
import time
import json
import base64
import bpy
from mathutils import Vector, Matrix

from . import anim_builder
from . import anim_data
from . import meta
from .anim_builder import BONE_LENGTH


def _store_metadata(root, model, fps):
    """Stash the non-geometry M2 tables on ``root`` so export can recover them."""
    if root is None:
        return
    root["m2_meta"] = json.dumps(meta.build(model, fps))


def _log(msg):
    """Print to Blender's system console so the import never looks frozen."""
    print("[M2] " + msg, flush=True)


def _convert_pos(p, mirror_x):
    # WoW model space and Blender are both right-handed, Z-up, so positions
    x, y, z = p
    if mirror_x:
        x = -x
    return (x, y, z)


# M2Texture.type. Anything non-zero is supplied by the client at runtime and
# carries no FileDataID, so it must keep its type through a round trip.
_TEXTURE_TYPE_NAMES = {
    1: "SkinComposited", 2: "ObjectSkin", 3: "WeaponBlade", 4: "WeaponHandle",
    5: "Environment", 6: "CharHair", 7: "CharFacialHair", 8: "SkinExtra",
    9: "UISkin", 10: "TaurenMane", 11: "MonsterSkin1", 12: "MonsterSkin2",
    13: "MonsterSkin3", 14: "ItemIcon",
}


def _batch_textures(model, batch):
    """The M2Texture rows a batch draws with, one per layer."""
    out = []
    for k in range(max(1, batch.texture_count)):
        idx = batch.texture_combo_index + k
        if not (0 <= idx < len(model.texture_lookup)):
            continue
        tex_id = model.texture_lookup[idx]
        if 0 <= tex_id < len(model.textures):
            out.append(model.textures[tex_id])
    return out


def _texname_for_batch(model, batch):
    """Resolve a display/material name for the texture behind a batch."""
    texes = _batch_textures(model, batch)
    if texes:
        fn = texes[0].filename
        if fn:
            return fn
        fid = meta.texture_file_id(texes[0])
        if fid:
            return "FileDataID_%d" % fid
        # No filename and no id: a runtime-composited texture. Name it by type
        # rather than "FileDataID_0", which reads as a broken id.
        label = _TEXTURE_TYPE_NAMES.get(texes[0].type)
        if label:
            return "M2_%s" % label
        return "M2_TextureType_%d" % texes[0].type
    return "material_%d" % batch.material_index


def _stamp_material(mat, model, batch):
    """Record the M2 render state on the Blender material."""
    if batch is None:
        return
    texes = _batch_textures(model, batch)
    if texes:
        # Always record BOTH, even when the ids are 0. A non-zero texture type
        fids = [meta.texture_file_id(t) for t in texes]
        mat["m2_texture_ids"] = ",".join(str(f) for f in fids)
        mat["m2_texture_types"] = ",".join(str(t.type) for t in texes)
    if texes and texes[0].filename:
        mat["m2_texture_filename"] = texes[0].filename
    mi = batch.material_index
    if 0 <= mi < len(model.materials):
        m2mat = model.materials[mi]
        mat["m2_render_flags"] = int(m2mat.flags)
        mat["m2_blend_mode"] = int(m2mat.blend_mode)
        # Mirror the blend into Blender's own setting so the viewport roughly
        # matches, without it being the source of truth on export.
        try:
            if m2mat.blend_mode >= 2:
                mat.blend_method = "BLEND"
            elif m2mat.blend_mode == 1:
                mat.blend_method = "CLIP"
        except Exception:  # noqa: BLE001 - property removed on newer Blender
            pass
    mat["m2_shader_id"] = int(batch.shader_id)
    mat["m2_material_layer"] = int(batch.material_layer)
    # Batch-level render data with no Blender equivalent, kept so export can
    # reproduce it: batch flags (retail: 0x10 on body batches, 0x80 on eye
    # glow), the colour-track index, and per layer the UV set, the texture
    # transform (the UV scroll that makes eye glow pulse) and the weight track.
    mat["m2_batch_flags"] = int(batch.flags)
    mat["m2_color_index"] = int(batch.color_index)
    coords, transforms, weights = [], [], []
    for k in range(max(1, batch.texture_count)):
        ci = batch.texture_coord_combo + k
        # No coord-combo entry (retail characters leave the array empty) means
        # layer k samples UV set k.
        coords.append(int(model.tex_coord_combos[ci]) if 0 <= ci < len(model.tex_coord_combos) else k)
        ti = batch.texture_transform_combo + k
        transforms.append(int(model.tex_transform_combos[ti])
                          if 0 <= ti < len(model.tex_transform_combos) else 0xFFFF)
        wi = batch.texture_weight_combo + k
        weights.append(int(model.tex_weight_combos[wi]) if 0 <= wi < len(model.tex_weight_combos) else 0)
    mat["m2_texture_coords"] = ",".join(str(c) for c in coords)
    # M2Texture.flags per layer: 0x1 wrap U, 0x2 wrap V. A scrolling glow layer
    # needs wrap or the scroll clamps at the texture edge.
    mat["m2_texture_flags"] = ",".join(str(int(t.flags)) for t in texes)
    mat["m2_texture_transforms"] = ",".join(str(t) for t in transforms)
    mat["m2_texture_weights"] = ",".join(str(w) for w in weights)
    try:
        from . import m2_ui
        m2_ui.sync_props_from_customprops(mat)
    except Exception as exc:  # noqa: BLE001 - never let panel sync break import
        _log("warning: could not sync panel props on %s: %r" % (mat.name, exc))


def _normalize_uv_tiles(uv_flat, faces, face_slot, slot_textures):
    """Shift every UV island by whole tiles onto the 0..1 grid.

    Retail meshes (hair especially) lay their UV islands several tiles away
    from the origin; with a wrapping texture that draws identically, but it is
    a mess to edit. A whole-tile shift changes nothing the client draws, so it
    is only done on axes where the island's texture wraps (M2Texture flags
    0x1 / 0x2). Islands are UV islands: faces joined only where they share an
    edge with the same UVs on both sides, since one hair mesh holds many
    separate strands in UV space.
    """
    import math
    nf = len(faces)
    parent = list(range(nf))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def key(v, li):
        return (v, round(uv_flat[2 * li], 4), round(uv_flat[2 * li + 1], 4))

    edge_owner = {}
    for fi, f in enumerate(faces):
        for c in range(3):
            a, b = key(f[c], 3 * fi + c), key(f[(c + 1) % 3], 3 * fi + (c + 1) % 3)
            ek = (a, b) if a <= b else (b, a)
            other = edge_owner.setdefault(ek, fi)
            if other != fi:
                ra, rb = find(other), find(fi)
                if ra != rb:
                    parent[rb] = ra
    sums = {}
    for fi in range(nf):
        acc = sums.setdefault(find(fi), [0.0, 0.0, 0, face_slot[fi]])
        for c in range(3):
            acc[0] += uv_flat[2 * (3 * fi + c)]
            acc[1] += uv_flat[2 * (3 * fi + c) + 1]
            acc[2] += 1
    shifts = {}
    for isl, (su, sv, n, slot) in sums.items():
        texes = slot_textures[slot] if 0 <= slot < len(slot_textures) else []
        flags = texes[0].flags if texes else 0x3
        du = math.floor(su / n) if flags & 0x1 else 0
        dv = math.floor(sv / n) if flags & 0x2 else 0
        if du or dv:
            shifts[isl] = (du, dv)
    moved = 0
    for fi in range(nf):
        sh = shifts.get(find(fi))
        if sh is None:
            continue
        moved += 1
        for c in range(3):
            uv_flat[2 * (3 * fi + c)] -= sh[0]
            uv_flat[2 * (3 * fi + c) + 1] -= sh[1]
    return moved


def _make_mesh_object(model, subs, name, mirror_x, batch_for_submesh,
                      arm_obj, collection, weld_seams=True, normalize_uvs=True,
                      repair_weights=True):
    """Build one mesh object from a list of (submesh_index, submesh) pairs."""
    skin = model.skin
    tri = skin.triangles
    vl = skin.vertices
    mverts = model.vertices

    verts = []            # blender vertex positions
    local_to_global = []  # blender vertex -> a representative M2 global index
    faces = []
    face_slot = []
    corner_globals = []   # per face-corner (loop) M2 global index, in face order

    if weld_seams:
        weld = {}         # (pos, bone_indices, bone_weights) -> blender vertex

        def local_of(global_idx):
            mv = mverts[global_idx]
            key = (_convert_pos(mv.pos, mirror_x), mv.bone_indices, mv.bone_weights)
            l = weld.get(key)
            if l is None:
                l = len(verts)
                weld[key] = l
                verts.append(key[0])
                local_to_global.append(global_idx)
            return l
    else:
        remap = {}

        def local_of(global_idx):
            l = remap.get(global_idx)
            if l is None:
                l = len(verts)
                remap[global_idx] = l
                local_to_global.append(global_idx)
                verts.append(_convert_pos(mverts[global_idx].pos, mirror_x))
            return l

    materials = []        # ordered (name, batch) pairs
    slot_lookup = {}      # name -> slot index

    def slot_for_submesh(si):
        b = batch_for_submesh.get(si)
        if b is None:
            mat_name = "material_unbatched"
        else:
            # Dedup key includes texture type + render state so a body geoset
            base = _texname_for_batch(model, b)
            texes = _batch_textures(model, b)
            first_type = texes[0].type if texes else 0
            state = "u"
            if 0 <= b.material_index < len(model.materials):
                mm = model.materials[b.material_index]
                state = "b%d.s%d.f%d.l%d" % (int(mm.blend_mode),
                                              int(b.shader_id),
                                              int(mm.flags),
                                              int(b.material_layer))
            mat_name = "%s_t%d_%s" % (base, first_type, state)
        if mat_name not in slot_lookup:
            slot_lookup[mat_name] = len(materials)
            materials.append((mat_name, b))
        return slot_lookup[mat_name]

    for si, sub in subs:
        slot = slot_for_submesh(si)
        start = sub.index_start
        end = start + sub.index_count
        for i in range(start, end, 3):
            if i + 2 >= len(tri):
                break
            g0, g1, g2 = vl[tri[i]], vl[tri[i + 1]], vl[tri[i + 2]]
            # Skip degenerate triangles (two corners share a vertex). They draw
            if g0 == g1 or g1 == g2 or g0 == g2:
                continue
            l0, l1, l2 = local_of(g0), local_of(g1), local_of(g2)
            # Welding can make a formerly-valid triangle degenerate (two corners
            # collapse to the same welded vertex); drop those too.
            if l0 == l1 or l1 == l2 or l0 == l2:
                continue
            faces.append((l0, l1, l2))
            face_slot.append(slot)
            corner_globals.extend((g0, g1, g2))

    if not faces:
        return None

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))

    for mat_name, batch in materials:
        mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        _stamp_material(mat, model, batch)
        mesh.materials.append(mat)
    for poly, slot in zip(mesh.polygons, face_slot):
        poly.material_index = slot

    n_loops = len(mesh.loops)
    uv_flat = [0.0] * (2 * n_loops)
    for i, g in enumerate(corner_globals):
        u, v = mverts[g].uv1
        uv_flat[2 * i] = u
        uv_flat[2 * i + 1] = 1.0 - v
    if normalize_uvs:
        _normalize_uv_tiles(uv_flat, faces, face_slot,
                            [_batch_textures(model, b) for _, b in materials])
    uv_layer = mesh.uv_layers.new(name="UVMap")
    uv_layer.data.foreach_set("uv", uv_flat)
    # Second UV set. Multi-layer materials (eye glow, eyeballs, tabards) sample
    # it with their second texture layer; single-layer geosets store zeros and
    # get no second map.
    if any(mverts[g].uv2 != (0.0, 0.0) for g in corner_globals):
        uv2_flat = [0.0] * (2 * n_loops)
        for i, g in enumerate(corner_globals):
            u, v = mverts[g].uv2
            uv2_flat[2 * i] = u
            uv2_flat[2 * i + 1] = 1.0 - v
        uv2_layer = mesh.uv_layers.new(name="UVMap2")
        uv2_layer.data.foreach_set("uv", uv2_flat)
        mesh.uv_layers.active_index = 0

    # Tag each vertex with a representative M2 global index. Export's in-place
    try:
        attr = mesh.attributes.new("m2_vidx", "INT", "POINT")
        attr.data.foreach_set("value", list(local_to_global))
    except Exception:  # noqa: BLE001
        pass

    # Per-loop custom normals from each corner's M2 vertex normal. Zero-length
    try:
        loop_normals = []
        for g in corner_globals:
            nx, ny, nz = mverts[g].normal
            if mirror_x:
                nx = -nx
            if nx == 0.0 and ny == 0.0 and nz == 0.0:
                nz = 1.0
            loop_normals.append((nx, ny, nz))
        setter = getattr(mesh, "normals_split_custom_set", None)
        if setter is not None:
            setter(loop_normals)
    except Exception as exc:  # noqa: BLE001
        _log("warning: could not set custom normals on %s: %r" % (name, exc))

    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)

    if arm_obj is not None:
        _apply_skinning(model, obj, arm_obj, local_to_global, faces=faces,
                        repair=repair_weights)

    return obj


def _repair_stray_weights(model, local_to_global, faces, log):
    """Weights for vertices bound to bones none of their neighbours use.

    Retail meshes occasionally leave a single vertex bound to the root bone in
    the middle of a part that is otherwise all one bone (a blood elf eye glow
    has one); it stays behind when the head moves. Such a vertex takes the
    averaged weights of the vertices it shares faces with. Returns
    {local vertex: [(bone, weight 0..255), ...]} for the repaired vertices.
    """
    verts = model.vertices

    def bones_of(g):
        v = verts[g]
        return {bi for bi, bw in zip(v.bone_indices, v.bone_weights) if bw}

    neighbours = {}
    for f in faces:
        for a in f:
            for b in f:
                if a != b:
                    neighbours.setdefault(a, set()).add(b)
    fixed = {}
    for local_idx, nbrs in neighbours.items():
        mine = bones_of(local_to_global[local_idx])
        if not mine:
            continue
        theirs = set()
        for nb in nbrs:
            theirs |= bones_of(local_to_global[nb])
        if not theirs or mine & theirs:
            continue
        acc = {}
        for nb in nbrs:
            v = verts[local_to_global[nb]]
            for bi, bw in zip(v.bone_indices, v.bone_weights):
                if bw:
                    acc[bi] = acc.get(bi, 0.0) + bw / len(nbrs)
        if acc:
            fixed[local_idx] = sorted(acc.items(), key=lambda kv: -kv[1])[:4]
    if fixed:
        log("repaired %d stray-weighted vertex(es) bound to a bone none of their "
            "neighbours use (e.g. root); they now follow their neighbours" % len(fixed))
    return fixed


def _apply_skinning(model, mesh_obj, arm_obj, local_to_global, faces=None, repair=True):
    """Create per-bone vertex groups for this object's vertices and bind it."""
    n_bones = len(model.bones)
    verts = model.vertices
    fixed = _repair_stray_weights(model, local_to_global, faces, _log) if (repair and faces) else {}
    # buckets[bone_index][raw_weight] -> list of local vertex indices
    buckets = {}
    for local_idx, global_idx in enumerate(local_to_global):
        if local_idx in fixed:
            for bi, bw in fixed[local_idx]:
                if 0 <= bi < n_bones and bw > 0:
                    buckets.setdefault(bi, {}).setdefault(bw / 255.0 * 255.0, []).append(local_idx)
            continue
        v = verts[global_idx]
        for bi, bw in zip(v.bone_indices, v.bone_weights):
            if bw == 0 or not (0 <= bi < n_bones):
                continue
            buckets.setdefault(bi, {}).setdefault(bw, []).append(local_idx)

    for bi in sorted(buckets):
        vg = mesh_obj.vertex_groups.new(name="bone_%03d" % bi)
        for bw, idxs in buckets[bi].items():
            vg.add(idxs, bw / 255.0, "REPLACE")

    mod = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
    mod.object = arm_obj
    mesh_obj.parent = arm_obj


def _build_armature(model, name, mirror_x, collection, bone_tilt=0.0):
    if not model.bones:
        return None

    arm_data = bpy.data.armatures.new(name + "_Armature")
    arm_obj = bpy.data.objects.new(name + "_Armature", arm_data)
    collection.objects.link(arm_obj)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")

    # By default every bone points along +Y with zero roll, so its rest matrix
    direction = Vector((0.0, 1.0, 0.0))
    if bone_tilt:
        direction = (Matrix.Rotation(math.radians(bone_tilt), 3, "Z")
                     @ direction).normalized()

    edit_bones = []
    for i, b in enumerate(model.bones):
        eb = arm_data.edit_bones.new("bone_%03d" % i)
        hx, hy, hz = _convert_pos(b.pivot, mirror_x)
        eb.head = (hx, hy, hz)
        eb.tail = (hx + direction.x * BONE_LENGTH,
                   hy + direction.y * BONE_LENGTH,
                   hz + direction.z * BONE_LENGTH)
        eb.roll = 0.0
        edit_bones.append(eb)

    for i, b in enumerate(model.bones):
        if 0 <= b.parent < len(edit_bones):
            edit_bones[i].parent = edit_bones[b.parent]

    bpy.ops.object.mode_set(mode="OBJECT")

    # Stash each bone's M2-only fields on the Bone itself (edit bones are gone
    for i, b in enumerate(model.bones):
        db = arm_data.bones.get("bone_%03d" % i)
        if db is None:
            continue
        db["m2_key_bone_id"] = int(b.key_bone_id)
        db["m2_bone_flags"] = int(b.flags)
        db["m2_name_crc"] = str(getattr(b, "name_crc", 0) & 0xFFFFFFFF)
        db["m2_submesh_id"] = int(getattr(b, "submesh_id", 0))

    _log("armature: %d bones" % len(edit_bones))
    return arm_obj


def _build_bounds_box(model, name, mirror_x, collection, kind="bounding"):
    """An editable wireframe box: the render bounds ("bounding") or the
    collision box ("collision"). The client frames the character and transmog
    screens from the collision box, so that is the one to move or resize to
    change where those cameras look."""
    if kind == "collision":
        mn, mx = model.collision_min, model.collision_max
        if mn is None or mx is None or all(mx[i] - mn[i] <= 1e-6 for i in range(3)):
            return None
    else:
        mn, mx = model.bounding_min, model.bounding_max
    if mn is None or mx is None:
        if not model.vertices:
            return None
        xs = [v.pos[0] for v in model.vertices]
        ys = [v.pos[1] for v in model.vertices]
        zs = [v.pos[2] for v in model.vertices]
        mn = (min(xs), min(ys), min(zs))
        mx = (max(xs), max(ys), max(zs))
    c0 = _convert_pos(mn, mirror_x)
    c1 = _convert_pos(mx, mirror_x)
    lo = (min(c0[0], c1[0]), min(c0[1], c1[1]), min(c0[2], c1[2]))
    hi = (max(c0[0], c1[0]), max(c0[1], c1[1]), max(c0[2], c1[2]))
    verts = [(lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]),
             (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
             (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]),
             (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2])]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    label = "M2_CollisionBox" if kind == "collision" else "M2_BoundingBox"
    me = bpy.data.meshes.new(name + "_" + label[3:])
    me.from_pydata(verts, edges, [])
    me.update()
    obj = bpy.data.objects.new(label, me)
    obj["m2_collision_box" if kind == "collision" else "m2_bounding_box"] = 1
    obj.display_type = "WIRE"
    obj.hide_render = True
    obj.show_in_front = True
    collection.objects.link(obj)
    return obj


def build(model, *, import_armature=True, mirror_x=False,
          split_geosets=True, import_animations=True,
          import_attachments=True, import_cameras=True, fps=30, max_animations=0,
          collection=None, bone_tilt=0.0, weld_seams=True, import_bounds=True,
          normalize_uvs=True, repair_weights=True,
          import_events=True):
    """Create Blender objects from ``model``."""
    if collection is None:
        collection = bpy.context.scene.collection

    name = model.name or os.path.splitext(
        os.path.basename(model.source_path or "M2_Model"))[0] or "M2_Model"

    wm = bpy.context.window_manager
    t0 = time.perf_counter()
    _log("building '%s' (%d verts, %d bones, %d sequences, %d attachments)"
         % (name, len(model.vertices), len(model.bones),
            len(model.sequences), len(model.attachments)))
    try:
        wm.progress_begin(0, 100)
    except Exception:  # noqa: BLE001 - headless / no window manager
        wm = None

    def _progress(p):
        if wm is not None:
            try:
                wm.progress_update(p)
            except Exception:  # noqa: BLE001
                pass

    skin = model.skin
    if skin is None:
        # Nothing to build geometry from; still surface an empty for the user.
        _log("no skin/geometry found: creating an empty placeholder")
        empty = bpy.data.objects.new(name, None)
        collection.objects.link(empty)
        if wm is not None:
            wm.progress_end()
        return empty

    arm_obj = None
    if import_armature:
        arm_obj = _build_armature(model, name, mirror_x, collection,
                                  bone_tilt=bone_tilt)
    _progress(15)

    # First batch referencing each submesh decides its material.
    batch_for_submesh = {}
    for b in skin.batches:
        batch_for_submesh.setdefault(b.submesh_index, b)

    created = []
    if split_geosets:
        n = len(skin.submeshes)
        _log("building %d geosets" % n)
        for si, sub in enumerate(skin.submeshes):
            obj_name = "%s_geoset_%04d" % (name, sub.skin_section_id)
            obj = _make_mesh_object(
                model, [(si, sub)], obj_name, mirror_x,
                batch_for_submesh, arm_obj, collection, weld_seams=weld_seams, normalize_uvs=normalize_uvs, repair_weights=repair_weights)
            if obj is not None:
                obj["m2_order"] = si                      # submesh index in skin
                obj["m2_skin_section_id"] = sub.skin_section_id
                # Split the id into WoW's group/variant convention so the M2
                # panel can show/edit them without exposing raw ints. id = g*100 + v
                obj["m2_geoset_group"] = int(sub.skin_section_id) // 100
                obj["m2_geoset_variant"] = int(sub.skin_section_id) % 100
                obj["m2_center_bone"] = sub.center_bone_index
                created.append(obj)
            if n:
                _progress(15 + int(45 * (si + 1) / n))
    else:
        _log("building merged mesh")
        obj = _make_mesh_object(
            model, list(enumerate(skin.submeshes)), name, mirror_x,
            batch_for_submesh, arm_obj, collection, weld_seams=weld_seams, normalize_uvs=normalize_uvs, repair_weights=repair_weights)
        if obj is not None:
            created.append(obj)
        _progress(60)
    _log("geometry done (%d objects)" % len(created))

    # Group the parts so they move/select together.
    root = arm_obj
    if root is None:
        if split_geosets and len(created) > 1:
            root = bpy.data.objects.new(name, None)   # empty parent
            collection.objects.link(root)
            for obj in created:
                obj.parent = root
        elif created:
            root = created[0]

    # Attachments and animations both need the armature for proper placement.
    attachments = []
    if import_attachments:
        attachments = anim_builder.build_attachments(
            model, arm_obj, name, mirror_x, collection, _log)
    if import_cameras:
        anim_builder.build_cameras(model, name, mirror_x, collection, fps, _log)
    if import_events:
        anim_builder.build_events(model, arm_obj, name, mirror_x, collection, _log)
        # Fill in standard events the model is missing so triggers work.
        anim_builder.add_missing_events(model, arm_obj, name, mirror_x,
                                        collection, _log)
    if import_bounds:
        if _build_bounds_box(model, name, mirror_x, collection) is not None:
            _log("created editable render bounds 'M2_BoundingBox'")
        if _build_bounds_box(model, name, mirror_x, collection, kind="collision") is not None:
            _log("created editable collision box 'M2_CollisionBox' (character-screen framing)")
    _progress(70)

    if import_animations:
        anim_builder.build_animations(
            model, arm_obj, mirror_x, fps, _log,
            max_animations=max_animations, name=name)
    _progress(95)

    # Preserve the tables export needs to reproduce the file exactly.
    _store_metadata(root, model, fps)

    # Selection feedback.
    bpy.ops.object.select_all(action="DESELECT")
    for obj in created:
        obj.select_set(True)
    if created:
        bpy.context.view_layer.objects.active = created[0]
    elif root is not None:
        root.select_set(True)
        bpy.context.view_layer.objects.active = root

    if wm is not None:
        wm.progress_end()
    _log("done in %.2fs" % (time.perf_counter() - t0))

    return root if root is not None else (created[0] if created else None)
