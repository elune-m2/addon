"""Retarget M2I: move another rig's meshes onto the imported M2 skeleton.

Two armatures take part:

* the **M2 rig** (imported by this add-on: ``bone_NNN`` names, ``m2_*``
  properties on the geoset meshes, configured materials), and
* the **other rig** (the same skeleton with other bone names, e.g. an .m2i
  round trip through another tool, carrying edited meshes with no M2 data).

The retarget

1. matches the other rig's bones to the M2 rig's (by order; by rest
   position when the bone counts differ) and renames them to the M2
   names, so vertex groups and bone parents line up,
2. moves every child object of the other rig under the M2 rig (parent and
   armature modifier),
3. for each moved mesh finds the M2 geoset mesh with the same shape and
   copies its ``m2_*`` properties, materials (per face) and name onto it.

Everything here is plain functions over Blender objects; ``m2_ui`` wraps
them in an operator and a panel.
"""
import bpy
from mathutils import Vector, Matrix
from mathutils.kdtree import KDTree

# Custom properties that belong to the geometry itself and must not be
# copied between meshes of a different vertex order.
_SKIP_PROPS = {"m2_vidx"}


class RetargetError(Exception):
    pass


# ---------------------------------------------------------------------------
# bones
def _bone_ends_world(arm_obj):
    """{bone name: (head, tail)} in world space, from the rest pose."""
    mw = arm_obj.matrix_world
    return {b.name: (mw @ b.head_local, mw @ b.tail_local) for b in arm_obj.data.bones}


def match_bones(src_arm, dst_arm, method="ORDER", max_distance=0.05):
    """Map M2I-rig bone name -> M2-rig bone name.

    ``ORDER`` (the default) pairs the two bone lists index by index, which
    is how an M2I round trip keeps them; it needs the same bone count.
    ``POSITION`` pairs bones by rest head (and tail) position, closest pairs
    first, each bone used once; a bone already carrying an M2 name keeps it.
    Returns (mapping, unmatched_dst_names)."""
    src_bones = list(src_arm.data.bones)
    dst_bones = list(dst_arm.data.bones)
    mapping = {}
    if method == "ORDER":
        if len(src_bones) != len(dst_bones):
            raise RetargetError("Order matching needs the same bone count (%d vs %d)"
                                % (len(src_bones), len(dst_bones)))
        for s, d in zip(src_bones, dst_bones):
            mapping[d.name] = s.name
        return mapping, []

    src_pos = _bone_ends_world(src_arm)
    dst_pos = _bone_ends_world(dst_arm)
    src_names = {b.name for b in src_bones}
    taken = set()
    # Same name on both rigs: that is the answer, no geometry needed.
    for d in dst_bones:
        if d.name in src_names:
            mapping[d.name] = d.name
            taken.add(d.name)

    def children_of(bones):
        out = {None: []}
        for b in bones:
            out.setdefault(b.parent.name if b.parent else None, []).append(b.name)
        return out
    src_kids, dst_kids = children_of(src_bones), children_of(dst_bones)
    sib_index = {}
    for kids in list(src_kids.values()) + list(dst_kids.values()):
        for i, n in enumerate(kids):
            sib_index[n] = i

    def cost(dn, sn):
        dh, dt = dst_pos[dn]
        sh, st = src_pos[sn]
        return (dh - sh).length + 0.5 * (dt - st).length

    def assign(dnames, snames):
        pairs = []
        for dn in dnames:
            if dn in mapping:
                continue
            for sn in snames:
                if sn in taken:
                    continue
                c = cost(dn, sn)
                if c <= max_distance:
                    # Ties (bones sharing a pivot) break on sibling order.
                    pairs.append((round(c, 6), abs(sib_index.get(dn, 0) - sib_index.get(sn, 0)), dn, sn))
        pairs.sort()
        for _, _, dn, sn in pairs:
            if dn in mapping or sn in taken:
                continue
            mapping[dn] = sn
            taken.add(sn)

    # Walk the other rig top-down: a bone's candidates are the children of
    # its parent's match, so coincident bones in different limbs never mix.
    queue = [None]
    while queue:
        p = queue.pop(0)
        kids = dst_kids.get(p, [])
        if p is None:
            cands = src_kids.get(None, [])
        elif p in mapping:
            cands = src_kids.get(mapping[p], [])
        else:
            cands = []
        if cands:
            assign(kids, cands)
        queue.extend(kids)
    # Whatever is left (parent unmatched, or hierarchy differs): any free bone.
    free = [b.name for b in src_bones if b.name not in taken]
    if free:
        assign([b.name for b in dst_bones if b.name not in mapping], free)
    unmatched = [d.name for d in dst_bones if d.name not in mapping]
    return mapping, unmatched


def _under(obj, arm):
    p = obj.parent
    while p is not None:
        if p is arm:
            return True
        p = p.parent
    return False


def attached_objects(arm):
    """Objects that belong to a rig: anything parented under it, plus meshes
    an armature modifier binds to it (M2I importers often park the meshes
    beside the armature under a root empty rather than under the rig)."""
    out = []
    for obj in bpy.data.objects:
        if obj is arm:
            continue
        if _under(obj, arm) or any(
                m.type == "ARMATURE" and m.object is arm for m in obj.modifiers):
            out.append(obj)
    return out


def _merge_group(obj, old_name, new_name):
    """Fold vertex group old_name into new_name (weights added), drop old."""
    old = obj.vertex_groups[old_name]
    new = obj.vertex_groups[new_name]
    oi = old.index
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group == oi and g.weight > 0:
                new.add([v.index], g.weight, "ADD")
                break
    obj.vertex_groups.remove(old)


def apply_group_map(obj, changes):
    """Rename obj's vertex groups (and bone parent) by {old: new}. A group
    that already carries the new name receives the old group's weights."""
    if obj.type != "MESH":
        return 0
    n = 0
    # Two phases here too: a group may be renamed onto a name another group
    # still holds until that one is renamed away.
    todo = [vg.name for vg in obj.vertex_groups if vg.name in changes and changes[vg.name] != vg.name]
    temp = {}
    for i, name in enumerate(todo):
        t = "__retarget_vg_%04d" % i
        obj.vertex_groups[name].name = t
        temp[t] = changes[name]
    for t, final in temp.items():
        if final in obj.vertex_groups:
            _merge_group(obj, t, final)
        else:
            obj.vertex_groups[t].name = final
        n += 1
    if obj.parent_type == "BONE" and obj.parent_bone in changes:
        obj.parent_bone = changes[obj.parent_bone]
    return n


def rename_bones(dst_arm, mapping):
    """Rename the other rig's bones in place, then rename the vertex groups
    of every attached mesh by the same map. Blender renames groups itself
    only for meshes it sees as deformed by this rig, so this is done
    explicitly and is a no-op where Blender already did it."""
    arm = dst_arm.data
    changes = {d: s for d, s in mapping.items() if d != s and d in arm.bones}
    if not changes:
        return 0, 0
    attached = [o for o in attached_objects(dst_arm) if o.type == "MESH"]
    # Two phases so a swap (A->B, B->A) cannot collide.
    temp = {}
    for i, d in enumerate(changes):
        t = "__retarget_%04d" % i
        arm.bones[d].name = t
        temp[t] = changes[d]
    for t, final in temp.items():
        arm.bones[t].name = final
    groups = 0
    for obj in attached:
        groups += apply_group_map(obj, changes)
        # Blender may have renamed the groups to the temp names already.
        groups += apply_group_map(obj, temp)
    return len(changes), groups


def stale_groups(obj, arm):
    """Vertex groups on obj that name no bone of arm."""
    return [vg.name for vg in obj.vertex_groups if vg.name not in arm.data.bones]


# ---------------------------------------------------------------------------
# vertex groups by bone position
def _strip_suffix(name):
    """'bone_012.001' -> 'bone_012'."""
    base, dot, tail = name.rpartition(".")
    return base if dot and tail.isdigit() else name


def bone_positions_by_name(arms):
    """{bone name: (head, tail) in world space} over several armatures; the
    first armature that has a name wins."""
    out = {}
    for arm in arms:
        if arm is None or arm.type != "ARMATURE":
            continue
        for name, ends in _bone_ends_world(arm).items():
            out.setdefault(name, ends)
    return out


def nearest_bone(src_ends, head, tail=None, max_distance=0.05):
    """Name of the M2 bone whose rest head (and tail) is closest, or None."""
    best, best_c = None, None
    for name, (sh, st) in src_ends.items():
        c = (head - sh).length
        if tail is not None:
            c += 0.5 * (tail - st).length
        if best_c is None or c < best_c:
            best, best_c = name, c
    return best if best is not None and best_c <= max_distance else None


def _segment_distance(p, a, b):
    ab = b - a
    l2 = ab.length_squared
    t = 0.0 if l2 < 1e-12 else max(0.0, min(1.0, (p - a).dot(ab) / l2))
    return (p - (a + ab * t)).length


def _group_centroid(obj, vg):
    """Weighted world-space centre of a vertex group, or None if empty."""
    gi = vg.index
    mw = obj.matrix_world
    total = 0.0
    acc = Vector((0.0, 0.0, 0.0))
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group == gi and g.weight > 0:
                acc += (mw @ v.co) * g.weight
                total += g.weight
                break
    return (acc / total) if total > 0 else None


def copy_groups_from(src, dst):
    """Replace dst's vertex groups with src's, carried over by nearest
    vertex position: exact for the same shape, sensible for an edited one."""
    pts = [src.matrix_world @ v.co for v in src.data.vertices]
    if not pts:
        return 0
    kd = _kd(pts)
    names = {g.index: g.name for g in src.vertex_groups}
    for vg in list(dst.vertex_groups):
        dst.vertex_groups.remove(vg)
    groups = {}
    dmw = dst.matrix_world
    sverts = src.data.vertices
    for v in dst.data.vertices:
        _, i, _ = kd.find(dmw @ v.co)
        for g in sverts[i].groups:
            if g.weight > 0 and g.group in names:
                groups.setdefault(names[g.group], []).append((v.index, g.weight))
    for name in sorted(groups):
        vg = dst.vertex_groups.new(name=name)
        for idx, w in groups[name]:
            vg.add([idx], w, "REPLACE")
    return len(groups)


def reassign_stale_groups(obj, src_arm, name_positions, max_distance=0.05,
                          guess_from_weights=True, guess_distance=0.5):
    """Rename every vertex group of obj that names no bone of the M2 rig to
    the M2 bone standing where the group's own bone stood.

    The group's bone is looked up by name in ``name_positions`` (rest
    head/tail of the other rig's bones under their old names, plus any
    other armature in the file). When no bone of that name exists anywhere
    any more, the group's weighted centre is matched to the closest M2 bone
    segment instead (``guess_from_weights``).
    Returns (changes, guessed, unresolved)."""
    if obj.type != "MESH" or not obj.vertex_groups:
        return {}, [], []
    bones = src_arm.data.bones
    src_ends = _bone_ends_world(src_arm)
    changes, guessed, unresolved = {}, [], []
    for vg in obj.vertex_groups:
        if vg.name in bones:
            continue
        base = _strip_suffix(vg.name)
        if base in bones:
            changes[vg.name] = base
            continue
        ends = name_positions.get(vg.name) or name_positions.get(base)
        target = None
        if ends is not None:
            target = nearest_bone(src_ends, ends[0], ends[1], max_distance)
        if target is None and guess_from_weights:
            c = _group_centroid(obj, vg)
            if c is not None:
                best, best_d = None, None
                for name, (sh, st) in src_ends.items():
                    d = _segment_distance(c, sh, st)
                    if best_d is None or d < best_d:
                        best, best_d = name, d
                if best is not None and best_d <= guess_distance:
                    target = best
                    guessed.append((vg.name, best))
        if target is None:
            unresolved.append(vg.name)
        else:
            changes[vg.name] = target
    apply_group_map(obj, changes)
    return changes, guessed, unresolved


# ---------------------------------------------------------------------------
# objects
def move_children(src_arm, dst_arm):
    """Re-parent every object of the other rig to the M2 rig, keeping its
    world transform, and point armature modifiers at the M2 rig. Direct
    children move as a unit with their own children; a mesh bound to the
    rig only through its modifier moves too."""
    moved = []
    todo = list(dst_arm.children)
    for obj in attached_objects(dst_arm):
        if obj not in todo and not _under(obj, dst_arm):
            todo.append(obj)
    for obj in todo:
        mw = obj.matrix_world.copy()
        ptype, pbone = obj.parent_type, obj.parent_bone
        obj.parent = src_arm
        obj.parent_type = ptype
        if ptype == "BONE":
            obj.parent_bone = pbone if pbone in src_arm.data.bones else ""
            if not obj.parent_bone:
                obj.parent_type = "OBJECT"
        obj.matrix_parent_inverse = Matrix.Identity(4)
        obj.matrix_world = mw
        if obj.type == "MESH":
            mods = [m for m in obj.modifiers if m.type == "ARMATURE"]
            if mods:
                for m in mods:
                    m.object = src_arm
            else:
                m = obj.modifiers.new(name="Armature", type="ARMATURE")
                m.object = src_arm
        # Keep it in the M2 rig's collections so it exports with the model.
        for coll in src_arm.users_collection:
            if obj.name not in coll.objects:
                coll.objects.link(obj)
        moved.append(obj)
    return moved


# ---------------------------------------------------------------------------
# shapes
def _world_points(obj, limit=4000):
    mw = obj.matrix_world
    verts = obj.data.vertices
    n = len(verts)
    step = max(1, n // limit)
    return [mw @ verts[i].co for i in range(0, n, step)]


def _kd(points):
    kd = KDTree(len(points))
    for i, p in enumerate(points):
        kd.insert(p, i)
    kd.balance()
    return kd


def _bbox_world(obj):
    """World-space bounds from the mesh vertices themselves. Object.bound_box
    is a cached runtime value that is stale (or zero) for an object whose
    modifiers or parent just changed, which made freshly moved meshes miss
    every candidate."""
    verts = obj.data.vertices
    n = len(verts)
    if not n:
        return Vector((0, 0, 0)), Vector((0, 0, 0))
    co = [0.0] * (3 * n)
    verts.foreach_get("co", co)
    mw = obj.matrix_world
    lo = Vector((1e30, 1e30, 1e30))
    hi = Vector((-1e30, -1e30, -1e30))
    for i in range(0, 3 * n, 3):
        p = mw @ Vector((co[i], co[i + 1], co[i + 2]))
        lo.x = min(lo.x, p.x); lo.y = min(lo.y, p.y); lo.z = min(lo.z, p.z)
        hi.x = max(hi.x, p.x); hi.y = max(hi.y, p.y); hi.z = max(hi.z, p.z)
    return lo, hi


def shape_distance(a, b, a_pts=None, b_kd=None):
    """Mean nearest-vertex distance from mesh a to mesh b, in world units.
    Symmetric-ish: the larger of the two directions is returned."""
    a_pts = a_pts if a_pts is not None else _world_points(a)
    b_pts = _world_points(b)
    b_kd = b_kd if b_kd is not None else _kd(b_pts)
    a_kd = _kd(a_pts)
    d_ab = sum(b_kd.find(p)[2] for p in a_pts) / max(1, len(a_pts))
    d_ba = sum(a_kd.find(p)[2] for p in b_pts) / max(1, len(b_pts))
    return max(d_ab, d_ba)


def match_meshes(moved, originals, max_distance=0.01):
    """Pair each moved mesh with the original geoset mesh of the same shape.
    Returns ({moved: original}, [unmatched moved])."""
    cands = [o for o in originals if o.type == "MESH" and len(o.data.vertices)]
    boxes = {o: _bbox_world(o) for o in cands}
    kds = {}
    result, unmatched = {}, []
    used = set()
    for m in moved:
        if m.type != "MESH" or not len(m.data.vertices):
            continue
        lo, hi = _bbox_world(m)
        diag = (hi - lo).length
        slack = max(max_distance * 4, diag * 0.05)
        m_pts = _world_points(m)
        best, best_d = None, None
        for o in cands:
            if o in used:
                continue
            olo, ohi = boxes[o]
            if (olo - lo).length > slack or (ohi - hi).length > slack:
                continue
            if o not in kds:
                kds[o] = _kd(_world_points(o))
            d = shape_distance(m, o, m_pts, kds[o])
            # Prefer equal vertex counts when shapes tie (duplicate variants).
            key = (d, 0 if len(o.data.vertices) == len(m.data.vertices) else 1,
                   int(o.get("m2_order", 1 << 30)))
            if best is None or key < best_d:
                best, best_d = o, key
        if best is not None and best_d[0] <= max_distance:
            result[m] = best
            used.add(best)
        else:
            unmatched.append(m)
    return result, unmatched


def _natural_key(name):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def match_meshes_by_name(moved, originals):
    """Pair by object name: 'X_geoset_0001.001' or 'X_geoset_0001_edit'
    pairs with the original 'X_geoset_0001'."""
    by_name = {o.name: o for o in originals}
    base_names = sorted(by_name, key=len, reverse=True)
    result, used = {}, set()
    for m in moved:
        if m.type != "MESH":
            continue
        cand = by_name.get(_strip_suffix(m.name)) or by_name.get(m.name)
        if cand is None:
            for n in base_names:
                if m.name.startswith(n) or m.data.name.startswith(n):
                    cand = by_name[n]
                    break
        if cand is not None and cand not in used:
            result[m] = cand
            used.add(cand)
    return result


def match_meshes_by_order(moved, originals):
    """Pair index by index, both sides in natural name order (the M2 side
    by its geoset order when stamped)."""
    ms = sorted((m for m in moved if m.type == "MESH"), key=lambda o: _natural_key(o.name))
    os_ = sorted(originals, key=lambda o: (int(o.get("m2_order", 1 << 30)), _natural_key(o.name)))
    return dict(zip(ms, os_))


def pair_meshes(moved, originals, method="AUTO", max_distance=0.01):
    """Pair moved meshes with the geoset meshes they replace.
    AUTO: by name, then by shape for the rest. NAME / SHAPE / ORDER: that
    one rule only. Returns ({moved: original}, [unmatched moved])."""
    meshes = [m for m in moved if m.type == "MESH"]
    pairs = {}
    if method in ("AUTO", "NAME"):
        pairs.update(match_meshes_by_name(meshes, originals))
    if method in ("AUTO", "SHAPE"):
        rest = [m for m in meshes if m not in pairs]
        free = [o for o in originals if o not in pairs.values()]
        more, _ = match_meshes(rest, free, max_distance)
        pairs.update(more)
    if method == "ORDER":
        pairs.update(match_meshes_by_order(meshes, originals))
    unmatched = [m for m in meshes if m not in pairs]
    return pairs, unmatched


# ---------------------------------------------------------------------------
# settings
def _copy_props(src, dst):
    n = 0
    for k in list(src.keys()):
        if not str(k).startswith("m2_") or k in _SKIP_PROPS:
            continue
        try:
            dst[k] = src[k]
            n += 1
        except Exception:  # noqa: BLE001 - odd ID props stay behind
            pass
    return n


def copy_materials(src, dst):
    """Give dst the material slots of src and assign each face the material
    of the nearest src face, so multi-material geosets keep their split."""
    mats = [s.material for s in src.material_slots]
    dst.data.materials.clear()
    for mat in mats:
        dst.data.materials.append(mat)
    if not mats:
        return
    if len(mats) == 1 or not len(src.data.polygons):
        for p in dst.data.polygons:
            p.material_index = 0
        return
    smw, dmw = src.matrix_world, dst.matrix_world
    kd = KDTree(len(src.data.polygons))
    for p in src.data.polygons:
        kd.insert(smw @ p.center, p.index)
    kd.balance()
    spolys = src.data.polygons
    for p in dst.data.polygons:
        _, i, _ = kd.find(dmw @ p.center)
        p.material_index = min(spolys[i].material_index, len(mats) - 1)


def copy_missing_uv_layers(src, dst):
    """Give dst every UV set src has and dst lacks (UVMap2 for the eye-glow
    layers, for one), filled per loop from the nearest src face so seams
    stay intact. Returns the names created."""
    missing = [uv.name for uv in src.data.uv_layers if uv.name not in dst.data.uv_layers]
    if not missing or not len(src.data.polygons) or not len(dst.data.polygons):
        return []
    smw, dmw = src.matrix_world, dst.matrix_world
    spolys, sverts, sloops = src.data.polygons, src.data.vertices, src.data.loops
    kd = KDTree(len(spolys))
    for p in spolys:
        kd.insert(smw @ p.center, p.index)
    kd.balance()
    # dst loop -> src loop, by nearest face then nearest corner in it
    loop_map = [0] * len(dst.data.loops)
    dverts = dst.data.vertices
    for dp in dst.data.polygons:
        _, si, _ = kd.find(dmw @ dp.center)
        sp = spolys[si]
        corners = [(smw @ sverts[sloops[li].vertex_index].co, li) for li in sp.loop_indices]
        for li in dp.loop_indices:
            p = dmw @ dverts[dst.data.loops[li].vertex_index].co
            loop_map[li] = min(corners, key=lambda c: (c[0] - p).length_squared)[1]
    for name in missing:
        s_uv = src.data.uv_layers[name].data
        d_uv = dst.data.uv_layers.new(name=name).data
        for li, sl in enumerate(loop_map):
            d_uv[li].uv = s_uv[sl].uv
    return missing


def repair_stray_weights(obj, arm):
    """A vertex that shares no bone with any neighbour (or has no weight at
    all) takes its neighbours' averaged weights: the retail eye-glow vertex
    bound to the root, and anything an M2I round trip left unbound.
    Returns the number of vertices repaired."""
    mesh = obj.data
    if not obj.vertex_groups or not len(mesh.edges):
        return 0
    names = {g.index: g.name for g in obj.vertex_groups}
    bone_names = set(arm.data.bones.keys())
    weights = []
    for v in mesh.vertices:
        weights.append({names[g.group]: g.weight for g in v.groups
                        if g.weight > 0 and names[g.group] in bone_names})
    nbrs = [set() for _ in mesh.vertices]
    for e in mesh.edges:
        a, b = e.vertices
        nbrs[a].add(b); nbrs[b].add(a)
    fixed = 0
    for vi, w in enumerate(weights):
        around = [weights[n] for n in nbrs[vi] if weights[n]]
        if not around:
            continue
        shared = any(set(w) & set(nw) for nw in around)
        if w and shared:
            continue
        acc = {}
        for nw in around:
            for name, val in nw.items():
                acc[name] = acc.get(name, 0.0) + val / len(around)
        top = sorted(acc.items(), key=lambda kv: kv[1], reverse=True)[:4]
        tot = sum(v for _, v in top) or 1.0
        for vg in obj.vertex_groups:
            vg.remove([vi])
        for name, val in top:
            vg = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
            vg.add([vi], val / tot, "REPLACE")
        fixed += 1
    if fixed:
        # Drop groups the repair emptied (e.g. the root group that held only
        # the stray vertex) so the mesh lists exactly the bones it uses.
        used = set()
        for v in mesh.vertices:
            for g in v.groups:
                if g.weight > 0:
                    used.add(g.group)
        for vg in [vg for vg in obj.vertex_groups if vg.index not in used]:
            obj.vertex_groups.remove(vg)
    return fixed


def copy_settings(src, dst, take_name=True):
    """Copy the M2 geoset data (custom properties, materials, name) from the
    original geoset mesh onto the retargeted one."""
    n = _copy_props(src, dst) + _copy_props(src.data, dst.data)
    copy_materials(src, dst)
    # Name the UV sets the way the exporter reads them.
    uvs = dst.data.uv_layers
    if uvs and "UVMap" not in uvs:
        uvs[0].name = "UVMap"
    if take_name:
        old = src.name
        src.name = old + ".replaced"
        dst.name = old
        old_data = src.data.name
        src.data.name = old_data + ".replaced"
        dst.data.name = old_data
    return n


# ---------------------------------------------------------------------------
def geoset_meshes(arm):
    """The imported geoset meshes of an M2 rig."""
    return [o for o in arm.children if o.type == "MESH" and "m2_skin_section_id" in o]


def retarget(src_arm, dst_arm, bone_method="ORDER", bone_distance=0.05,
             shape_distance_max=0.01, take_names=True, originals="DELETE",
             remove_other=True, guess_from_weights=True, mesh_method="AUTO",
             copy_uvs=True, repair_weights=True, log=print):
    """Run the whole retarget. ``originals``: DELETE the replaced geoset
    meshes, HIDE them, or KEEP them untouched. ``remove_other`` deletes the
    now-empty other rig, so the export sees a single armature.
    Returns a summary dict."""
    if src_arm is dst_arm:
        raise RetargetError("Pick two different armatures.")
    for a in (src_arm, dst_arm):
        if a is None or a.type != "ARMATURE":
            raise RetargetError("Both picks must be armatures.")
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    originals_before = geoset_meshes(src_arm)
    # Where every bone name in the file stands right now, before anything is
    # renamed: this is what a stale vertex group name is resolved against.
    name_positions = bone_positions_by_name(
        [dst_arm] + [o for o in bpy.data.objects if o.type == "ARMATURE" and o not in (src_arm, dst_arm)])
    try:
        mapping, unmatched_bones = match_bones(src_arm, dst_arm, bone_method, bone_distance)
    except RetargetError as e:
        # Different bone counts: order cannot pair them, fall back to position.
        log("[retarget] %s; pairing bones by rest position instead" % e)
        mapping, unmatched_bones = match_bones(src_arm, dst_arm, "POSITION", bone_distance)
    renamed, groups = rename_bones(dst_arm, mapping)
    log("[retarget] bones: %d matched, %d renamed, %d vertex group(s) renamed, %d unmatched%s"
        % (len(mapping), renamed, groups, len(unmatched_bones),
           (" (%s)" % ", ".join(unmatched_bones[:8])) if unmatched_bones else ""))

    moved = move_children(src_arm, dst_arm)
    bpy.context.view_layer.update()
    meshes = [o for o in moved if o.type == "MESH"]
    log("[retarget] moved %d object(s) under %s" % (len(moved), src_arm.name))
    pairs, unmatched_meshes = pair_meshes(meshes, originals_before, mesh_method, shape_distance_max)

    # Anything the name map could not place: put it on the M2 bone that
    # stands where the group's bone stood. A mesh whose group names exist
    # nowhere any more takes the weights of the geoset it replaces instead
    # (same shape, so nearest-vertex transfer is exact); the weighted-centre
    # guess is the last resort for a mesh that matched no geoset.
    stale = {}
    repositioned = 0
    for m in meshes:
        if not stale_groups(m, src_arm):
            continue
        changes, guessed, unresolved = reassign_stale_groups(
            m, src_arm, name_positions, bone_distance, guess_from_weights=False)
        repositioned += len(changes)
        if changes:
            log("[retarget] %s: %d vertex group(s) re-pointed by bone position" % (m.name, len(changes)))
        if unresolved and m in pairs:
            n = copy_groups_from(pairs[m], m)
            repositioned += n
            log("[retarget] %s: %d group name(s) exist on no rig; took the %d vertex group(s) of %s by vertex position"
                % (m.name, len(unresolved), n, pairs[m].name))
            unresolved = stale_groups(m, src_arm)
        elif unresolved and guess_from_weights:
            changes, guessed, unresolved = reassign_stale_groups(
                m, src_arm, {}, bone_distance, guess_from_weights=True)
            repositioned += len(changes)
            if guessed:
                log("[retarget] %s: %d vertex group(s) guessed from their weights (%s)"
                    % (m.name, len(guessed), ", ".join("%s>%s" % g for g in guessed[:4])))
        if unresolved:
            stale[m.name] = unresolved
            log("[retarget] %s: %d vertex group(s) still name no bone of %s: %s"
                % (m.name, len(unresolved), src_arm.name, ", ".join(unresolved[:6])))
    copied = 0
    uv_sets = 0
    for m, o in pairs.items():
        copied += copy_settings(o, m, take_name=take_names)
        if copy_uvs:
            made = copy_missing_uv_layers(o, m)
            uv_sets += len(made)
            if made:
                log("[retarget] %s: UV set(s) %s taken from %s" % (m.name, ", ".join(made), o.name))
    repaired = 0
    if repair_weights:
        for m in meshes:
            n = repair_stray_weights(m, src_arm)
            if n:
                repaired += n
                log("[retarget] %s: %d stray / unweighted vertex(es) re-weighted from their neighbours" % (m.name, n))
    log("[retarget] meshes (%s): %d took over a geoset, %d unmatched%s"
        % (mesh_method.lower(), len(pairs), len(unmatched_meshes),
           (" (%s)" % ", ".join(x.name for x in unmatched_meshes[:8])) if unmatched_meshes else ""))

    replaced = list(pairs.values())
    if originals == "DELETE":
        for o in replaced:
            bpy.data.objects.remove(o, do_unlink=True)
    elif originals == "HIDE":
        for o in replaced:
            o.hide_set(True)
            o.hide_render = True
    other_removed = False
    if remove_other and not dst_arm.children:
        data = dst_arm.data
        bpy.data.objects.remove(dst_arm, do_unlink=True)
        if data.users == 0:
            bpy.data.armatures.remove(data)
        other_removed = True
    return {
        "other_removed": other_removed,
        "bones_matched": len(mapping), "bones_renamed": renamed,
        "bones_unmatched": unmatched_bones, "moved": len(moved),
        "matched": len(pairs), "unmatched": [m.name for m in unmatched_meshes],
        "props_copied": copied, "replaced": len(replaced),
        "groups_renamed": groups + repositioned, "groups_repositioned": repositioned,
        "stale_groups": stale, "uv_sets": uv_sets, "weights_repaired": repaired,
    }


def fix_groups(meshes, src_arm, bone_distance=0.05, guess_from_weights=True,
               shape_distance_max=0.01, log=print):
    """Standalone pass over meshes already sitting on the M2 rig: every
    vertex group that names no M2 bone is re-pointed by the rest position
    of the bone it was made for (looked up by name on any other armature in
    the file). Names that exist nowhere: the mesh takes the weights of a
    correctly skinned mesh of the same shape on the rig if there is one,
    else its groups are guessed from their weighted centre."""
    others = [o for o in bpy.data.objects if o.type == "ARMATURE" and o is not src_arm]
    name_positions = bone_positions_by_name(others)
    meshes = [m for m in meshes if m.type == "MESH"]
    broken = [m for m in meshes if stale_groups(m, src_arm)]
    donors = [o for o in src_arm.children_recursive
              if o.type == "MESH" and o not in broken and o.vertex_groups and not stale_groups(o, src_arm)]
    pairs, _ = match_meshes(broken, donors, shape_distance_max) if donors else ({}, broken)
    total, guessed_n, copied_n, stale = 0, 0, 0, {}
    for m in broken:
        changes, guessed, unresolved = reassign_stale_groups(
            m, src_arm, name_positions, bone_distance, guess_from_weights=False)
        total += len(changes)
        if unresolved and m in pairs:
            n = copy_groups_from(pairs[m], m)
            copied_n += 1
            total += n
            log("[retarget] %s: took the %d vertex group(s) of %s by vertex position" % (m.name, n, pairs[m].name))
            unresolved = stale_groups(m, src_arm)
        elif unresolved and guess_from_weights:
            changes, guessed, unresolved = reassign_stale_groups(
                m, src_arm, {}, bone_distance, guess_from_weights=True)
            total += len(changes)
            guessed_n += len(guessed)
        log("[retarget] %s: %d group(s) re-pointed, %d guessed, %d unresolved%s"
            % (m.name, len(changes), len(guessed), len(unresolved),
               (" (%s)" % ", ".join(unresolved[:6])) if unresolved else ""))
        if unresolved:
            stale[m.name] = unresolved
    return {"groups_repositioned": total, "guessed": guessed_n, "copied": copied_n, "stale_groups": stale}
