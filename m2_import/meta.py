"""The non-geometry M2 tables, as a plain dict."""

import re
import base64

from . import anim_data


def texture_file_id(tex):
    """Best-effort FileDataID for a texture (TXID value, or parsed from name)."""
    if tex.file_data_id:
        return tex.file_data_id
    m = re.search(r"(\d{4,})", tex.filename or "")
    return int(m.group(1)) if m else 0


def build(model, fps):
    """The full metadata dict for ``model``."""
    return {
        "name": model.name,
        "fps": fps,
        "version": model.version or 274,
        "source_path": model.source_path or "",
        "skin_file_ids": list(model.skin_file_ids),
        "texture_file_ids": [texture_file_id(t) for t in model.textures],
        "aux_chunks": {name: base64.b64encode(raw).decode("ascii")
                       for name, raw in (model.aux_chunks or {}).items()},
        "bounds": {
            "bbox": [list(model.bounding_min or (0, 0, 0)), list(model.bounding_max or (0, 0, 0)),
                     float(model.bounding_radius or 0.0)],
            "collision": [list(model.collision_min or (0, 0, 0)),
                          list(model.collision_max or (0, 0, 0)),
                          float(model.collision_radius or 0.0)],
        },
        "global_loops": list(model.global_loops),
        "seq_lookup": list(model.seq_lookup),
        "sequences": [
            {"id": s.id, "var": s.variation_index,
             "dur": s.duration, "flags": s.flags,
             "movespeed": getattr(s, "movespeed", 0.0),
             "blend_in": getattr(s, "blend_time_in", 150),
             "blend_out": getattr(s, "blend_time_out", 0),
             "alias_next": int(getattr(s, "alias_next", i)),
             "bounds": ([list(s.bounds[0]), list(s.bounds[1]), s.bounds[2]]
                        if getattr(s, "bounds", None) else None)}
            for i, s in enumerate(model.sequences)
        ],
        "textures": [
            {"type": t.type, "flags": t.flags, "filename": t.filename}
            for t in model.textures
        ],
        "materials": [
            {"flags": m.flags, "blend": m.blend_mode} for m in model.materials
        ],
        "texture_lookup": list(model.texture_lookup),
        "n_colors": model.n_colors,
        "n_texture_weights": model.n_texture_weights,
        "n_texture_transforms": model.n_texture_transforms,
        "tex_coord_combos": list(model.tex_coord_combos),
        "tex_weight_combos": list(model.tex_weight_combos),
        "tex_transform_combos": list(model.tex_transform_combos),
        "texture_indices_by_id": list(model.texture_indices_by_id),
        "colors": [[anim_data.track_to_json(c), anim_data.track_to_json(a)]
                   for c, a in model.colors],
        "weights": [anim_data.track_to_json(w) for w in model.weights],
        "transforms": [[anim_data.track_to_json(t), anim_data.track_to_json(r),
                        anim_data.track_to_json(s)] for t, r, s in model.transforms],
        "bones": [
            {"key_bone_id": b.key_bone_id, "flags": b.flags,
             "submesh_id": getattr(b, "submesh_id", 0),
             "name_crc": getattr(b, "name_crc", 0)} for b in model.bones
        ],
        "attachments": [
            {"id": a.id, "bone": a.bone, "pos": list(a.position)}
            for a in model.attachments
        ],
        "attachment_lookup": list(model.attachment_lookup),
        "batches": [
            {"flags": b.flags, "shader_id": b.shader_id,
             "submesh_index": b.submesh_index, "material_index": b.material_index,
             "geoset_index": getattr(b, "geoset_index", 0),
             "texture_combo_index": b.texture_combo_index,
             "color_index": b.color_index, "material_layer": b.material_layer,
             "texture_count": b.texture_count,
             "coord_combo": b.texture_coord_combo,
             "weight_combo": b.texture_weight_combo,
             "transform_combo": b.texture_transform_combo}
            for b in (model.skin.batches if model.skin else [])
        ],
    }
