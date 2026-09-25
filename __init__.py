"""Elune M2: Blender 5.x extension entry point."""

import os
import sys
import traceback

for _stale in [m for m in list(sys.modules) if m.startswith(__name__ + ".m2_import")]:
    del sys.modules[_stale]

import bpy
from bpy.props import StringProperty, EnumProperty, BoolProperty, CollectionProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper, ExportHelper

from . import m2_import
from .m2_import import versions, builder
from .m2_import import m2_ui
from .m2_import import m2_physics_ui


def _copy_base_skins(base_m2, out_m2):
    """Copy the source model's .skin files next to the exported .m2, renaming them to match."""
    import shutil
    base_dir = os.path.dirname(base_m2)
    base_stem = os.path.basename(base_m2)
    base_stem = base_stem[:-3] if base_stem.lower().endswith(".m2") else base_stem
    out_dir = os.path.dirname(out_m2)
    out_stem = os.path.basename(out_m2)
    out_stem = out_stem[:-3] if out_stem.lower().endswith(".m2") else out_stem
    copied = 0
    for f in os.listdir(base_dir):
        if f.lower().endswith(".skin") and f.startswith(base_stem):
            suffix = f[len(base_stem):]
            shutil.copyfile(os.path.join(base_dir, f),
                            os.path.join(out_dir, out_stem + suffix))
            copied += 1
    return copied


def _expansion_items(self, context):
    items = [("AUTO", "Auto-detect", "Pick a parser from the file's version field")]
    for key in versions.EXPANSION_ORDER:
        items.append((key, versions.EXPANSION_LABELS[key], ""))
    return items


class IMPORT_SCENE_OT_wow_m2(Operator, ImportHelper):
    """Import a World of Warcraft M2 model"""
    bl_idname = "import_scene.wow_m2"
    bl_label = "Import M2"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".m2"
    filter_glob: StringProperty(default="*.m2", options={"HIDDEN"})

    # Set by Blender for multi-file selection and drag-and-drop.
    files: CollectionProperty(type=bpy.types.OperatorFileListElement, options={"HIDDEN", "SKIP_SAVE"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN", "SKIP_SAVE"})

    expansion: EnumProperty(
        name="Expansion",
        description="Which M2 format to parse the file as",
        items=_expansion_items,
        default=0,
    )
    import_armature: BoolProperty(
        name="Import Skeleton",
        description="Build an armature from the model's bones and skin the mesh to it",
        default=True,
    )
    split_geosets: BoolProperty(
        name="Split Geosets",
        description="Import each geoset (submesh) as its own mesh object instead of one merged mesh",
        default=True,
    )
    import_animations: BoolProperty(
        name="Import Animations",
        description="Build a Blender Action for each animation sequence from the bone tracks "
                    "(requires Import Skeleton)",
        default=True,
    )
    import_attachments: BoolProperty(
        name="Import Attachments",
        description="Create an empty at each attachment point (weapon/effect mounts), "
                    "parented to its bone",
        default=True,
    )
    import_cameras: BoolProperty(
        name="Import Cameras",
        description="Create a Blender camera for each M2 camera (portrait / "
                    "character-info), aimed at a target empty, with its FOV and "
                    "any animation",
        default=True,
    )
    import_bounds: BoolProperty(
        name="Import Bounding Box",
        description="Create an editable wireframe 'M2_BoundingBox' object at the "
                    "model's render bounds. Move/scale it (or edit its corners) and "
                    "export will write the new box: the client frames the view "
                    "camera around its centre, so this is how you raise/lower where "
                    "the camera looks",
        default=True,
    )
    import_events: BoolProperty(
        name="Import Events",
        description="Create an empty per M2 event ($SHR, $FL0, ...) parented to "
                    "its bone. Also fills in standard events the model is missing "
                    "so triggers (sheathe, footsteps, cast, breath) fire in-game",
        default=True,
    )
    anim_fps: bpy.props.IntProperty(
        name="Animation FPS",
        description="Frames per second used to convert millisecond keyframe times",
        default=30, min=1, max=240,
    )
    max_animations: bpy.props.IntProperty(
        name="Max Animations",
        description="Limit how many animation clips to build (0 = all). Large characters "
                    "can have hundreds of sequences and millions of keyframes; lower this "
                    "for a faster import",
        default=0, min=0,
    )
    bone_tilt: bpy.props.FloatProperty(
        name="Bone Tilt (deg)",
        description="Rotate every bone's direction about Z by this many degrees "
                    "so they fan outward from the model's front instead of all "
                    "pointing +Y: much easier to see and click in the viewport. "
                    "Animations are unaffected: tracks are rotated into bone "
                    "space on import and back out on export. 0 = classic +Y",
        default=0.0, min=-180.0, max=180.0,
    )
    weld_seams: BoolProperty(
        name="Weld Seam Vertices",
        description="Merge the duplicate vertices M2 stores at UV/normal seams "
                    "into single connected points, keeping their UVs and normals "
                    "per-face. Gives a properly welded mesh to edit. Turn off for "
                    "an exact one-vertex-per-M2-slot import (only needed for the "
                    "in-place vertex patch, which requires identical topology)",
        default=True,
    )
    mirror_x: BoolProperty(
        name="Mirror X",
        description="Negate the X axis (flip handedness) if your pipeline expects it",
        default=False,
    )
    repair_weights: BoolProperty(
        name="Repair Stray Weights",
        description="A vertex bound to a bone that none of its neighbours use (retail "
                    "files have the odd one bound to the root in the middle of a head "
                    "part) takes its neighbours' weights, so it moves with the rest",
        default=True,
    )
    normalize_uvs: BoolProperty(
        name="Normalize UV Tiles",
        description="Shift each UV island by whole tiles onto the 0-1 grid. Retail hair "
                    "sits several tiles away from the origin; with its wrapping texture "
                    "that draws the same in game but is awkward to edit. Only axes the "
                    "texture wraps on are shifted, so the model looks identical",
        default=True,
    )
    import_phys: BoolProperty(
        name="Import Physics (.phys)",
        description="Build an editable physics rig from the model's physics "
                    "data: the PFDC chunk embedded in the .m2 (modern retail "
                    "models), or a '<name>.phys' file next to it. Edit it in the "
                    "M2 Physics panel and preview it before exporting",
        default=True,
    )
    phys_override_path: StringProperty(
        name="Phys File Override",
        description="Explicit path to a .phys file. Use this when the .phys "
                    "next to the .m2 has a different stem (e.g. a loose "
                    "extract). Leave blank to auto-discover: exact stem "
                    "match, else the single .phys in the folder",
        subtype="FILE_PATH", default="",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "expansion")
        layout.prop(self, "import_armature")
        layout.prop(self, "split_geosets")

        col = layout.column()
        col.enabled = self.import_armature
        col.prop(self, "import_animations")
        col.prop(self, "import_attachments")
        col.prop(self, "import_cameras")
        layout.prop(self, "import_bounds")
        layout.prop(self, "import_events")
        col.prop(self, "bone_tilt")
        col.prop(self, "weld_seams")
        sub = col.column()
        sub.enabled = self.import_animations
        sub.prop(self, "anim_fps")
        sub.prop(self, "max_animations")

        layout.prop(self, "mirror_x")
        layout.prop(self, "normalize_uvs")
        layout.prop(self, "repair_weights")
        layout.prop(self, "import_phys")
        if self.import_phys:
            layout.prop(self, "phys_override_path")

    def execute(self, context):
        paths = []
        if self.files and self.directory:
            for f in self.files:
                if f.name:
                    paths.append(os.path.join(self.directory, f.name))
        if not paths and self.filepath:
            paths = [self.filepath]

        if not paths:
            self.report({"ERROR"}, "No .m2 file selected.")
            return {"CANCELLED"}

        ok = 0
        for path in paths:
            try:
                print("[M2] loading %s" % path, flush=True)
                model = m2_import.factory.load_model(path, self.expansion)
                root = builder.build(
                    model,
                    import_armature=self.import_armature,
                    mirror_x=self.mirror_x,
                    split_geosets=self.split_geosets,
                    import_animations=self.import_animations,
                    import_attachments=self.import_attachments,
                    import_cameras=self.import_cameras,
                    fps=self.anim_fps,
                    max_animations=self.max_animations,
                    collection=context.scene.collection,
                    bone_tilt=self.bone_tilt,
                    weld_seams=self.weld_seams,
                    import_bounds=self.import_bounds,
                    import_events=self.import_events,
                    normalize_uvs=self.normalize_uvs,
                    repair_weights=self.repair_weights,
                )
                # Remember where this came from so export can re-read the M2
                context.scene["m2_source_m2"] = path

                if self.import_phys:
                    try:
                        from .m2_import import phys_to_scene
                        arm_obj = root if (root is not None
                                           and root.type == "ARMATURE") else None
                        model_name = os.path.splitext(
                            os.path.basename(path))[0] or "M2_Model"
                        explicit = (bpy.path.abspath(self.phys_override_path)
                                    if self.phys_override_path else "")
                        phys_to_scene.load_phys_into_scene(
                            context, path, arm_obj, model_name,
                            mirror_x=self.mirror_x,
                            explicit_path=explicit,
                            pfdc=(model.aux_chunks or {}).get("PFDC", b""),
                        )
                    except Exception as exc:  # noqa: BLE001
                        traceback.print_exc()
                        print("[phys-import] failed: %s" % exc, flush=True)
                detected = versions.EXPANSION_LABELS.get(
                    model.expansion, model.expansion)
                verts = len(model.vertices)
                faces = len(model.skin.triangles) // 3 if model.skin else 0
                self.report(
                    {"INFO"},
                    f"Imported {os.path.basename(path)} "
                    f"[{detected}, v{model.version}]: "
                    f"{verts} verts, {faces} tris, "
                    f"{len(model.sequences)} anims, "
                    f"{len(model.attachments)} attachments"
                    + ("" if model.skin else " (no skin file found: mesh empty)"),
                )
                ok += 1
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self.report(
                    {"ERROR"}, f"Failed to import {os.path.basename(path)}: {exc}")

        return {"FINISHED"} if ok else {"CANCELLED"}


def _validate_and_report(model, op):
    """Surface any geoset that won't draw in-game, before the file is written."""
    from .m2_import import writer
    problems = writer.validate_skin(
        model.skin, log=lambda m: print("[M2] " + m, flush=True))
    if problems:
        op.report({"WARNING"},
                  "%d geoset(s) will NOT display in-game: see the System "
                  "Console for details (first: %s)" % (len(problems), problems[0]))
    return problems


def _apply_geoset_visibility(model, force_zero, op):
    """Optionally flatten geoset ids to 0, and warn about ones that won't draw."""
    if model.skin is None:
        return
    subs = model.skin.submeshes
    if force_zero:
        changed = sum(1 for s in subs if s.skin_section_id != 0)
        for s in subs:
            s.skin_section_id = 0
        if changed:
            print("[M2] forced %d submesh(es) to geoset 0 (always drawn)"
                  % changed, flush=True)
        return
    nonzero = sorted({s.skin_section_id for s in subs if s.skin_section_id})
    if nonzero:
        n = sum(1 for s in subs if s.skin_section_id)
        msg = ("%d of %d submeshes use non-zero geoset ids %s: WoW only draws "
               "those when the geoset group is enabled (character display info "
               "or CreatureGeosetData), so they may be invisible. Tick 'Force "
               "All Geosets Visible' if every mesh should always render."
               % (n, len(subs), nonzero[:10]))
        print("[M2] WARNING: " + msg, flush=True)
        op.report({"WARNING"}, msg)


def _build_phys_bytes(context, objects, mirror_x, op, model):
    """.phys bytes for the scene's physics rig, or None when there isn't one.

    Marks ``model`` so the client loads it: GlobalModelFlags 0x20 and the data
    itself in a PFDC chunk, the way retail models carry physics. Nothing extra
    has to be registered.
    """
    from .m2_import import phys, phys_from_scene, m2_physics_ui
    arm = next((o for o in objects if o.type == "ARMATURE"), None)
    try:
        with m2_physics_ui.at_rest(context):
            doc = phys_from_scene.build_phys_doc(context, arm, mirror_x=mirror_x)
        data = None if doc is None else phys.write(doc)
    except phys_from_scene.RigError as exc:
        op.report({"WARNING"}, "Physics NOT exported: %s" % exc)
        return None
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        op.report({"WARNING"}, "Physics export failed: %s" % exc)
        return None
    if data is None:
        if model.aux_chunks:
            model.aux_chunks.pop("PFDC", None)     # rig was removed: drop stale data
        return None

    import struct
    if model.aux_chunks is None:
        model.aux_chunks = {}
    model.global_flags = (int(getattr(model, "global_flags", 0)) | 0x20) & 0xFFFFFFFF
    # The client only lets physics move a bone that carries flag 0x400
    # ("kinematic bone"). Retail sets it on every simulated bone; a character
    # bone you rig in Blender will not have it, so set it here.
    # Retail physics bones carry 0x400 and never 0x200 ("transformed" by
    # animation); a character bone keeps its keyframes otherwise.
    flagged = 0
    for body in doc.bodies:
        if body.type == phys.BODY_DYNAMIC and body.bone_index < len(model.bones):
            bone = model.bones[body.bone_index]
            new = (bone.flags | 0x400) & ~0x200
            if new != bone.flags:
                bone.flags = new
                flagged += 1
    if flagged:
        print("[phys] set bone flag 0x400 / cleared 0x200 on %d physics bone(s)" % flagged, flush=True)
    model.aux_chunks.pop("PFID", None)          # a stale id would send the client elsewhere
    model.aux_chunks["PFDC"] = data + bytes(-len(data) % 4)
    print("[phys] exported %d bytes (embedded as PFDC)" % len(data), flush=True)
    bones = sorted({b.bone_index for b in doc.bodies if b.type == phys.BODY_DYNAMIC})
    op.report({"INFO"}, "Physics embedded: %d bodies, %d joints; bone flag 0x400 on bones %s"
              % (len(doc.bodies), len(doc.joints), ", ".join(str(b) for b in bones) or "(none)"))
    return data


def _model_objects(objects):
    """Scene objects minus the physics rig, which must not become geometry."""
    from .m2_import import phys_rig
    return [o for o in objects if not phys_rig.is_rig_object(o)]


class EXPORT_SCENE_OT_wow_m2(Operator, ExportHelper):
    """Export the imported model back to a self-contained M2 (+ .skin)"""
    bl_idname = "export_scene.wow_m2"
    bl_label = "Export M2"
    bl_options = {"REGISTER"}

    filename_ext = ".m2"
    filter_glob: StringProperty(default="*.m2", options={"HIDDEN"})

    container: EnumProperty(
        name="Format",
        description="M2 container to write",
        items=[
            ("CHUNKED", "Chunked MD21 (retail)",
             "Modern retail container (Legion+). Wraps the data in MD21 and "
             "writes SFID/TXID FileDataID chunks; use for retail clients"),
            ("FLAT", "Flat MD20 (WotLK)",
             "Legacy WotLK-era flat container with a name-based .skin"),
        ],
        default="CHUNKED",
    )
    m2_version: bpy.props.IntProperty(
        name="M2 Version",
        description="M2 version field to write (274 = Legion..retail, 264 = WotLK)",
        default=274, min=256, max=300,
    )
    skin_file_id: bpy.props.IntProperty(
        name="Skin FileDataID",
        description="FileDataID to record in the SFID chunk for the exported .skin "
                    "(0 = reuse the source model's). Register the .skin under this id "
                    "in your CASCHost/Arctium listfile",
        default=0, min=0,
    )
    base_m2_path: StringProperty(
        name="Patch into M2 (target slot)",
        description="Path to the TARGET slot's original .m2 (e.g. the nightelf you're "
                    "replacing). Your geometry is spliced into it, keeping the target's "
                    "exact structure (LOD count, SFID/LDV1, chunks): the correct way "
                    "to put one model on another's slot. Leave blank to build from scratch",
        subtype="FILE_PATH", default="",
    )
    body_texture_id: bpy.props.IntProperty(
        name="Body Skin Texture FileDataID",
        description="Hardcode the body skin: convert the runtime-composited skin "
                    "texture (type 1) to a fixed texture (type 0) pointing at this "
                    "BLP FileDataID. Use this if the body geosets are invisible: "
                    "the client composites type-1 skin itself and may not bind it "
                    "for a custom character. 0 = leave as composited",
        default=0, min=0,
    )
    new_model: BoolProperty(
        name="New Model (from scratch)",
        description="Build a complete M2 from a raw scene: mesh, armature, animation. Bones and animations are embedded.",
        default=True,
    )
    texture_file_id: bpy.props.IntProperty(
        name="Texture FileDataID",
        description="FileDataID of the .blp texture for a New Model (written to TXID). "
                    "Register your .blp under this id",
        default=0, min=0,
    )
    full_custom: BoolProperty(
        name="Full Custom (write .skel)",
        description="Build a brand-new retail character: a mesh-only single-LOD M2 "
                    "plus a .skel carrying the skeleton + animations. Use this to "
                    "change meshes AND re-animate. Requires a Skeleton FileDataID "
                    "(SKID) and a Skin FileDataID. Register the .m2, .skel, .skin "
                    "and textures in your listfile",
        default=False,
    )
    skel_file_id: bpy.props.IntProperty(
        name="Skeleton FileDataID",
        description="FileDataID for the exported .skel, recorded in the M2's SKID "
                    "chunk. Register the .skel under this id",
        default=0, min=0,
    )
    all_geosets_visible: BoolProperty(
        name="Force All Geosets Visible",
        description="Set every submesh's geoset id to 0 ('always drawn'). WoW "
                    "only draws a NON-ZERO geoset when something explicitly "
                    "enables that group (character display info, or "
                    "CreatureGeosetData): so meshes carrying ids inherited "
                    "from an imported character are invisible on a new model. "
                    "Turn this on for creature/object-style models where every "
                    "mesh should always render",
        default=False,
    )
    source_m2_path: StringProperty(
        name="Source M2 (read tables from)",
        description="Path to the .m2 this scene came from. The texture/material/"
                    "batch tables have no Blender equivalent, so they are read "
                    "back out of this file at export time: which means the "
                    "export does NOT depend on anything stashed in the .blend. "
                    "Leave blank to use the metadata saved at import instead",
        subtype="FILE_PATH", default="",
    )
    export_scale: bpy.props.FloatProperty(
        name="Export Scale",
        description="Uniformly resize the model as it is written. Only distances "
                    "scale (vertices, bone pivots, translation keyframes, "
                    "attachments, cameras, bounds): rotation and scale keyframes "
                    "are left alone, so animations survive exactly. Use this "
                    "instead of scaling the rig in Blender, which desyncs the "
                    "location keyframes from the bones. 1.0 = unchanged",
        default=1.0, min=0.001, max=1000.0,
    )
    mirror_x: BoolProperty(
        name="Mirror X",
        description="Negate the X axis on the way out (match the import setting you used)",
        default=False,
    )
    selected_only: BoolProperty(
        name="Selected Only",
        description="Export from the selected objects only, instead of the whole scene",
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "container")
        if self.container == "CHUNKED":
            layout.prop(self, "new_model")
            if self.new_model:
                layout.prop(self, "skin_file_id")
                layout.prop(self, "texture_file_id")
                layout.prop(self, "m2_version")
                layout.prop(self, "all_geosets_visible")
                layout.label(text="Mesh + armature + animation -> standalone M2.")
                layout.prop(self, "export_scale")
                layout.prop(self, "selected_only")
                layout.prop(self, "mirror_x")
                self._draw_phys(context, layout)
                return
            layout.prop(self, "full_custom")
            if self.full_custom:
                layout.prop(self, "skel_file_id")
                layout.prop(self, "skin_file_id")
                layout.prop(self, "m2_version")
                layout.label(text="Auto-uses the M2 you imported; no file dialog.")
            else:
                layout.prop(self, "base_m2_path")
                layout.prop(self, "m2_version")
                layout.prop(self, "skin_file_id")
            layout.prop(self, "body_texture_id")
        layout.prop(self, "source_m2_path")
        layout.prop(self, "export_scale")
        layout.prop(self, "selected_only")
        layout.prop(self, "mirror_x")
        if self.container == "CHUNKED" and not self.base_m2_path:
            self._draw_phys(context, layout)

    def _draw_phys(self, context, layout):
        from .m2_import import phys_rig
        if not phys_rig.rig_collections():
            return
        layout.label(text="Physics rig found: it will be embedded in the M2.",
                     icon="RIGID_BODY")

    def _source_m2(self, context, tried=None):
        """Path to the .m2 this scene came from, or "" if we can't find one."""
        candidates = [
            ("Source M2 field", self.source_m2_path),
            ("Patch-into-M2 field", self.base_m2_path),
            ("path remembered at import", context.scene.get("m2_source_m2", "")),
        ]
        for label, raw in candidates:
            if not raw:
                continue
            p = bpy.path.abspath(raw)
            if os.path.isfile(p):
                print("[M2] source M2: %s (%s)" % (p, label), flush=True)
                return p
            if tried is not None:
                tried.append("%s -> %s (not found)" % (label, p))
        if tried:
            print("[M2] no usable source M2; tried:\n  " + "\n  ".join(tried),
                  flush=True)
        return ""

    def execute(self, context):
        from .m2_import import scene_reader, writer
        objects = _model_objects(context.selected_objects if self.selected_only
                                 else list(context.scene.objects))
        chunked = self.container == "CHUNKED"
        try:
            print("[M2] exporting %s (%s)" % (self.filepath, self.container), flush=True)

            # New-model mode: build a complete M2 from a raw scene (no import).
            if chunked and self.new_model:
                from .m2_import import from_scene, factory
                fps = context.scene.render.fps or 30
                src_model = None
                src_path = self._source_m2(context)
                if src_path:
                    try:
                        src_model = factory.load_model(src_path)
                        print("[M2] material auto-fill source: %s"
                              % os.path.basename(src_path), flush=True)
                    except Exception as exc:  # noqa: BLE001
                        print("[M2] could not read %s for material auto-fill (%s)"
                              % (src_path, exc), flush=True)
                model = from_scene.build_model_from_scene(
                    objects, fps=fps, mirror_x=self.mirror_x,
                    texture_file_id=self.texture_file_id,
                    name=os.path.splitext(os.path.basename(self.filepath))[0] or "Model",
                    source_model=src_model)
                writer.scale_model(model, self.export_scale)
                _apply_geoset_visibility(model, self.all_geosets_visible, self)

                phys_bytes = _build_phys_bytes(context, objects, self.mirror_x, self, model)

                skin_ids = [self.skin_file_id] if self.skin_file_id else None
                # One TXID entry per texture the materials resolved to, not just
                # the fallback id: a multi-texture model needs them all.
                tex_ids = [t.file_data_id for t in model.textures]
                m2_bytes, skin_bytes = writer.write_m2_chunked(
                    model, version=self.m2_version, skin_file_ids=skin_ids,
                    texture_file_ids=tex_ids if any(tex_ids) else None)
                bbase = self.filepath[:-3] if self.filepath.lower().endswith(".m2") else self.filepath
                with open(self.filepath, "wb") as f:
                    f.write(m2_bytes)
                if skin_bytes is not None:
                    with open(bbase + "00.skin", "wb") as f:
                        f.write(skin_bytes)
                if phys_bytes is not None:
                    with open(bbase + ".phys", "wb") as f:
                        f.write(phys_bytes)
                print("[M2] NEW model: %d verts, %d bones, %d sequences, %d geosets, "
                      "%d textures, %d materials, %d batches -> .m2 + 00.skin "
                      "(SFID=%s TXID=%s)"
                      % (len(model.vertices), len(model.bones), len(model.sequences),
                         len(model.skin.submeshes) if model.skin else 0,
                         len(model.textures), len(model.materials),
                         len(model.skin.batches) if model.skin else 0,
                         skin_ids, tex_ids), flush=True)
                missing = sum(1 for t, tx in zip(tex_ids, model.textures)
                              if not t and tx.type == 0)
                if missing:
                    self.report(
                        {"WARNING"},
                        "Exported, but %d of %d texture(s) have no FileDataID: "
                        "the model will be untextured in-game. Set 'Texture "
                        "FileDataID', or an 'm2_texture_id' custom property on "
                        "each material." % (missing, len(tex_ids)))
                else:
                    self.report({"INFO"}, "Exported new model: .m2 + .skin "
                                "(SFID=%s, TXID=%s)" % (skin_ids, tex_ids))
                return {"FINISHED"}

            tried = []
            src = self._source_m2(context, tried)
            model = scene_reader.read_scene(
                objects, mirror_x=self.mirror_x, source_m2=src,
                fps=context.scene.render.fps or 30, tried=tried)
            writer.scale_model(model, self.export_scale)
            if self.body_texture_id:
                n = 0
                for t in model.textures:
                    if t.type == 1:
                        t.type = 0
                        t.file_data_id = self.body_texture_id
                        t.filename = ""
                        n += 1
                print("[M2] hardcoded %d skin texture(s) -> FileDataID %d"
                      % (n, self.body_texture_id), flush=True)

            if chunked and self.full_custom:
                if not self.skin_file_id:
                    self.report({"ERROR"}, "Full Custom needs a Skin FileDataID: "
                                "register your 00.skin under it.")
                    return {"CANCELLED"}
                if not self.skel_file_id:
                    self.report({"ERROR"}, "Full Custom needs a Skeleton FileDataID: "
                                "register your .skel under it (the M2's SKID points at it).")
                    return {"CANCELLED"}
                # Optional: hardcode the skin texture to a BLP you supply.
                if self.body_texture_id:
                    for t in model.textures:
                        if t.type == 1:
                            t.type = 0
                            t.file_data_id = self.body_texture_id
                            t.filename = ""
                tex_fids = [t.file_data_id for t in model.textures]
                phys_bytes = _build_phys_bytes(context, objects, self.mirror_x, self, model)
                skel_bytes = writer.write_skel(model)
                m2_bytes, skin_bytes = writer.write_custom_character(
                    model, self.skel_file_id, [self.skin_file_id],
                    texture_file_ids=tex_fids, version=self.m2_version)
                bbase = self.filepath[:-3] if self.filepath.lower().endswith(".m2") else self.filepath
                with open(self.filepath, "wb") as f:
                    f.write(m2_bytes)
                with open(bbase + ".skel", "wb") as f:
                    f.write(skel_bytes)
                if skin_bytes is not None:
                    with open(bbase + "00.skin", "wb") as f:
                        f.write(skin_bytes)
                if phys_bytes is not None:
                    with open(bbase + ".phys", "wb") as f:
                        f.write(phys_bytes)
                tx = ", ".join(str(t.file_data_id) for t in model.textures if t.file_data_id)
                print("[M2] standalone custom: .m2 + .skel + 00.skin (single-LOD) | "
                      "SKID=%d SFID=%d TXID=[%s]" % (self.skel_file_id, self.skin_file_id, tx),
                      flush=True)
                self.report({"INFO"}, "Standalone export: .m2 + .skel + 00.skin "
                            "(SKID=%d, SFID=%d)" % (self.skel_file_id, self.skin_file_id))
                return {"FINISHED"}

            sfids = None
            tex_fids = None
            lod_count = 1
            if chunked:
                n_lod = max(1, len(model.skin_file_ids))
                lod_count = n_lod
                if self.skin_file_id:
                    sfids = [self.skin_file_id] * n_lod
                else:
                    sfids = list(model.skin_file_ids) or None
                tex_fids = [t.file_data_id for t in model.textures]

            base = bpy.path.abspath(self.base_m2_path) if self.base_m2_path else ""
            if not (base and os.path.isfile(base)):
                src = bpy.path.abspath(model.source_path) if model.source_path else ""
                if src and os.path.isfile(src):
                    base = src
            patched = False
            if chunked and base and os.path.isfile(base):
                import shutil
                with open(base, "rb") as f:
                    base_bytes = f.read()
                # Mesh-data-only: only the vertex pos/normal/uv are written back
                # into the original M2's slots; everything else stays byte-identical.
                updates = scene_reader.read_vertex_updates(objects, mirror_x=self.mirror_x)
                vcount = writer.vertex_count(base_bytes)
                in_place = updates and max(updates) < vcount
                if in_place:
                    m2_bytes, applied, _ = writer.patch_m2_vertices(base_bytes, updates)
                    with open(self.filepath, "wb") as f:
                        f.write(m2_bytes)
                    _copy_base_skins(base, self.filepath)
                    print("[M2] in-place patch of %s: %d/%d vertices updated; "
                          "original skins copied" % (os.path.basename(base), applied, vcount),
                          flush=True)
                    patched = True
                else:
                    # Different model/topology: splice geometry, keep target's structure.
                    lod_count = writer.sfid_count(base_bytes)
                    patch_sfids = [self.skin_file_id] * lod_count if self.skin_file_id else None
                    m2_bytes, skin_bytes = writer.patch_m2(base_bytes, model, skin_file_ids=patch_sfids)
                    with open(self.filepath, "wb") as f:
                        f.write(m2_bytes)
                    if skin_bytes is not None:
                        b = self.filepath[:-3] if self.filepath.lower().endswith(".m2") else self.filepath
                        writer.write_skin_lods(b, skin_bytes, lod_count)
                    print("[M2] patched geometry into %s (%d LODs)"
                          % (os.path.basename(base), lod_count), flush=True)
                    patched = True
            if not patched:
                if chunked:
                    phys_bytes = _build_phys_bytes(context, objects, self.mirror_x, self, model)
                    if phys_bytes is not None:
                        pb = (self.filepath[:-3] if self.filepath.lower().endswith(".m2")
                              else self.filepath)
                        with open(pb + ".phys", "wb") as f:
                            f.write(phys_bytes)
                writer.write_model(
                    model, self.filepath, chunked=chunked,
                    version=self.m2_version if chunked else 264,
                    skin_file_ids=sfids, texture_file_ids=tex_fids,
                    lod_count=lod_count)
        except scene_reader.ExportError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.report({"ERROR"}, f"Export failed: {exc}")
            return {"CANCELLED"}

        tris = len(model.skin.triangles) // 3 if model.skin else 0
        if chunked:
            sf = sfids[0] if sfids else None
            extra = (f"; SFID={sfids}" if sf else
                     "; no Skin FileDataID set: set one and register the .skin")
            tx = ", ".join(str(t.file_data_id) for t in model.textures if t.file_data_id)
            print("[M2] export TXID FileDataIDs: %s" % (tx or "(none)"), flush=True)
            lod_names = "00.skin" + ((" + _lod01..%02d.skin" % (lod_count - 1))
                                     if lod_count > 1 else "")
            print("[M2] wrote %d skin LOD file(s): %s" % (lod_count, lod_names), flush=True)
        else:
            extra = ""
        self.report(
            {"INFO"},
            f"Exported {os.path.basename(self.filepath)} [{self.container}]: "
            f"{len(model.vertices)} verts, {tris} tris, {len(model.bones)} bones, "
            f"{len(model.sequences)} anims (+ {lod_count} skin LOD{'s' if lod_count>1 else ''}){extra}")
        if patched:
            from .m2_import import phys_rig
            if phys_rig.rig_collections():
                self.report({"WARNING"},
                            "Patch mode keeps the target's own physics: the rig in this "
                            "scene was NOT written. Export with New Model or Full Custom "
                            "to include it.")
        return {"FINISHED"}


class IMPORT_SCENE_FH_wow_m2(bpy.types.FileHandler):
    bl_idname = "IMPORT_SCENE_FH_wow_m2"
    bl_label = "WoW M2"
    bl_import_operator = "import_scene.wow_m2"
    bl_file_extensions = ".m2"

    @classmethod
    def poll_drop(cls, context):
        return context.area is not None and context.area.type == "VIEW_3D"


def _menu_func_import(self, context):
    self.layout.operator(IMPORT_SCENE_OT_wow_m2.bl_idname, text="WoW Model (.m2)")


def _menu_func_export(self, context):
    self.layout.operator(EXPORT_SCENE_OT_wow_m2.bl_idname, text="WoW Model (.m2)")


_classes = (
    IMPORT_SCENE_OT_wow_m2,
    EXPORT_SCENE_OT_wow_m2,
    IMPORT_SCENE_FH_wow_m2,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(_menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(_menu_func_export)
    m2_ui.register()
    m2_physics_ui.register()


def unregister():
    m2_physics_ui.unregister()
    m2_ui.unregister()
    bpy.types.TOPBAR_MT_file_export.remove(_menu_func_export)
    bpy.types.TOPBAR_MT_file_import.remove(_menu_func_import)
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
