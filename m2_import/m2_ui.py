"""In-viewport UI for M2 material settings and geoset assignment."""

import bpy
from bpy.props import (
    IntProperty, FloatProperty, BoolProperty, EnumProperty, PointerProperty,
    StringProperty,
)
from bpy.types import Panel, Operator, PropertyGroup


# ---------------------------------------------------------------------------
TEXTURE_TYPE_ITEMS = [
    ("0",  "File (BLP)",         "A real texture file: set the FileDataID below"),
    ("1",  "Skin (composited)",  "Character body skin composited by the client"),
    ("2",  "Object Skin",        "Cloak / object skin composited by the client"),
    ("3",  "Weapon Blade",       "Weapon blade texture (client-supplied)"),
    ("4",  "Weapon Handle",      "Weapon handle texture (client-supplied)"),
    ("5",  "Environment",        "Specular / reflection environment map"),
    ("6",  "Character Hair",     "Client-supplied hair texture"),
    ("7",  "Facial Hair",        "Client-supplied facial-hair texture"),
    ("8",  "Skin Extra",         "Extra composited skin layer"),
    ("9",  "UI Skin",            "UI-only skin (paper doll / preview)"),
    ("10", "Char Accessory (Mane/Leaves)", "Client-supplied accessory slot: originally tauren manes, also nightelf hair leaves, druid foliage, etc."),
    ("11", "Monster Skin 1",     "Creature skin slot 1"),
    ("12", "Monster Skin 2",     "Creature skin slot 2"),
    ("13", "Monster Skin 3",     "Creature skin slot 3"),
    ("14", "Item Icon",          "Item icon texture"),
    ("15", "Guild Background Color", "Guild tabard background colour (Cata+)"),
    ("16", "Guild Emblem Color", "Guild tabard emblem colour (Cata+)"),
    ("17", "Guild Border Color", "Guild tabard border colour (Cata+)"),
    ("18", "Guild Emblem",       "Guild tabard emblem (Cata+)"),
    ("19", "Character Eyes",     "Eye texture chosen by the eye-colour customization (Shadowlands+)"),
    ("20", "Character Jewelry",  "Accessory / jewelry texture from customization (Shadowlands+)"),
    ("21", "Character Secondary Skin", "Second skin layer from customization (Shadowlands+)"),
    ("22", "Character Secondary Hair", "Second hair layer from customization (Shadowlands+)"),
    ("23", "Character Secondary Armor", "Second armor layer from customization (Shadowlands+)"),
    ("24", "Type 24",            "Shadowlands+ customization slot (undocumented)"),
    ("25", "Type 25",            "Seen on Dracthyr (10.0+), undocumented"),
    ("26", "Type 26",            "Seen on Dracthyr (10.0+), undocumented"),
]

BLEND_MODE_ITEMS = [
    ("OPAQUE", "Opaque", "No blending"),
    ("ALPHA_KEY", "Alpha Key", "1-bit cutout (hair edges, foliage)"),
    ("ALPHA", "Alpha Blend", "Smooth transparency"),
    ("NO_ALPHA_ADD", "No-Alpha Add", "Additive, ignore alpha"),
    ("ADD", "Add (glow)", "Additive: glows, black becomes invisible"),
    ("MOD", "Modulate", "Multiply"),
    ("MOD2X", "Modulate 2x", "Multiply, doubled"),
    ("BLEND_ADD", "Blend Add", "Alpha-weighted additive"),
]

# Common geoset groups; id = group*100 + variant.
GEOSET_GROUP_ITEMS = [
    ("0", "Body / Hair (0)", "0=base body, 1-99=hairstyles"),
    ("4", "Gloves (4xx)", "equipment-gated"),
    ("5", "Boots (5xx)", "equipment-gated"),
    ("7", "Ears (7xx)", ""),
    ("8", "Sleeves (8xx)", "equipment-gated"),
    ("9", "Leg cuffs (9xx)", "equipment-gated"),
    ("10", "Chest (10xx)", "equipment-gated"),
    ("11", "Pants (11xx)", "equipment-gated"),
    ("12", "Tabard (12xx)", "equipment-gated"),
    ("13", "Trousers/Robe (13xx)", "equipment-gated"),
    ("15", "Cloak (15xx)", "equipment-gated"),
    ("18", "Belt (18xx)", "equipment-gated"),
    ("CUSTOM", "Custom id", "Type the full geoset id yourself"),
]


# ---------------------------------------------------------------------------
def _sync_material(mat):
    """Write a material's M2 property group to the exporter's custom props."""
    p = mat.m2
    if "m2_auto_imported" in mat:
        try:
            del mat["m2_auto_imported"]
        except Exception:  # noqa: BLE001
            mat["m2_auto_imported"] = 0
    # A material can draw several texture layers (retail eyes and eye glow use
    # two, tabards six). The panel edits the first, and layer 2 when present;
    # any further layers are kept exactly as imported.
    types = _csv_list(mat.get("m2_texture_types"))
    ids = _csv_list(mat.get("m2_texture_ids"))
    n = max(len(types), len(ids), 1)
    types = (types + ["0"] * n)[:n]
    ids = (ids + ["0"] * n)[:n]
    types[0] = str(int(p.texture_type))
    ids[0] = str(int(p.texture_id)) if p.texture_type == "0" else "0"
    if p.layer2_enabled:
        if n < 2:
            types.append("0"); ids.append("0"); n = 2
        types[1] = str(int(p.layer2_type))
        ids[1] = str(int(p.layer2_id)) if p.layer2_type == "0" else "0"
    elif n >= 2 and p.layer2_seen:
        types, ids = types[:1], ids[:1]            # user removed the second layer
    mat["m2_texture_types"] = ",".join(types)
    mat["m2_texture_ids"] = ",".join(ids)
    if p.texture_type == "0":
        # Hardcode a texture file path (baked into the M2's inline
        # texture record so the client loads the .blp by path — no FID
        # lookup needed). Absolute or //-relative to the .blend; the
        # exporter passes the string through verbatim as the M2Texture
        # filename field.
        path = (p.texture_path or "").strip()
        if path:
            mat["m2_texture_paths"] = path
        elif "m2_texture_paths" in mat:
            del mat["m2_texture_paths"]
    mat["m2_blend_mode"] = p.blend_mode
    mat["m2_transparency"] = float(p.transparency)
    mat["m2_shader_id"] = int(p.shader_id)
    flags = 0
    if p.two_sided:
        flags |= 0x04
    if p.unlit:
        flags |= 0x01
    if p.unfogged:
        flags |= 0x02
    if p.no_depth_test:
        flags |= 0x08
    if p.no_depth_write:
        flags |= 0x10
    mat["m2_render_flags"] = flags
    # Individual flag props too so alternate exporter paths still read them.
    mat["m2_two_sided"] = int(p.two_sided)
    mat["m2_unlit"] = int(p.unlit)
    mat["m2_unfogged"] = int(p.unfogged)
    mat["m2_no_depth_test"] = int(p.no_depth_test)
    mat["m2_no_depth_write"] = int(p.no_depth_write)


SUPPRESS_MATERIAL_UPDATE = False


def _csv_list(value):
    return [p.strip() for p in str(value or "").split(",") if p.strip()]


def _on_material_update(self, context):
    if SUPPRESS_MATERIAL_UPDATE:
        return
    mat = getattr(self, "id_data", None)
    if isinstance(mat, bpy.types.Material):
        _sync_material(mat)


# int blend_mode -> BLEND_MODE_ITEMS token used by the panel enum.
_BLEND_INT_TO_TOKEN = {
    0: "OPAQUE", 1: "ALPHA_KEY", 2: "ALPHA", 3: "NO_ALPHA_ADD",
    4: "ADD", 5: "MOD", 6: "MOD2X", 7: "BLEND_ADD",
}
# Which type values the enum can represent. Anything else falls back to "1".
_TEXTURE_TYPE_ENUM_VALUES = {item[0] for item in TEXTURE_TYPE_ITEMS}


def sync_props_from_customprops(mat):
    """Pull the exporter's custom properties (m2_texture_types, m2_blend_mode, ...)"""
    if mat is None or not hasattr(mat, "m2"):
        return
    global SUPPRESS_MATERIAL_UPDATE
    prev = SUPPRESS_MATERIAL_UPDATE
    SUPPRESS_MATERIAL_UPDATE = True
    try:
        p = mat.m2
        # Texture type: read first CSV entry, coerce to a value the enum knows.
        raw_types = str(mat.get("m2_texture_types", "") or "")
        first_type = ""
        for part in raw_types.split(","):
            part = part.strip()
            if part:
                first_type = part
                break
        if first_type in _TEXTURE_TYPE_ENUM_VALUES:
            p.texture_type = first_type
        # Texture id (only meaningful when type=0 File, but harmless to set).
        raw_ids = str(mat.get("m2_texture_ids", "") or "")
        first_id = 0
        for part in raw_ids.split(","):
            part = part.strip()
            if part.lstrip("-").isdigit():
                first_id = int(part)
                break
        p.texture_id = first_id
        # Re-hydrate the hardcoded texture path from the material's
        # custom prop after a .blend reload or re-import (props on the
        # PropertyGroup itself don't persist across a fresh Blender
        # session, only the underlying custom props do).
        raw_path = mat.get("m2_texture_paths")
        if raw_path is not None:
            p.texture_path = str(raw_path)
        # Second texture layer, when the material has one.
        types = _csv_list(mat.get("m2_texture_types"))
        ids = _csv_list(mat.get("m2_texture_ids"))
        has2 = len(types) >= 2 or len(ids) >= 2
        p.layer2_enabled = has2
        p.layer2_seen = has2
        if has2:
            t2 = types[1] if len(types) >= 2 else "0"
            p.layer2_type = t2 if t2 in _TEXTURE_TYPE_ENUM_VALUES else "0"
            i2 = ids[1] if len(ids) >= 2 else "0"
            p.layer2_id = int(i2) if i2.lstrip("-").isdigit() else 0
        p.extra_layers = max(0, max(len(types), len(ids)) - 2)
        # Blend: accept int or already-token string.
        bm = mat.get("m2_blend_mode")
        if isinstance(bm, str):
            if bm.upper() in {t for t in _BLEND_INT_TO_TOKEN.values()}:
                p.blend_mode = bm.upper()
        elif bm is not None:
            token = _BLEND_INT_TO_TOKEN.get(int(bm))
            if token:
                p.blend_mode = token
        # Shader / transparency / render flags.
        p.shader_id = int(mat.get("m2_shader_id", 0) or 0)
        tr = mat.get("m2_transparency")
        if tr is not None:
            try:
                p.transparency = max(0.0, min(1.0, float(tr)))
            except Exception:  # noqa: BLE001
                pass
        flags = int(mat.get("m2_render_flags", 0) or 0)
        p.unlit = bool(flags & 0x01)
        p.unfogged = bool(flags & 0x02)
        p.two_sided = bool(flags & 0x04)
        p.no_depth_test = bool(flags & 0x08)
        p.no_depth_write = bool(flags & 0x10)
    finally:
        SUPPRESS_MATERIAL_UPDATE = prev


class M2MaterialProps(PropertyGroup):
    texture_type: EnumProperty(
        name="Texture Type", items=TEXTURE_TYPE_ITEMS, default="1",
        update=_on_material_update)
    texture_id: IntProperty(
        name="Texture FileDataID", min=0, default=0,
        description="BLP FileDataID (only used when Texture Type is File)",
        update=_on_material_update)
    texture_path: StringProperty(
        name="Texture Path",
        description=("Hardcode a .blp file path for this material's texture. "
                     "Written into the M2's inline texture record on export so "
                     "the client loads this exact file. Leave empty to use the "
                     "FileDataID field instead."),
        default="", subtype="FILE_PATH", update=_on_material_update)
    layer2_enabled: BoolProperty(
        name="Second Texture Layer", default=False,
        description="Draw a second texture with this material (eyes and eye glow use two "
                    "layers of the same texture with a multi-layer Shader ID)",
        update=_on_material_update)
    layer2_seen: BoolProperty(default=False)
    layer2_type: EnumProperty(
        name="Layer 2 Type", items=TEXTURE_TYPE_ITEMS, default="0",
        update=_on_material_update)
    layer2_id: IntProperty(
        name="Layer 2 FileDataID", min=0, default=0, update=_on_material_update)
    extra_layers: IntProperty(default=0)
    blend_mode: EnumProperty(
        name="Blend", items=BLEND_MODE_ITEMS, default="OPAQUE",
        update=_on_material_update)
    transparency: FloatProperty(
        name="Opacity", min=0.0, max=1.0, default=1.0, subtype="FACTOR",
        description="1 = opaque, 0 = invisible. Below 1 forces an alpha blend",
        update=_on_material_update)
    shader_id: IntProperty(
        name="Shader ID", min=0, default=0, update=_on_material_update)
    two_sided: BoolProperty(name="Two-Sided", default=False,
                            update=_on_material_update)
    unlit: BoolProperty(name="Unlit (full-bright)", default=False,
                        update=_on_material_update)
    unfogged: BoolProperty(name="Unfogged", default=False,
                           update=_on_material_update)
    no_depth_test: BoolProperty(name="No Depth Test", default=False,
                                update=_on_material_update)
    no_depth_write: BoolProperty(name="No Depth Write", default=False,
                                 update=_on_material_update)


# Guard against ping-pong: picker->active writes set this; _sync_picker_to_active
_SYNC_LOCK = [False]


def _apply_geoset_to_active(scene):
    """Write scene.m2_geoset picker values onto the active mesh object."""
    if _SYNC_LOCK[0]:
        return
    obj = bpy.context.view_layer.objects.active if bpy.context else None
    if obj is None or obj.type != "MESH":
        return
    g = scene.m2_geoset
    sid = g.resolved_id()
    grp = sid // 100
    var = sid % 100
    obj["m2_skin_section_id"] = sid
    obj["m2_geoset_group"] = grp
    obj["m2_geoset_variant"] = var
    # Rename if the object matches our _geoset_NNNN pattern.
    import re as _re
    m = _re.match(r"(.*?)_geoset_\d+(\.\d+)?$", obj.name)
    if m:
        obj.name = "%s_geoset_%04d%s" % (m.group(1), sid, m.group(2) or "")


def _on_geoset_prop_update(self, context):
    scene = getattr(context, "scene", None) or bpy.context.scene
    if scene is not None:
        _apply_geoset_to_active(scene)


class M2GeosetProps(PropertyGroup):
    group: EnumProperty(name="Group", items=GEOSET_GROUP_ITEMS, default="0",
                        update=_on_geoset_prop_update)
    variant: IntProperty(name="Variant", min=0, max=99, default=1,
                         description="The x in group*100 + x (e.g. hair style 2)",
                         update=_on_geoset_prop_update)
    raw_id: IntProperty(name="Geoset ID", min=0, default=0,
                        description="Full geoset id, used when Group is Custom",
                        update=_on_geoset_prop_update)

    def resolved_id(self):
        if self.group == "CUSTOM":
            return int(self.raw_id)
        return int(self.group) * 100 + int(self.variant)


# ---------------------------------------------------------------------------
class M2_OT_add_material(Operator):
    bl_idname = "m2.add_material"
    bl_label = "Add M2 Material"
    bl_description = "Create a new material with M2 settings on the active object"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Select a mesh object first.")
            return {"CANCELLED"}
        mat = bpy.data.materials.new("M2_Material")
        mat.use_nodes = True
        obj.data.materials.append(mat)
        obj.active_material_index = len(obj.data.materials) - 1
        _sync_material(mat)
        self.report({"INFO"}, "Added M2 material '%s'." % mat.name)
        return {"FINISHED"}


class M2_OT_material_to_selected(Operator):
    bl_idname = "m2.material_to_selected"
    bl_label = "Apply Material to Selected"
    bl_description = ("Give every selected mesh the active object's active "
                     "material (its M2 settings and all)")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = context.active_object
        mat = obj.active_material if obj else None
        if mat is None:
            self.report({"ERROR"}, "The active object has no active material.")
            return {"CANCELLED"}
        _sync_material(mat)
        n = 0
        for o in context.selected_objects:
            if o.type != "MESH" or o is obj:
                continue
            o.data.materials.clear()
            o.data.materials.append(mat)
            for poly in o.data.polygons:
                poly.material_index = 0
            n += 1
        self.report({"INFO"}, "Applied '%s' to %d object(s)." % (mat.name, n))
        return {"FINISHED"}


class M2_OT_apply_geoset(Operator):
    bl_idname = "m2.apply_geoset"
    bl_label = "Rename to Geoset"
    bl_description = ("Rename the selected meshes to <base>_geoset_<id> and set "
                     "the m2_skin_section_id property so export uses this geoset")
    bl_options = {"REGISTER", "UNDO"}

    import re as _re
    _RE = _re.compile(r"(.*?)(?:[._]?geoset[_-]?\d+)?(\.\d+)?$", _re.IGNORECASE)

    def execute(self, context):
        gid = context.scene.m2_geoset.resolved_id()
        targets = [o for o in context.selected_objects if o.type == "MESH"]
        if not targets:
            self.report({"ERROR"}, "Select one or more mesh objects.")
            return {"CANCELLED"}
        for o in targets:
            m = self._RE.match(o.name)
            base = (m.group(1) if m else o.name).rstrip("._") or "mesh"
            o.name = "%s_geoset_%04d" % (base, gid)
            o["m2_skin_section_id"] = gid
        self.report({"INFO"}, "Set %d object(s) to geoset %d." % (len(targets), gid))
        return {"FINISHED"}


# ---------------------------------------------------------------------------
class _M2PanelBase:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "M2"


class VIEW3D_PT_m2_material(_M2PanelBase, Panel):
    bl_idname = "VIEW3D_PT_m2_material"
    bl_label = "M2 Material"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            layout.label(text="Select a mesh object.", icon="INFO")
            return
        mat = obj.active_material
        if mat is None:
            layout.operator("m2.add_material", icon="ADD")
            return
        layout.label(text=mat.name, icon="MATERIAL")
        p = mat.m2
        col = layout.column(align=True)
        col.prop(p, "texture_type")
        if p.texture_type == "0":
            col.prop(p, "texture_id")
            col.prop(p, "texture_path")
            if p.texture_path:
                col.label(text="Path overrides FileDataID on export",
                          icon="INFO")
        col.prop(p, "layer2_enabled")
        if p.layer2_enabled:
            sub = col.column(align=True)
            sub.prop(p, "layer2_type", text="Layer 2")
            if p.layer2_type == "0":
                sub.prop(p, "layer2_id")
        if p.extra_layers:
            col.label(text="+%d more layer(s) kept from the import" % p.extra_layers, icon="INFO")
        col.prop(p, "blend_mode")
        col.prop(p, "transparency", slider=True)
        col.prop(p, "shader_id")

        box = layout.box()
        box.label(text="Render Flags")
        row = box.row(align=True)
        row.prop(p, "two_sided", toggle=True)
        row.prop(p, "unlit", toggle=True)
        row = box.row(align=True)
        row.prop(p, "unfogged", toggle=True)
        row.prop(p, "no_depth_test", toggle=True)
        box.prop(p, "no_depth_write", toggle=True)

        layout.operator("m2.add_material", icon="ADD")
        layout.operator("m2.material_to_selected", icon="COPYDOWN")


def _active_action(obj):
    """Return the AnimationData action currently assigned to `obj`, or None."""
    if obj is None:
        return None
    ad = getattr(obj, "animation_data", None)
    return ad.action if ad else None


# Well-known WoW sequence IDs so users don't have to memorize numbers.
# From wowdev.wiki AnimationData.dbc; not exhaustive but covers common cases.
_SEQ_ID_NAMES = {
    0: "Stand",
    1: "Death",
    2: "Spell",
    3: "Stop",
    4: "Walk",
    5: "Run",
    6: "Dead",
    26: "AttackUnarmed",
    27: "Attack1H",
    28: "Attack2H",
    29: "Attack2HL",
    38: "Sheath",
    39: "Unsheath",
    64: "AttackOff",
    68: "SheathOff",
    69: "UnsheathOff",
    75: "Jump",
    89: "Emote",
    140: "SheathHigh",
    141: "UnsheathHigh",
}


# Well-known key_bone_id -> human name. From wowdev.wiki M2 Bone.
_KEY_BONE_NAMES = {
    -1: "unlisted",
    0: "ArmL",
    1: "ArmR",
    2: "ShoulderL",
    3: "ShoulderR",
    4: "SpineLow",
    5: "Waist",
    6: "Head",
    7: "Jaw",
    8: "IndexFingerR",
    9: "MiddleFingerR",
    10: "PinkyFingerR",
    11: "RingFingerR",
    12: "ThumbR",
    13: "IndexFingerL",
    14: "MiddleFingerL",
    15: "PinkyFingerL",
    16: "RingFingerL",
    17: "ThumbL",
    18: "$BTH (head look)",
    19: "$CSR (right hand cast)",
    20: "$CSL (left hand cast)",
    21: "_Breath",
    22: "_Name",
    23: "_NameMount",
    24: "$CHD (head chin)",
    25: "$CCH (chest)",
    26: "Root",
    27: "Wheel1",
    28: "Wheel2",
    29: "Wheel3",
    30: "Wheel4",
    31: "Wheel5",
    32: "Wheel6",
    33: "Wheel7",
    34: "Wheel8",
}


def _iter_action_fcurves(action):
    """Yield every FCurve on `action` across Blender API versions. Try the 5.x"""
    layers = getattr(action, "layers", None) or ()
    if layers:
        for layer in layers:
            for strip in getattr(layer, "strips", ()) or ():
                bags = getattr(strip, "channelbags", None)
                if bags is not None:
                    for cbag in bags:
                        for fc in getattr(cbag, "fcurves", ()) or ():
                            yield fc
                    continue
                # Older 4.4-ish path: query per slot.
                slots = getattr(action, "slots", None) or ()
                for slot in slots:
                    try:
                        cbag = strip.channelbag(slot)
                    except Exception:  # noqa: BLE001
                        cbag = None
                    if cbag is None:
                        continue
                    for fc in getattr(cbag, "fcurves", ()) or ():
                        yield fc
        return
    # Legacy pre-4.4 Blender: fcurves lives directly on the action.
    for fc in getattr(action, "fcurves", ()) or ():
        yield fc


def _get_or_make_action_fcurves(action):
    """Return the fcurves collection to add new curves into, across API versions."""
    layers = getattr(action, "layers", None)
    if layers is None or len(layers) == 0:
        # Legacy: fcurves live directly on the action.
        return getattr(action, "fcurves", None)
    layer = layers[0]
    if len(layer.strips) == 0:
        strip = layer.strips.new(type="KEYFRAME")
    else:
        strip = layer.strips[0]
    slots = getattr(action, "slots", None) or ()
    if len(slots) == 0:
        try:
            slot = action.slots.new(id_type="OBJECT", name="Object")
        except TypeError:
            slot = action.slots.new("OBJECT", "Object")
    else:
        slot = slots[0]
    try:
        cbag = strip.channelbag(slot, ensure=True)
    except TypeError:
        cbag = strip.channelbag(slot)
    return cbag.fcurves


def _insert_bone_rotation_key(action, bone_name, frame, quat_wxyz):
    """Insert a 4-component rotation_quaternion keyframe on a bone at ``frame``."""
    fcurves = _get_or_make_action_fcurves(action)
    if fcurves is None:
        return False
    dp = 'pose.bones["%s"].rotation_quaternion' % bone_name
    for i, val in enumerate(quat_wxyz):
        fc = None
        for existing in fcurves:
            if existing.data_path == dp and existing.array_index == i:
                fc = existing
                break
        if fc is None:
            try:
                fc = fcurves.new(data_path=dp, index=i)
            except Exception:  # noqa: BLE001
                continue
        kp = fc.keyframe_points.insert(frame, val, options={"REPLACE"})
        try:
            kp.interpolation = "LINEAR"
        except Exception:  # noqa: BLE001
            pass
        fc.update()
    return True


class M2_OT_apply_pose_as_rest_all_actions(Operator):
    """Bake the current pose as a frame-0 keyframe across every action, then apply the pose as rest."""
    bl_idname = "m2.apply_pose_as_rest_all_actions"
    bl_label = "Keyframe Pose + Apply Rest (all actions)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return (context.active_object is not None
                and context.active_object.type == "ARMATURE"
                and context.mode == "POSE")

    def execute(self, context):
        from mathutils import Quaternion
        arm = context.active_object

        # 1. Snapshot each pose bone's rotation-delta in local space. Only
        # non-identity rotations are worth adjusting.
        pose_deltas = {}
        for pb in arm.pose.bones:
            if pb.rotation_mode == "QUATERNION":
                q = Quaternion(pb.rotation_quaternion)
            else:
                q = pb.rotation_euler.to_quaternion()
            # Skip identity (dot(q, identity) ~ 1) to save work.
            if abs(q.w - 1.0) < 1e-6 and q.x**2 + q.y**2 + q.z**2 < 1e-10:
                continue
            pose_deltas[pb.name] = q.copy()

        if not pose_deltas:
            self.report({"WARNING"}, "No non-identity pose rotations to apply.")
            return {"CANCELLED"}

        # 2. For each action, insert a frame-0 keyframe with the pose delta
        keyframed_actions = 0
        total_keyed = 0
        for act in bpy.data.actions:
            local_added = 0
            for pb_name, delta_q in pose_deltas.items():
                path_prefix = 'pose.bones["%s"]' % pb_name
                has_rot = False
                for fc in _iter_action_fcurves(act):
                    if fc.data_path.startswith(path_prefix) \
                            and fc.data_path.endswith("rotation_quaternion"):
                        has_rot = True
                        break
                if has_rot:
                    continue
                # No existing rotation curves - add frame-0 keys with delta_q.
                if not _insert_bone_rotation_key(act, pb_name, 0.0,
                                                 (delta_q.w, delta_q.x,
                                                  delta_q.y, delta_q.z)):
                    continue
                local_added += 1
            if local_added:
                keyframed_actions += 1
                total_keyed += local_added

        # 3. Apply pose as rest so Blender's viewport shows the new orientation
        # by default. The inserted keyframes carry the intent through to WoW.
        bpy.ops.pose.armature_apply(selected=False)

        self.report({"INFO"},
                    "Added frame-0 rotation key for %d bone(s) in %d action(s) "
                    "(%d keys). Applied pose as rest."
                    % (len(pose_deltas), keyframed_actions, total_keyed))
        return {"FINISHED"}


class VIEW3D_PT_m2_bone(_M2PanelBase, Panel):
    bl_idname = "VIEW3D_PT_m2_bone"
    bl_label = "M2 Bone"

    def draw(self, context):
        layout = self.layout
        arm = context.active_object
        if arm is None or arm.type != "ARMATURE":
            layout.label(text="Select an armature to see its bones.")
            return
        bone = context.active_bone or (context.active_pose_bone.bone
                                        if context.active_pose_bone else None)
        if bone is None:
            layout.label(text="No active bone.")
            return
        kbi = int(bone.get("m2_key_bone_id", -1))
        flags = int(bone.get("m2_bone_flags", 0))
        subm = int(bone.get("m2_submesh_id", 0))
        crc_str = bone.get("m2_name_crc", "")
        box = layout.box()
        box.label(text=bone.name, icon="BONE_DATA")
        if kbi >= 0 or kbi == -1:
            label = _KEY_BONE_NAMES.get(kbi, "custom (%d)" % kbi)
            box.label(text="Key Bone: %d - %s" % (kbi, label))
        box.label(text="Flags: 0x%X" % flags)
        if subm:
            box.label(text="Submesh: %d" % subm)
        if crc_str:
            box.label(text="Name CRC: %s" % crc_str)
        layout.separator()
        # Pose-mode-only convenience: apply the current pose as rest AND
        # re-base rotation keyframes across every action.
        row = layout.row()
        row.enabled = context.mode == "POSE"
        row.operator("m2.apply_pose_as_rest_all_actions", icon="POSE_HLT")


class VIEW3D_PT_m2_action(_M2PanelBase, Panel):
    bl_idname = "VIEW3D_PT_m2_action"
    bl_label = "M2 Action"

    def draw(self, context):
        layout = self.layout
        act = _active_action(context.active_object)
        if act is None:
            layout.label(text="No active action on the selected object.")
            return
        sid = int(act.get("m2_seq_id", -1))
        var = int(act.get("m2_seq_var", -1))
        idx = int(act.get("m2_seq_index", -1))
        dur = int(act.get("m2_seq_duration", 0))
        flags = int(act.get("m2_seq_flags", 0))
        if sid < 0:
            layout.label(text="Action '%s'" % act.name)
            layout.label(text="No M2 metadata (custom action).", icon="INFO")
            return
        box = layout.box()
        box.label(text=act.name, icon="ACTION")
        name = _SEQ_ID_NAMES.get(sid, "seq %d" % sid)
        row = box.row(align=True)
        row.label(text="ID: %d (%s)" % (sid, name))
        row.label(text="Variant: %d" % var)
        col = box.column(align=True)
        col.label(text="Sequence index: %d" % idx)
        col.label(text="Duration: %d ms" % dur)
        col.label(text="Flags: 0x%X" % flags)
        # Cross-fade into / out of this animation in game. 0 = hard snap.
        if "m2_seq_blend_time" in act.keys():
            bcol = box.column(align=True)
            bcol.prop(act, '["m2_seq_blend_time"]', text="Blend In (ms)")
            if "m2_seq_blend_out" in act.keys():
                bcol.prop(act, '["m2_seq_blend_out"]', text="Blend Out (ms)")
        else:
            box.label(text="Blend: 150 ms (default)")
            box.operator("m2.action_add_blend_time", icon="ADD")


class M2_OT_action_add_blend_time(Operator):
    """Add editable blend-in / blend-out times to the active action"""
    bl_idname = "m2.action_add_blend_time"
    bl_label = "Edit Blend Times"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        act = _active_action(context.active_object)
        if act is None:
            return {"CANCELLED"}
        act["m2_seq_blend_time"] = 150
        act["m2_seq_blend_out"] = 0
        return {"FINISHED"}


class M2_OT_load_geoset_from_active(Operator):
    """Copy the active mesh's imported geoset id into the picker so you can edit it."""
    bl_idname = "m2.load_geoset_from_active"
    bl_label = "Load from Active"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        active = context.active_object
        if active is None or active.type != "MESH":
            self.report({"ERROR"}, "Select a mesh first.")
            return {"CANCELLED"}
        if "m2_skin_section_id" not in active:
            self.report({"ERROR"}, "Active mesh has no m2_skin_section_id.")
            return {"CANCELLED"}
        sid = int(active["m2_skin_section_id"])
        g = context.scene.m2_geoset
        grp = sid // 100
        var = sid % 100
        # Match one of the enum items if the group is known; otherwise fall back
        # to CUSTOM with the raw id.
        known = {int(item[0]) for item in GEOSET_GROUP_ITEMS if item[0] != "CUSTOM"}
        if grp in known:
            g.group = str(grp)
            g.variant = min(max(var, 0), 99)
        else:
            g.group = "CUSTOM"
            g.raw_id = sid
        return {"FINISHED"}


class VIEW3D_PT_m2_geoset(_M2PanelBase, Panel):
    bl_idname = "VIEW3D_PT_m2_geoset"
    bl_label = "M2 Geoset"

    def draw(self, context):
        # NEVER write to any data-block or property during draw. Reads only.
        layout = self.layout
        active = context.active_object
        has_id = (active is not None
                  and active.type == "MESH"
                  and "m2_skin_section_id" in active)

        box = layout.box()
        if has_id:
            sid = int(active["m2_skin_section_id"])
            box.label(text="Active: %s" % active.name, icon="MESH_DATA")
            box.label(text="ID %d  (Group %d, Variant %d)"
                      % (sid, sid // 100, sid % 100))
        else:
            box.label(text="Select an imported geoset mesh.", icon="INFO")

        # Interactive picker: editing these dropdowns writes m2_skin_section_id /
        # m2_geoset_group / m2_geoset_variant onto the active object immediately.
        g = context.scene.m2_geoset
        col = layout.column(align=True)
        col.enabled = has_id
        col.prop(g, "group")
        if g.group == "CUSTOM":
            col.prop(g, "raw_id")
        else:
            col.prop(g, "variant")
        layout.label(text="Resolved id: %d" % g.resolved_id())

        # Bulk "apply to N selected" button: still useful for multi-select.
        n = sum(1 for o in context.selected_objects if o.type == "MESH")
        row = layout.row()
        row.enabled = n > 1
        row.operator("m2.apply_geoset", icon="SORTALPHA",
                     text="Apply to %d Selected" % n)


class M2_OT_sync_geoset_id_from_group_var(Operator):
    """After you edit Group or Variant, this recomputes m2_skin_section_id = group*100 + variant."""
    bl_idname = "m2.sync_geoset_id_from_group_var"
    bl_label = "Recalc ID from Group/Variant"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        active = context.active_object
        if active is None or active.type != "MESH":
            self.report({"ERROR"}, "Select a mesh first.")
            return {"CANCELLED"}
        grp = int(active.get("m2_geoset_group", 0))
        var = int(active.get("m2_geoset_variant", 0))
        sid = grp * 100 + var
        active["m2_skin_section_id"] = sid
        # Also rename to match the new id if the name matches our pattern.
        import re as _re
        m = _re.match(r"(.*?)_geoset_\d+(\.\d+)?$", active.name)
        if m:
            active.name = "%s_geoset_%04d%s" % (m.group(1), sid, m.group(2) or "")
        self.report({"INFO"}, "Set ID = %d (%d*100+%d)" % (sid, grp, var))
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Retarget M2I: move another rig's meshes onto the imported M2 skeleton.
def _poll_armature(self, obj):
    return obj is not None and obj.type == "ARMATURE"


class M2RetargetProps(PropertyGroup):
    m2_rig: PointerProperty(
        type=bpy.types.Object, poll=_poll_armature, name="M2 Rig",
        description="The imported M2 armature: its bone names, geoset "
                    "settings and materials are the reference")
    other_rig: PointerProperty(
        type=bpy.types.Object, poll=_poll_armature, name="M2I Rig",
        description="The M2I armature whose bones get renamed (paired with the M2 "
                    "bones by order) and whose meshes move onto the M2 rig")
    bone_distance: FloatProperty(
        name="Bone Tolerance", default=0.05, min=0.0, soft_max=1.0, precision=3,
        description="Largest head distance (model units) still accepted as the same bone "
                    "when a bone has to be found by position (rigs with different bone "
                    "counts, or vertex groups fixed by bone position)")
    mesh_method: EnumProperty(
        name="Match Meshes",
        items=[("AUTO", "By name, then shape", "A mesh named like a geoset (suffixes such as .001 ignored) "
                "takes that geoset; the rest are paired by shape"),
               ("NAME", "By name", "Only meshes named like a geoset are paired"),
               ("SHAPE", "By shape", "Pair by geometry: same vertices in the same place"),
               ("ORDER", "By order", "Pair index by index: meshes in name order against geosets in "
                "their M2 order. Use when names and shapes both changed")],
        default="AUTO")
    shape_distance: FloatProperty(
        name="Shape Tolerance", default=0.01, min=0.0, soft_max=1.0, precision=4,
        description="Largest mean vertex distance (model units) for two meshes "
                    "to count as the same geoset shape")
    take_names: BoolProperty(
        name="Take Over Names", default=True,
        description="Rename each converted mesh to the geoset it replaces "
                    "(the original gets a .replaced suffix)")
    originals: EnumProperty(
        name="Replaced Geosets",
        items=[("DELETE", "Delete", "Remove the original geoset meshes the converted ones replace"),
               ("HIDE", "Hide", "Keep them hidden (note: hidden meshes still export unless "
                "you export with Selected Only)"),
               ("KEEP", "Keep", "Leave them as they are")],
        default="DELETE")
    remove_other: BoolProperty(
        name="Remove M2I Rig", default=True,
        description="Delete the M2I armature once it has no children left, "
                    "so the exporter sees one skeleton")
    guess_from_weights: BoolProperty(
        name="Guess Lost Groups From Weights", default=True,
        description="A vertex group whose bone exists on no armature any more is "
                    "put on the M2 bone closest to the group's weighted centre")
    copy_uvs: BoolProperty(
        name="Copy Missing UV Sets", default=True,
        description="UV sets the replaced geoset has and the new mesh lacks (UVMap2 for "
                    "eye-glow layers) are transferred by position")
    repair_weights: BoolProperty(
        name="Repair Stray Weights", default=True,
        description="A vertex sharing no bone with its neighbours, or with no weight at all, "
                    "takes its neighbours' weights")


class M2_OT_retarget_fix_groups(Operator):
    """Re-point every vertex group that names no M2 bone: by the rest position of the bone it was made for (looked up on any other armature in the file), else by the group's weighted centre. Works on the selected meshes, or on all meshes of the M2 rig when none are selected"""
    bl_idname = "m2.retarget_fix_groups"
    bl_label = "Fix Vertex Groups by Bone Position"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from . import retarget as _rt
        p = context.scene.m2_retarget
        if p.m2_rig is None:
            self.report({"ERROR"}, "Pick the M2 rig first.")
            return {"CANCELLED"}
        meshes = [o for o in context.selected_objects if o.type == "MESH"]
        if not meshes:
            meshes = [o for o in p.m2_rig.children_recursive if o.type == "MESH"]
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        r = _rt.fix_groups(meshes, p.m2_rig, p.bone_distance, p.guess_from_weights,
                           log=lambda s: print(s, flush=True))
        msg = "%d vertex group(s) re-pointed on %d mesh(es), %d guessed from weights" % (
            r["groups_repositioned"], len(meshes), r["guessed"])
        if r["stale_groups"]:
            msg += "; %d still unresolved (see console)" % sum(len(v) for v in r["stale_groups"].values())
        self.report({"WARNING"} if r["stale_groups"] else {"INFO"}, msg)
        return {"FINISHED"}


class M2_OT_retarget_pick(Operator):
    """Fill the pickers from the selection: the armature that owns geoset meshes becomes the M2 rig"""
    bl_idname = "m2.retarget_pick"
    bl_label = "Use Selected Armatures"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from . import retarget as _rt
        arms = [o for o in context.selected_objects if o.type == "ARMATURE"]
        if len(arms) != 2:
            self.report({"ERROR"}, "Select exactly two armatures (%d selected)." % len(arms))
            return {"CANCELLED"}
        p = context.scene.m2_retarget
        a, b = arms
        if _rt.geoset_meshes(b) and not _rt.geoset_meshes(a):
            a, b = b, a
        elif not _rt.geoset_meshes(a) and not _rt.geoset_meshes(b):
            # Neither has geosets: the active one is the M2I rig.
            if context.active_object is a:
                a, b = b, a
        p.m2_rig, p.other_rig = a, b
        return {"FINISHED"}


class M2_OT_retarget(Operator):
    """Rename the M2I rig's bones to the M2 names (paired by order), move its meshes onto the M2 rig, and copy each replaced geoset's name, settings, materials and UV sets onto the matching mesh"""
    bl_idname = "m2.retarget"
    bl_label = "Retarget"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from . import retarget as _rt
        p = context.scene.m2_retarget
        if p.m2_rig is None or p.other_rig is None:
            self.report({"ERROR"}, "Pick the M2 rig and the M2I rig first.")
            return {"CANCELLED"}
        if p.m2_rig is p.other_rig:
            self.report({"ERROR"}, "The two rigs must be different armatures.")
            return {"CANCELLED"}
        if not _rt.geoset_meshes(p.m2_rig):
            self.report({"WARNING"}, "%s has no imported geoset meshes; only bones and "
                        "parenting will be transferred." % p.m2_rig.name)
        try:
            r = _rt.retarget(p.m2_rig, p.other_rig, bone_method="ORDER",
                             bone_distance=p.bone_distance, shape_distance_max=p.shape_distance,
                             take_names=p.take_names, originals=p.originals,
                             remove_other=p.remove_other, guess_from_weights=p.guess_from_weights,
                             mesh_method=p.mesh_method, copy_uvs=p.copy_uvs,
                             repair_weights=p.repair_weights,
                             log=lambda s: print(s, flush=True))
        except _rt.RetargetError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        msg = ("Bones %d matched (%d unmatched), %d vertex groups renamed, %d objects moved, "
               "%d meshes took over a geoset, %d unmatched, %d UV sets copied, %d stray vertices re-weighted"
               % (r["bones_matched"], len(r["bones_unmatched"]), r["groups_renamed"], r["moved"],
                  r["matched"], len(r["unmatched"]), r["uv_sets"], r["weights_repaired"]))
        if r["stale_groups"]:
            n = sum(len(v) for v in r["stale_groups"].values())
            msg += "; %d vertex group(s) on %d mesh(es) still name no M2 bone (see console)" % (n, len(r["stale_groups"]))
        self.report({"WARNING"} if r["unmatched"] or r["bones_unmatched"] or r["stale_groups"] else {"INFO"}, msg)
        if r["other_removed"]:
            p.other_rig = None
        return {"FINISHED"}


class VIEW3D_PT_m2_retarget(_M2PanelBase, Panel):
    bl_idname = "VIEW3D_PT_m2_retarget"
    bl_label = "Retarget M2I"

    def draw(self, context):
        layout = self.layout
        p = context.scene.m2_retarget
        col = layout.column(align=True)
        col.prop(p, "m2_rig")
        col.prop(p, "other_rig")
        layout.operator("m2.retarget_pick", icon="RESTRICT_SELECT_OFF")
        box = layout.box()
        box.prop(p, "mesh_method")
        if p.mesh_method in ("AUTO", "SHAPE"):
            box.prop(p, "shape_distance")
        box.prop(p, "take_names")
        box.prop(p, "originals")
        box.prop(p, "remove_other")
        box.prop(p, "guess_from_weights")
        box.prop(p, "copy_uvs")
        box.prop(p, "repair_weights")
        row = layout.row()
        row.scale_y = 1.4
        row.enabled = p.m2_rig is not None and p.other_rig is not None and p.m2_rig is not p.other_rig
        row.operator("m2.retarget", icon="ARMATURE_DATA", text="Retarget Meshes onto M2 Rig")
        row = layout.row()
        row.enabled = p.m2_rig is not None
        row.operator("m2.retarget_fix_groups", icon="GROUP_VERTEX")


_classes = (
    M2MaterialProps,
    M2GeosetProps,
    M2RetargetProps,
    M2_OT_retarget_pick,
    M2_OT_retarget,
    M2_OT_retarget_fix_groups,
    VIEW3D_PT_m2_retarget,
    M2_OT_add_material,
    M2_OT_material_to_selected,
    M2_OT_apply_geoset,
    M2_OT_load_geoset_from_active,
    M2_OT_sync_geoset_id_from_group_var,
    M2_OT_apply_pose_as_rest_all_actions,
    M2_OT_action_add_blend_time,
    VIEW3D_PT_m2_material,
    VIEW3D_PT_m2_bone,
    VIEW3D_PT_m2_action,
    VIEW3D_PT_m2_geoset,
)


_MSGBUS_OWNER = object()


def _sync_picker_to_active():
    """msgbus callback: whenever the active object changes, mirror its imported"""
    _SYNC_LOCK[0] = True
    try:
        obj = bpy.context.view_layer.objects.active
        if obj is None or obj.type != "MESH":
            return
        if "m2_skin_section_id" not in obj:
            return
        sid = int(obj["m2_skin_section_id"])
        grp = sid // 100
        var = sid % 100
        scene = bpy.context.scene
        if scene is None:
            return
        g = scene.m2_geoset
        known = {int(item[0]) for item in GEOSET_GROUP_ITEMS if item[0] != "CUSTOM"}
        if grp in known:
            new_group = str(grp)
            new_var = max(0, min(var, 99))
            if g.group != new_group:
                g.group = new_group
            if g.variant != new_var:
                g.variant = new_var
        else:
            if g.group != "CUSTOM":
                g.group = "CUSTOM"
            if g.raw_id != sid:
                g.raw_id = sid
    except Exception:  # noqa: BLE001 - never let callback break Blender
        pass
    finally:
        _SYNC_LOCK[0] = False


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Material.m2 = PointerProperty(type=M2MaterialProps)
    bpy.types.Scene.m2_geoset = PointerProperty(type=M2GeosetProps)
    bpy.types.Scene.m2_retarget = PointerProperty(type=M2RetargetProps)
    bpy.msgbus.subscribe_rna(
        key=(bpy.types.LayerObjects, "active"),
        owner=_MSGBUS_OWNER,
        args=(),
        notify=_sync_picker_to_active,
    )


def unregister():
    try:
        bpy.msgbus.clear_by_owner(_MSGBUS_OWNER)
    except Exception:  # noqa: BLE001
        pass
    del bpy.types.Scene.m2_retarget
    del bpy.types.Scene.m2_geoset
    del bpy.types.Material.m2
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
