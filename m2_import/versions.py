"""Maps the M2 ``version`` field to a WoW expansion."""

# Canonical expansion keys used throughout the add-on.
VANILLA = "VANILLA"
TBC = "TBC"
WRATH = "WRATH"
CATA = "CATA"
MOP_WOD = "MOP_WOD"
LEGION = "LEGION"  # Legion, BfA, Shadowlands, DF (chunked MD21)

EXPANSION_LABELS = {
    VANILLA: "Classic / Vanilla (v256-257)",
    TBC: "The Burning Crusade (v260-263)",
    WRATH: "Wrath of the Lich King (v264)",
    CATA: "Cataclysm (v265-272)",
    MOP_WOD: "Mists of Pandaria / Warlords of Draenor (v272)",
    LEGION: "Legion / BfA / Shadowlands+ (chunked, v272-274)",
}

# Order matters for the import dialog's dropdown.
EXPANSION_ORDER = [VANILLA, TBC, WRATH, CATA, MOP_WOD, LEGION]


def expansion_from_version(version: int) -> str:
    """Best-effort mapping from numeric version to expansion key."""
    if version <= 257:
        return VANILLA
    if version <= 263:
        return TBC
    if version == 264:
        return WRATH
    if 265 <= version <= 272:
        # 265-271 is firmly Cata. 272 is ambiguous (Cata / MoP / WoD / Legion);
        # the chunked detector in the loader overrides this when MD21 is seen.
        return CATA if version < 272 else MOP_WOD
    # 273-274 only ever appear in the chunked Legion-era format.
    return LEGION
