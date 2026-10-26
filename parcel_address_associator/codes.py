"""G-NAF authority code lexicons and free-text synonyms.

Everything here is embedded so the tool needs no downloads. Codes follow the
G-NAF ``Authority_Code_*_AUT`` tables (CODE -> NAME). The ``*_WORDS`` maps are
what the free-text parser accepts and normalise back to G-NAF codes.
"""
from __future__ import annotations

from typing import Dict, Optional

# --------------------------------------------------------------------------
# Authority code tables (CODE -> NAME)
# --------------------------------------------------------------------------
FLAT_TYPE_AUT: Dict[str, str] = {
    "ANT": "ANTENNA", "APT": "APARTMENT", "ATM": "AUTOMATED TELLER MACHINE",
    "BBQ": "BARBECUE", "BLCK": "BLOCK", "BLDG": "BUILDING", "BNGW": "BUNGALOW",
    "BTSD": "BOATSHED", "CAGE": "CAGE", "CARP": "CARPARK", "CARS": "CARSPACE",
    "CLUB": "CLUB", "COOL": "COOLROOM", "CTGE": "COTTAGE", "DUPL": "DUPLEX",
    "FCTY": "FACTORY", "FLAT": "FLAT", "GRGE": "GARAGE", "HALL": "HALL",
    "HSE": "HOUSE", "KSK": "KIOSK", "LBBY": "LOBBY", "LOFT": "LOFT", "LOT": "LOT",
    "LSE": "LEASE", "MBTH": "MARINE BERTH", "MSNT": "MAISONETTE", "OFFC": "OFFICE",
    "PTHS": "PENTHOUSE", "REAR": "REAR", "RESV": "RESERVE", "ROOM": "ROOM",
    "SE": "SUITE", "SEC": "SECTION", "SHED": "SHED", "SHOP": "SHOP",
    "SHRM": "SHOWROOM", "SIGN": "SIGN", "SITE": "SITE", "STLL": "STALL",
    "STOR": "STORE", "STR": "STRATA UNIT", "STU": "STUDIO", "SUBS": "SUBSTATION",
    "TNCY": "TENANCY", "TNHS": "TOWNHOUSE", "TWR": "TOWER", "UNIT": "UNIT",
    "VLLA": "VILLA", "VLT": "VAULT", "WARD": "WARD", "WHSE": "WAREHOUSE",
    "WKSH": "WORKSHOP",
}

LEVEL_TYPE_AUT: Dict[str, str] = {
    "B": "BASEMENT", "FL": "FLOOR", "G": "GROUND", "L": "LEVEL", "LB": "LOBBY",
    "LG": "LOWER GROUND FLOOR", "LL": "LOWER LEVEL", "M": "MEZZANINE",
    "OD": "OBSERVATION DECK", "P": "PARKING", "PDM": "PODIUM", "PLF": "PLATFORM",
    "PTHS": "PENTHOUSE", "RT": "ROOFTOP", "SB": "SUB-BASEMENT",
    "UG": "UPPER GROUND FLOOR", "UL": "UPPER LEVEL",
}

# G-NAF stores the full word as CODE and the abbreviation as NAME.
STREET_TYPE_AUT: Dict[str, str] = {
    "ACCESS": "ACCS", "ALLEY": "ALLY", "ALLEYWAY": "ALWY", "AMBLE": "AMBL",
    "APPROACH": "APP", "ARCADE": "ARC", "ARTERIAL": "ARTL", "AVENUE": "AV",
    "BANK": "BANK", "BAY": "BAY", "BEACH": "BCH", "BOULEVARD": "BVD",
    "BRACE": "BRCE", "BRAE": "BRAE", "BRIDGE": "BDGE", "BROADWAY": "BDWY",
    "BROW": "BROW", "BYPASS": "BYPA", "CAUSEWAY": "CWAY", "CENTRE": "CTR",
    "CHASE": "CH", "CIRCLE": "CIR", "CIRCUIT": "CCT", "CIRCUS": "CRCS",
    "CLOSE": "CL", "CONCOURSE": "CON", "CORNER": "CNR", "CORSO": "CSO",
    "COURT": "CT", "COURTYARD": "CTYD", "COVE": "COVE", "CRESCENT": "CR",
    "CREST": "CRST", "CROSS": "CRSS", "CROSSING": "CRSG", "DALE": "DALE",
    "DRIVE": "DR", "EDGE": "EDGE", "ESPLANADE": "ESP", "ESTATE": "EST",
    "EXPRESSWAY": "EXP", "FAIRWAY": "FAWY", "FREEWAY": "FWY", "GARDENS": "GDNS",
    "GATE": "GTE", "GLADE": "GLDE", "GLEN": "GLEN", "GRANGE": "GRA",
    "GREEN": "GRN", "GROVE": "GR", "HEIGHTS": "HTS", "HIGHWAY": "HWY",
    "HILL": "HILL", "KEY": "KEY", "LANE": "LANE", "LINK": "LINK", "LOOP": "LOOP",
    "MALL": "MALL", "MEWS": "MEWS", "MOTORWAY": "MWY", "NOOK": "NOOK",
    "OUTLOOK": "OTLK", "PARADE": "PDE", "PARKWAY": "PWAY", "PASS": "PASS",
    "PATH": "PATH", "PLACE": "PL", "PLAZA": "PLZA", "POCKET": "PKT",
    "POINT": "PNT", "PROMENADE": "PROM", "QUAY": "QY", "QUAYS": "QYS",
    "RAMBLE": "RMBL", "RETREAT": "RTT", "RIDGE": "RDGE", "RISE": "RISE",
    "ROAD": "RD", "ROW": "ROW", "SQUARE": "SQ", "STEPS": "STPS",
    "STREET": "ST", "STRIP": "STRP", "TERRACE": "TCE", "TRACK": "TRK",
    "TRAIL": "TRL", "VIEW": "VIEW", "VISTA": "VSTA", "WALK": "WALK",
    "WAY": "WAY", "WHARF": "WHRF", "WYND": "WYND",
}

STREET_SUFFIX_AUT: Dict[str, str] = {
    "CN": "CENTRAL", "DE": "DEVIATION", "E": "EAST", "EX": "EXTENSION",
    "IN": "INNER", "LR": "LOWER", "ML": "MALL", "N": "NORTH", "NE": "NORTH EAST",
    "NW": "NORTH WEST", "OF": "OFF", "ON": "ON", "OP": "OVERPASS", "S": "SOUTH",
    "SE": "SOUTH EAST", "SW": "SOUTH WEST", "UP": "UPPER", "W": "WEST",
}

STATE_AUT: Dict[str, tuple] = {  # STATE_PID -> (STATE_NAME, STATE_ABBREVIATION)
    "1": ("NEW SOUTH WALES", "NSW"), "2": ("VICTORIA", "VIC"),
    "3": ("QUEENSLAND", "QLD"), "4": ("SOUTH AUSTRALIA", "SA"),
    "5": ("WESTERN AUSTRALIA", "WA"), "6": ("TASMANIA", "TAS"),
    "7": ("NORTHERN TERRITORY", "NT"), "8": ("AUSTRALIAN CAPITAL TERRITORY", "ACT"),
    "9": ("OTHER TERRITORIES", "OT"),
}
STATE_ABBREVS = {abbr for _, abbr in STATE_AUT.values()}
STATE_NAME_TO_ABBR = {name: abbr for name, abbr in STATE_AUT.values()}

GEOCODE_TYPE_AUT: Dict[str, str] = {
    "BAP": "BUILDING ACCESS POINT", "BC": "BUILDING CENTROID",
    "CDF": "CENTRE-LINE DROPPED FRONTAGE", "DF": "DRIVEWAY FRONTAGE",
    "EA": "EMERGENCY ACCESS", "EAS": "EMERGENCY ACCESS SECONDARY",
    "FC": "FRONTAGE CENTRE", "FCS": "FRONTAGE CENTRE SETBACK", "GG": "GAP GEOCODE",
    "LB": "LETTERBOX", "PAP": "PROPERTY ACCESS POINT",
    "PAPS": "PROPERTY ACCESS POINT SETBACK", "PC": "PROPERTY CENTROID",
    "PCM": "PROPERTY CENTROID MANUAL", "STL": "STREET LOCALITY",
    "UC": "UNIT CENTROID", "UCM": "UNIT CENTROID MANUAL", "UNK": "UNKNOWN",
}

ALIAS_TYPE_AUT: Dict[str, str] = {
    "AL": "ALTERNATIVE LOCALITY", "CD": "CONTRIBUTOR DEFINED",
    "FNNFS": "FLAT NUMBER - NO FIRST SUFFIX CORRELATION",
    "FPS": "FLAT PREFIX - SUFFIX CORRELATION", "LD": "LEVEL DUPLICATION",
    "MR": "MAINTENANCE REF", "RA": "RANGED ADDRESS", "SYN": "SYNONYM",
}

PS_JOIN_TYPE_AUT: Dict[str, str] = {"1": "AUTO", "2": "MANUAL"}

# Reverse lookups used when loading G-NAF Core (which carries names, not codes)
FLAT_NAME_TO_CODE = {v: k for k, v in FLAT_TYPE_AUT.items()}
LEVEL_NAME_TO_CODE = {v: k for k, v in LEVEL_TYPE_AUT.items()}
STREET_ABBR_TO_CODE = {v: k for k, v in STREET_TYPE_AUT.items()}
SUFFIX_NAME_TO_CODE = {v: k for k, v in STREET_SUFFIX_AUT.items()}
GEOCODE_NAME_TO_CODE = {v: k for k, v in GEOCODE_TYPE_AUT.items()}

# --------------------------------------------------------------------------
# Flat type classes: duplicates are only sought within compatible classes
# --------------------------------------------------------------------------
RESIDENTIAL = "RESIDENTIAL"          # stacked dwellings
ATTACHED = "ATTACHED"                # ground-related dwellings
COMMERCIAL = "COMMERCIAL"
ANCILLARY = "ANCILLARY"
OTHER = "OTHER"
LEVEL_ONLY = "LEVEL_ONLY"            # sub-address with a level but no flat

_FLAT_CLASS = {
    RESIDENTIAL: {"UNIT", "APT", "FLAT", "STU", "LOFT", "MSNT", "STR", "ROOM", "PTHS"},
    ATTACHED: {"TNHS", "VLLA", "DUPL", "HSE", "CTGE", "BNGW"},
    COMMERCIAL: {"SHOP", "SE", "OFFC", "TNCY", "SHRM", "KSK", "STLL", "FCTY",
                 "WHSE", "WKSH", "CLUB", "HALL"},
    ANCILLARY: {"CARS", "CARP", "GRGE", "CAGE", "SHED", "BTSD", "MBTH", "STOR",
                "COOL", "VLT", "SIGN", "ATM", "ANT", "SUBS", "BBQ", "LBBY"},
}
FLAT_CLASS: Dict[str, str] = {}
for _cls, _codes in _FLAT_CLASS.items():
    for _c in _codes:
        FLAT_CLASS[_c] = _cls


def flat_class(code: str) -> str:
    if not code:
        return LEVEL_ONLY
    return FLAT_CLASS.get(code, OTHER)


# Commercial sub-families: SHOP~TENANCY is a plausible synonym, SHOP~SUITE less so.
_COMMERCIAL_FAMILY = {
    "SHOP": {"RETAIL"}, "KSK": {"RETAIL"}, "STLL": {"RETAIL"}, "SHRM": {"RETAIL"},
    "TNCY": {"RETAIL", "OFFICE"}, "SE": {"OFFICE"}, "OFFC": {"OFFICE"},
    "FCTY": {"INDUSTRIAL"}, "WHSE": {"INDUSTRIAL"}, "WKSH": {"INDUSTRIAL"},
    "CLUB": {"VENUE"}, "HALL": {"VENUE"},
}


def class_compatibility(code_a: str, code_b: str) -> Optional[float]:
    """Log-odds adjustment for comparing two flat types, or None if incompatible."""
    if code_a == code_b:
        return 0.0
    ca, cb = flat_class(code_a), flat_class(code_b)
    if ca == cb == COMMERCIAL:
        fa, fb = _COMMERCIAL_FAMILY.get(code_a, set()), _COMMERCIAL_FAMILY.get(code_b, set())
        return -0.3 if fa & fb else -1.5
    if ca == cb and ca not in (OTHER, ANCILLARY):
        return -0.3
    if {ca, cb} == {RESIDENTIAL, ATTACHED}:
        return -1.5
    return None


# --------------------------------------------------------------------------
# Free-text vocabularies (word -> G-NAF code)
# --------------------------------------------------------------------------
FLAT_TYPE_WORDS: Dict[str, str] = {name: code for code, name in FLAT_TYPE_AUT.items()}
FLAT_TYPE_WORDS.update({code: code for code in FLAT_TYPE_AUT})
FLAT_TYPE_WORDS.update({
    "U": "UNIT", "UN": "UNIT", "APPT": "APT", "APART": "APT", "F": "FLAT",
    "SHP": "SHOP", "STE": "SE", "SUITE": "SE", "VILLA": "VLLA", "VIL": "VLLA",
    "TH": "TNHS", "TWNHS": "TNHS", "TOWNHOUSE": "TNHS", "PH": "PTHS",
    "PENTHOUSE": "PTHS", "OFF": "OFFC", "OFFICE": "OFFC", "TEN": "TNCY",
    "TENANCY": "TNCY", "CARSPACE": "CARS", "CARPARK": "CARP", "RM": "ROOM",
    "STUDIO": "STU", "DUPLEX": "DUPL", "WAREHOUSE": "WHSE", "FACTORY": "FCTY",
    "KIOSK": "KSK", "SHOWROOM": "SHRM", "STORE": "STOR", "GARAGE": "GRGE",
    "COTTAGE": "CTGE", "MAISONETTE": "MSNT", "LOBBY": "LBBY",
})
# Single letters and tokens that are too ambiguous to stand alone are only
# accepted when glued to a number (e.g. U12) or immediately followed by one.
FLAT_TYPE_AMBIGUOUS = {"U", "F", "TH", "PH", "SE", "LOT", "SITE", "REAR", "BLOCK",
                       "BLCK", "BUILDING", "BLDG", "TOWER", "TWR", "HOUSE", "HSE",
                       "SECTION", "SEC", "LEASE", "LSE"}

LEVEL_TYPE_WORDS: Dict[str, str] = {name: code for code, name in LEVEL_TYPE_AUT.items()}
LEVEL_TYPE_WORDS.update({
    "LEVEL": "L", "LVL": "L", "LEV": "L", "L": "L", "FLOOR": "FL", "FLR": "FL",
    "FL": "FL", "GROUND": "G", "GROUND FLOOR": "G", "GROUND LEVEL": "G", "GF": "G",
    "G": "G", "BASEMENT": "B", "BSMT": "B", "B": "B", "LOWER GROUND": "LG",
    "LOWER GROUND FLOOR": "LG", "LG": "LG", "UPPER GROUND": "UG",
    "UPPER GROUND FLOOR": "UG", "UG": "UG", "MEZZANINE": "M", "MEZZ": "M",
    "M": "M", "PODIUM": "PDM", "ROOFTOP": "RT", "ROOF": "RT",
    "SUB-BASEMENT": "SB", "SUB BASEMENT": "SB", "PARKING": "P",
})
# Level words that never need a number
LEVEL_TYPES_WITHOUT_NUMBER = {"G", "LG", "UG", "M", "RT", "LB"}

STREET_TYPE_WORDS: Dict[str, str] = {code: code for code in STREET_TYPE_AUT}
STREET_TYPE_WORDS.update(STREET_ABBR_TO_CODE)
STREET_TYPE_WORDS.update({
    "AVE": "AVENUE", "AV": "AVENUE", "BLVD": "BOULEVARD", "BVDE": "BOULEVARD",
    "CRES": "CRESCENT", "CRS": "CRESCENT", "CR": "CRESCENT", "CT": "COURT",
    "CRT": "COURT", "DRV": "DRIVE", "DR": "DRIVE", "ESP": "ESPLANADE",
    "GDN": "GARDENS", "GDNS": "GARDENS", "GRV": "GROVE", "HWY": "HIGHWAY",
    "LN": "LANE", "PDE": "PARADE", "PL": "PLACE", "PLZ": "PLAZA", "PROM": "PROMENADE",
    "RD": "ROAD", "SQ": "SQUARE", "ST": "STREET", "STR": "STREET", "TCE": "TERRACE",
    "TER": "TERRACE", "TRL": "TRAIL", "WY": "WAY", "CCT": "CIRCUIT", "CL": "CLOSE",
    "QY": "QUAY",
})

STREET_SUFFIX_WORDS: Dict[str, str] = {code: code for code in STREET_SUFFIX_AUT}
STREET_SUFFIX_WORDS.update(SUFFIX_NAME_TO_CODE)

# --------------------------------------------------------------------------
# Level ordinals: a comparable numeric position for each level designation
# --------------------------------------------------------------------------
# Unit-number prefixes that conventionally denote a level (G01, B02, LG1 ...)
LEVEL_PREFIX_ORDINAL: Dict[str, float] = {
    "G": 0.0, "GF": 0.0, "LG": -0.5, "UG": 0.5, "B": -1.0, "M": 0.5,
}


def level_ordinal(level_type: str, level_number: Optional[int]) -> Optional[float]:
    """Map a level designation to a sortable number (G=0, B1=-1, L3=3...).

    Returns None when the designation has no reliable vertical position.
    """
    t = (level_type or "").upper()
    n = level_number
    if t in ("L", "FL", "") and n is not None:
        return float(n)
    if t in ("G", "LB"):
        return 0.0
    if t == "UG":
        return 0.5
    if t == "LG":
        return -0.5
    if t == "M":
        return (n or 0) + 0.5
    if t == "B":
        return -float(n or 1)
    if t == "SB":
        return -float(n or 1) - 0.5
    if t == "RT":
        return 1000.0
    if t == "PTHS":
        return 999.0
    return None


def level_label(ordinal: Optional[float]) -> str:
    if ordinal is None:
        return "?"
    if ordinal == 0:
        return "G"
    if ordinal == 0.5:
        return "UG"
    if ordinal == -0.5:
        return "LG"
    if ordinal >= 999:
        return "RT" if ordinal >= 1000 else "PH"
    if ordinal < 0:
        return "B%d" % int(round(-ordinal))
    if ordinal != int(ordinal):
        return "M%d" % int(ordinal)
    return "L%d" % int(ordinal)
