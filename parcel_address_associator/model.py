"""Input schema and the Address record the engine works on."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, List

from . import codes

# Input DataFrame columns, in order.
INPUT_COLUMNS: List[str] = [
    "ADDR_ID", "SOURCE_ID", "BUILDING_NAME",
    "LOT_NUMBER_PREFIX", "LOT_NUMBER", "LOT_NUMBER_SUFFIX",
    "FLAT_TYPE_CODE", "FLAT_NUMBER_PREFIX", "FLAT_NUMBER", "FLAT_NUMBER_SUFFIX",
    "LEVEL_TYPE_CODE", "LEVEL_NUMBER_PREFIX", "LEVEL_NUMBER", "LEVEL_NUMBER_SUFFIX",
    "NUMBER_FIRST_PREFIX", "NUMBER_FIRST", "NUMBER_FIRST_SUFFIX",
    "NUMBER_LAST_PREFIX", "NUMBER_LAST", "NUMBER_LAST_SUFFIX",
    "STREET_NAME", "STREET_TYPE_CODE", "STREET_SUFFIX_CODE",
    "LOCALITY_NAME", "STATE_ABBREVIATION", "POSTCODE", "CADASTRE_POLYGON_PID",
]
REQUIRED_COLUMNS = ["ADDR_ID", "CADASTRE_POLYGON_PID"]
# Optional: used only by the residential-use reconciliation, never by matching.
OPTIONAL_COLUMNS: List[str] = ["ADDRESS_TYPE"]

# input column -> Address attribute
COLUMN_TO_FIELD: Dict[str, str] = {
    "ADDR_ID": "addr_id", "SOURCE_ID": "source_id", "CADASTRE_POLYGON_PID": "parcel",
    "BUILDING_NAME": "building_name",
    "LOT_NUMBER_PREFIX": "lot_number_prefix", "LOT_NUMBER": "lot_number",
    "LOT_NUMBER_SUFFIX": "lot_number_suffix",
    "FLAT_TYPE_CODE": "flat_type", "FLAT_NUMBER_PREFIX": "flat_prefix",
    "FLAT_NUMBER": "flat_number", "FLAT_NUMBER_SUFFIX": "flat_suffix",
    "LEVEL_TYPE_CODE": "level_type", "LEVEL_NUMBER_PREFIX": "level_prefix",
    "LEVEL_NUMBER": "level_number", "LEVEL_NUMBER_SUFFIX": "level_suffix",
    "NUMBER_FIRST_PREFIX": "number_first_prefix", "NUMBER_FIRST": "number_first",
    "NUMBER_FIRST_SUFFIX": "number_first_suffix", "NUMBER_LAST_PREFIX": "number_last_prefix",
    "NUMBER_LAST": "number_last", "NUMBER_LAST_SUFFIX": "number_last_suffix",
    "STREET_NAME": "street_name", "STREET_TYPE_CODE": "street_type",
    "STREET_SUFFIX_CODE": "street_suffix", "LOCALITY_NAME": "locality_name",
    "STATE_ABBREVIATION": "state", "POSTCODE": "postcode", "ADDRESS_TYPE": "address_type",
}


@dataclass
class Address:
    """One input row. All values are clean upper-case strings ('' when absent)."""

    addr_id: str
    source_id: str = ""
    parcel: str = ""
    building_name: str = ""
    lot_number_prefix: str = ""
    lot_number: str = ""
    lot_number_suffix: str = ""
    flat_type: str = ""
    flat_prefix: str = ""
    flat_number: str = ""
    flat_suffix: str = ""
    level_type: str = ""
    level_prefix: str = ""
    level_number: str = ""
    level_suffix: str = ""
    number_first_prefix: str = ""
    number_first: str = ""
    number_first_suffix: str = ""
    number_last_prefix: str = ""
    number_last: str = ""
    number_last_suffix: str = ""
    street_name: str = ""
    street_type: str = ""
    street_suffix: str = ""
    locality_name: str = ""
    state: str = ""
    postcode: str = ""
    address_type: str = ""

    @property
    def pid(self) -> str:
        return self.addr_id

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> "Address":
        return cls(**{f: row.get(c, "") for c, f in COLUMN_TO_FIELD.items()})

    # -- display ----------------------------------------------------------
    def street_label(self) -> str:
        stype = codes.STREET_TYPE_AUT.get(self.street_type, self.street_type)
        return " ".join(p for p in (self.street_name, stype, self.street_suffix) if p)

    def number_label(self) -> str:
        first = self.number_first_prefix + self.number_first + self.number_first_suffix
        last = self.number_last_prefix + self.number_last + self.number_last_suffix
        if first and last:
            return "%s-%s" % (first, last)
        if first:
            return first
        if self.lot_number:
            return "LOT %s%s%s" % (self.lot_number_prefix, self.lot_number, self.lot_number_suffix)
        return ""

    def flat_label(self) -> str:
        ident = self.flat_prefix + self.flat_number + self.flat_suffix
        if not (self.flat_type or ident):
            return ""
        name = codes.FLAT_TYPE_AUT.get(self.flat_type, self.flat_type)
        return ("%s %s" % (name, ident)).strip()

    def level_label(self) -> str:
        ident = self.level_prefix + self.level_number + self.level_suffix
        if not (self.level_type or ident):
            return ""
        name = codes.LEVEL_TYPE_AUT.get(self.level_type, self.level_type or "LEVEL")
        return ("%s %s" % (name, ident)).strip()

    def short_label(self) -> str:
        """'BUILDING, UNIT 101, LEVEL 1, 1-15 HARBOUR ST' (no locality)."""
        head = [p for p in (self.building_name, self.flat_label(), self.level_label()) if p]
        street = " ".join(p for p in (self.number_label(), self.street_label()) if p)
        return ", ".join(head + ([street] if street else []))

    def full_label(self) -> str:
        tail = " ".join(p for p in (self.locality_name, self.state, self.postcode) if p)
        return ", ".join(p for p in (self.short_label(), tail) if p)


ADDRESS_FIELDS = [f.name for f in fields(Address)]
