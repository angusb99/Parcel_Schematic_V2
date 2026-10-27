"""Tunable thresholds and evidence weights (log-odds)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Dict


@dataclass
class Config:
    # ---- scope -------------------------------------------------------------
    min_addresses: int = 2                 # parcels with fewer addresses are skipped

    # ---- block alignment ---------------------------------------------------
    # Small penalties so the simplest transform wins ties.
    form_penalty: Dict[str, float] = field(default_factory=lambda: {
        "raw": 0.0, "lu": 0.005, "lr": 0.03, "seq": 0.08, "seqnum": 0.08})
    offset_penalty: float = 0.01
    seq_size_tolerance: float = 0.05       # 'seq' needs blocks of near-equal size
    rank_min_keys: int = 5                 # rank-based mappings need at least this many units
    w_containment: float = 0.6             # quality = w*containment + (1-w)*extent overlap

    # ---- block link evidence (log-odds) ------------------------------------
    w_quality: float = 8.0                 # * (quality - quality_pivot)
    quality_pivot: float = 0.6
    min_support: int = 3                   # fewer matched keys than this is penalised
    w_low_support: float = -0.5            # per missing key below min_support
    prior_same_style: float = -3.0         # one scheme on two bases is usually two buildings
    w_bref_equal: float = 2.0
    w_bref_ordinal: float = 0.7            # 'A' vs 'BUILDING 1'
    w_bref_conflict: float = -4.0
    w_same_base: float = 1.0
    w_range_contains: float = 0.8
    w_same_street_other_number: float = -1.0
    w_rank_unconfirmed: float = -3.0       # rank mappings (strata-lot order) can't be confirmed
    # A smaller block whose keys all sit inside a sub-extent of a larger block is
    # unlikely to be a sample of it: log-likelihood w * n * ln(share), capped.
    w_extent_llr: float = 0.25
    extent_llr_cap: float = 2.5
    extent_min_keys: int = 5
    link_threshold: float = 0.5
    ambiguity_margin: float = 1.5          # log-odds gap below which two candidate links tie
    ambiguity_factor: float = 0.7          # p multiplier when a block could map to two buildings

    # ---- record pairs ------------------------------------------------------
    exact_duplicate_logit: float = 2.5

    # ---- parents (addresses without a flat/level) ---------------------------
    parent_same_building_logit: float = 1.7
    parent_same_base_bonus: float = 0.6
    parent_standalone_same_street_logit: float = 1.2
    parent_standalone_other_logit: float = -1.0

    # ---- review candidates (reported, never grouped) -----------------------
    review_min_p: float = 0.6              # rank mappings this good before the rank penalty
    review_size_ratio: float = 0.75        # ...between blocks of similar size

    # ---- residential-use reconciliation -------------------------------------
    residential_address_types: tuple = ("RESIDENTIAL_DWELLING", "SHELL_CHILD")
    # structural prior (log-odds that a premise is a dwelling) by context
    use_prior: Dict[str, float] = field(default_factory=lambda: {
        "unit": 1.5,          # UNIT/APT/FLAT/TOWNHOUSE/VILLA... sub-address
        "commercial": -2.0,   # SHOP/SUITE/OFFICE/TENANCY...
        "ancillary": -2.5,    # CARSPACE/STORE/GARAGE...
        "other_sub": -0.5,    # level-only or unrecognised flat type
        "building": -6.0,     # parent address heading a building or site: not a dwelling
                              # unless both sources insist (then UNCERTAIN, for review)
        "standalone": 0.5,    # parent address with no sub-addresses (e.g. a house)
        "possible_shell": -1.0,  # parent detect_shells() could not settle (e.g. a front house)
    })
    w_use_peers: float = 1.0               # premises in the same block (same scheme, building)
    use_peers_cap: float = 3.0
    w_use_floor: float = 0.5               # premises on the same floor of the same structure
    use_floor_cap: float = 2.0
    use_source_cap: float = 3.0            # no single source's flag outweighs structure + peers alone
    use_reliability_prior: float = 0.9     # starting sensitivity/specificity of each source
    use_reliability_strength: float = 10.0 # pseudo-count weight of that prior
    use_reliability_bounds: tuple = (0.6, 0.995)
    # area patterns: a block where one source says residential and the other does not
    # for (nearly) every premise
    use_gap_share: float = 0.8             # share of the block's premises disagreeing one way
    use_gap_min_premises: int = 3
    use_gap_support_share: float = 0.8     # same style/structure elsewhere agreed residential
    use_iterations: int = 15
    use_res_threshold: float = 0.8         # P(dwelling) at or above -> RESIDENTIAL
    use_non_threshold: float = 0.2         # at or below -> NOT_RESIDENTIAL; between -> UNCERTAIN

    # ---- shell detection ----------------------------------------------------
    shell_address_types: tuple = ("SHELL_PARENT",)   # ADDRESS_TYPE values that mark a shell
    # prior log-odds that an address is a shell, by what kind of address it is
    shell_prior: Dict[str, float] = field(default_factory=lambda: {
        "parent": -1.5,        # no flat or level: a house, or a building/site over sub-addresses
        "unit": -5.0,          # UNIT/APT/FLAT/TOWNHOUSE/VILLA...
        "commercial": -4.0,    # SHOP/SUITE/OFFICE...
        "private_lot": -3.5,   # CARSPACE/STORE/GARAGE: someone's lot, not common property
        "common_type": 3.0,    # LOBBY/CARPARK/SUBSTATION/BBQ/ANTENNA/SIGN...
        "building_type": 2.5,  # BUILDING/BLOCK/TOWER n, SITE n
        "common_level": 2.5,   # BASEMENT/PARKING/PODIUM/ROOFTOP with no flat
        "floor": 0.0,          # LEVEL n with no flat: a whole floor, for review
        "other_sub": -1.5,
    })
    w_shell_children: float = 1.6          # * log2(1 + sub-address premises below the parent), capped
    shell_children_cap: float = 3.0
    w_shell_cross_parcel: float = 1.4      # the same, for premises on neighbouring parcels
    shell_cross_parcel_m: float = 150.0
    w_shell_no_children: float = -2.5      # parent with nothing below it: a house or a lot address
    w_shell_range: float = 0.8             # range street number (20-26) on a parent
    w_shell_keyword: float = 4.0           # COMMON PROPERTY, CAR PARK, SUBSTATION... in the building name
    # geometry (only when parcels/footprints are attached). Geocodes at the parcel centroid
    # (most G-NAF records) say nothing about which building an address is in: only
    # building-level geocodes feed the point rules below.
    shell_property_point_m: float = 2.0    # geocode this close to the parcel centroid/label point
    w_shell_geo_with_children: float = 1.0  # parent geocoded in a footprint that holds its sub-addresses
    w_shell_geo_own_building: float = -3.0  # parent alone in its own footprint, sub-addresses elsewhere
    w_shell_geo_unaddressed: float = -1.5   # parent off-footprint; a house-sized footprint holds no
    shell_house_min_m2: float = 60.0        # address while its sub-addresses are in other footprints
    w_shell_geo_bare_parcel: float = 1.0    # parent on a parcel with no building, premises next door
    # parcel-level, used when the parent has no building-level geocode (one building holding
    # all the parcel's premises reuses w_shell_geo_with_children):
    w_shell_geo_spare_building: float = -2.0  # more dwelling-sized buildings than the parcel's premises
    shell_spare_building_max: int = 6         # ...on parcels with at most this many premises
    w_shell_geo_small_scheme: float = -1.0    # a few premises spread over several buildings
    shell_small_scheme: int = 3               # 'a few'
    w_shell_geo_tiny_footprint: float = 1.5  # non-dwelling sub-address in a kiosk-sized footprint
    shell_tiny_footprint_m2: float = 25.0
    shell_snap_m: float = 1.0              # geocode this close to a footprint counts as inside it
    shell_off_parcel_m: float = 25.0       # geocode further than this outside its parcel is ignored
    shell_site_share: float = 0.2          # sub-addresses outside the main footprint -> SITE
    shell_threshold: float = 0.8           # P(shell) at or above -> SHELL
    shell_possible: float = 0.5            # at or above -> POSSIBLE_SHELL (review)

    # ---- output bands -------------------------------------------------------
    band_high: float = 0.85
    band_medium: float = 0.6
    band_low: float = 0.4

    def band(self, p: float) -> str:
        if p >= self.band_high:
            return "HIGH"
        if p >= self.band_medium:
            return "MEDIUM"
        if p >= self.band_low:
            return "LOW"
        return "REJECT"

    def with_(self, **changes) -> "Config":
        """Copy with some values changed: cfg.with_(link_threshold=0.6)."""
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)
