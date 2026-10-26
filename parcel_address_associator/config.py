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
