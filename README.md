# Parcel Address Associator

Takes addresses that are already linked to cadastre parcels and, parcel by parcel:

1. works out the **address styles** in use (the schematics), e.g.
   `UNIT {B}{L}{UU}, {N}-{N} {STREET}` and `TOWER {B}, UNIT {L}{UU}, {N}-{N} {STREET}`;
2. learns **how those styles map onto each other** (`UNIT A1103` ↔ `TOWER A, UNIT 1103` ↔
   `UNIT 3, LEVEL 11, 1 MERIDIAN AV`);
3. groups the **potential duplicate addresses** that mapping implies, suggests the
   record to keep, and lists cases it cannot confirm for review.

It is built for complex apartment precincts. It is rule-based and statistical only (no
LLMs, no libpostal), needs nothing but pandas, and does no I/O: a DataFrame in, DataFrames out.

## Input

One row per address. Only `ADDR_ID` and `CADASTRE_POLYGON_PID` are required; other
missing columns are treated as blank.

```
ADDR_ID, SOURCE_ID, BUILDING_NAME,
LOT_NUMBER_PREFIX, LOT_NUMBER, LOT_NUMBER_SUFFIX,
FLAT_TYPE_CODE, FLAT_NUMBER_PREFIX, FLAT_NUMBER, FLAT_NUMBER_SUFFIX,
LEVEL_TYPE_CODE, LEVEL_NUMBER_PREFIX, LEVEL_NUMBER, LEVEL_NUMBER_SUFFIX,
NUMBER_FIRST_PREFIX, NUMBER_FIRST, NUMBER_FIRST_SUFFIX,
NUMBER_LAST_PREFIX, NUMBER_LAST, NUMBER_LAST_SUFFIX,
STREET_NAME, STREET_TYPE_CODE, STREET_SUFFIX_CODE,
LOCALITY_NAME, STATE_ABBREVIATION, POSTCODE, CADASTRE_POLYGON_PID
```

`ADDR_ID`, `SOURCE_ID` and `CADASTRE_POLYGON_PID` are passed through unchanged, case
included. `SOURCE_ID` is carried into every output for traceability and is not used for
matching.

Validation repairs the following and reports each repair:

| problem | repair |
|---|---|
| numbers as floats | `101.0` → `101` |
| packed identifiers | `G01` in `FLAT_NUMBER` → prefix `G` + number `1` |
| names instead of codes | `APARTMENT` → `APT`, `ST` → `STREET` |
| blank or duplicate `ADDR_ID` | row dropped or first occurrence kept |
| blank parcel | row excluded from analysis |
| unknown codes | flagged only |

## Use it

**Notebook:** open [`notebooks/parcel_address_associator.ipynb`](notebooks/parcel_address_associator.ipynb).
It runs the whole process stage by stage on synthetic data, with outputs saved. Replace the
input cell with your DataFrame.

**From Python:**

```python
from parcel_address_associator import AssociatorSession

s = AssociatorSession(df)       # your DataFrame
s.validate()                    # issues table
s.profile()                     # what each parcel holds
s.run()                         # analyse every parcel
s.overview()                    # headline numbers
s.show_parcel(pid)              # styles, mappings, stacking plans, groups (HTML in a notebook)
s.explain(addr_id)              # trace one address through every stage

frames = s.to_frames()          # every output as a DataFrame
frames["duplicate_groups"]      # write wherever you like
```

| method | returns |
|---|---|
| `duplicate_groups()` | one row per group: members, suggested `PREFERRED_ADDR_ID`, relation, confidence band |
| `duplicate_pairs()` | the pairs behind the groups, with method and evidence |
| `review_candidates()` | plausible pairs that components cannot confirm (never grouped) |
| `addresses()` | every analysed input row with style, block, structure, level/unit slot, group, `DUPLICATE_OF` |
| `styles()`, `blocks()` | the schematics: templates, contexts, numbering models |
| `mappings(include_rejected=True)` | every block-to-block mapping, status and log-odds evidence |
| `structures()`, `parcel_summary()`, `log` | inferred buildings, per-parcel counts, stage timings |

Bands: HIGH ≥ 0.85, MEDIUM ≥ 0.6, LOW ≥ 0.4. `s.run(parcels=[...], config=s.config.with_(...))`
re-analyses a subset with different settings; every weight is in `config.py`.

## How it works (per parcel)

| step | what happens |
|---|---|
| style | addresses grouped by format: flat type, building prefix or name, explicit level, base shape (single number, range, lot) |
| block | one style on one context (base address + building reference): what that scheme says is in one building |
| numbering model | `encoded` (1103 = L11 U3), `explicit+per-level` (`UNIT 3, LEVEL 11`), `explicit+sequential`, `prefix-level` (G01), `plain`; two schemes sharing one format and base are split apart |
| mapping | for each pair of blocks, the transform explaining most records: same identifier; level + unit (± level offset, e.g. ground counted as level 1); position within level; position through the building |
| evidence | log-odds from coverage, layout extent (a 12-level block is not a sample of an 18-level tower), building references (`TOWER A` = prefix `A`), base relationship, same-style prior |
| exclusivity | a block maps to one building per style unless it spans several; near-ties become `AMBIGUOUS_BUILDING` review pairs |
| structures | linked blocks form inferred buildings, drawn as stacking plans |
| groups | record pairs grouped so no group holds two different units of one scheme |
| parents | site and building addresses (no flat or level) paired when they refer to the same building |

## Results on synthetic data

`make_synthetic(seed, replicates)` builds a fictional town of precincts, each reproducing
a duplicate pattern. On seeds not used for tuning (1, 7, 123, 2024 ×2, 99 ×4; up to 8,365
rows in 1.2 s):

* **Resolvable scenarios:** precision 1.000 and recall 1.000. These are single tower,
  multi-tower, corner frontage, ground-floor convention, twin towers with letters,
  mixed use and background. The exception is one seed where a 3-unit rear-lane source
  was too small to link.
* **Two scenarios are not decidable from this schema, by design:**
  * *strata-lot numbering:* `UNIT 5` vs `UNIT 105`, where the order matches but nothing
    confirms it;
  * *identical twin towers:* street numbers vs EAST/WEST TOWER names.

  They produce no groups and no false positives; their true pairs are all in
  `review_candidates()`. Overall grouped recall is about 0.87, and about 0.998 if the
  review list is included.

These are consistency checks on scenarios written alongside the engine, not accuracy on
real data. Calibrate by reviewing a sample of real parcels.

## What this schema cannot see

An earlier version (not in this repository) also used geocodes, `LEGAL_PARCEL_ID`
(strata lots), G-NAF alias links and the cadastre strata count. Those confirmed
strata-lot renumbering and told identical towers apart. With components only, such cases
go to the review list instead of being grouped. Adding any of those columns later would
let them be resolved.

## Tests

```bash
python -m unittest discover -s tests -v
```
