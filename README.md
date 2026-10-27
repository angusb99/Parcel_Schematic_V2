# Parcel Address Associator

Takes addresses that are already linked to cadastre parcels and, parcel by parcel:

1. works out the **address styles** in use (the schematics), e.g.
   `UNIT {B}{L}{UU}, {N}-{N} {STREET}` and `TOWER {B}, UNIT {L}{UU}, {N}-{N} {STREET}`;
2. learns **how those styles map onto each other** (`UNIT A1103` ↔ `TOWER A, UNIT 1103` ↔
   `UNIT 3, LEVEL 11, 1 MERIDIAN AV`);
3. groups the **potential duplicate addresses** that mapping implies, suggests the
   record to keep, and lists cases it cannot confirm for review;
4. finds **shell addresses**: addresses of a whole building, a site or common property
   rather than a dwelling (optionally using parcel polygons, building footprints and
   geocodes).

It is built for complex apartment precincts. It is rule-based and statistical only (no
LLMs, no libpostal). Matching needs only pandas; the optional geometry uses shapely. It
does no I/O: DataFrames in, DataFrames out.

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

Optional, never used for matching:
* `ADDRESS_TYPE`, for the residential-use reconciliation and shell detection;
* `LATITUDE`, `LONGITUDE` (decimal degrees), for shell detection. Swapped columns are
  detected and swapped back; out-of-range values are dropped and reported.

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
It runs the whole process stage by stage on synthetic data, with outputs saved, and writes
its HTML schematics to `outputs/`. Replace the input cell with your DataFrame.

**From Python:**

```python
from parcel_address_associator import AssociatorSession

s = AssociatorSession(df)       # your DataFrame
s.validate()                    # issues table
s.profile()                     # what each parcel holds
s.run()                         # analyse every parcel
s.overview()                    # headline numbers
s.save_html("report.html")      # standalone HTML report; returns the path, opens nothing
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
| `save_html(path, parcels=None, max_parcels=50)` | writes one self-contained HTML page with headline numbers, an index of all parcels, and a schematic section (styles, mappings, stacking plans, groups, review candidates) for the chosen parcels or the 50 with most duplicate groups; returns the path and never opens anything |

`show_parcel(pid)` is the optional inline alternative: it renders in the notebook cell
instead of writing a file. The notebook writes its HTML to `outputs/`, which git ignores.

Bands: HIGH ≥ 0.85, MEDIUM ≥ 0.6, LOW ≥ 0.4. `s.run(parcels=[...], config=s.config.with_(...))`
re-analyses a subset with different settings; every weight is in `config.py`.

## Residential use reconciliation

Two sources say whether an `ADDR_ID` is a residential dwelling:
* **base:** the optional `ADDRESS_TYPE` column, where `RESIDENTIAL_DWELLING` and
  `SHELL_CHILD` count as residential (`Config.residential_address_types`);
* **other:** a second dataset with one row per `ADDR_ID` and a yes/no
  `RESIDENTIAL_DWELLING` flag (Y/N, true/false, 1/0 are all accepted).

```python
s.attach_use(use_df)                              # validates coverage and flag values
s.reconcile_use()
s.use_reconciliation(direction="BASE_ONLY")       # base residential, other not: why, and the fix
s.use_areas()                                     # blocks to rectify as areas
s.use_summary()                                   # per parcel: base vs other vs reconciled dwelling counts
s.source_reliability()                            # how far each source can be trusted, per context
```

Neither source is trusted outright. The decision is made per **premise**: a duplicate
group (one physical place with several `ADDR_ID`s) or a lone address. It combines a
structural prior (flat type, and whether the address heads a building or site or stands
alone), consensus among premises in the same block and on the same floor, and both
sources' flags. Each source's flag is weighted by its sensitivity and specificity, fitted
from the data per context (units, commercial, ancillary, building addresses, standalone).
A residential premise is counted once, on one representative `ADDR_ID`, preferring the
record the sources already count.

Every `ADDR_ID` gets a `CATEGORY` and suggested flags (`CHANGE_BASE`, `CHANGE_OTHER`,
`SUGGESTED_ACTION`):

| category | meaning |
|---|---|
| `DUPLICATE_OF_COUNTED_DWELLING` | same dwelling as another `ADDR_ID` that is (or should be) the one counted |
| `BUILDING_OR_SITE_ADDRESS` | address of a building or site holding units, or a shell found by `detect_shells()`; not itself a dwelling |
| `MISSED_BY_OTHER` / `MISSED_BY_BASE` | residential premise that source does not count |
| `LIKELY_NOT_RESIDENTIAL` | flagged residential, but the premise looks non-residential (e.g. shop, suite) |
| `POSSIBLE_DUPLICATE` | may be a review-candidate duplicate; no change suggested until confirmed |
| `UNRESOLVED`, `MISSED_BY_BOTH` | balanced evidence, or structure-only signal: review |

**Areas.** Sometimes one source says residential and the other doesn't for nearly a whole
block, while the same scheme or building elsewhere on the parcel is residential in both.
That's a `*_COVERAGE_GAP` in the silent source: its flags are set aside for that block and
its missed dwellings are restored. Without that support, the block is a
`BLOCK_DISAGREEMENT`, reviewed as an area. In the HTML report, each parcel gets use plans:
cells coloured by reconciled use, with a red outline where a use correction is suggested.

If `detect_shells()` has run first, its verdicts decide which parents are buildings
(never dwellings) and which are standalone houses. A possible shell gets a mild prior, so
the two sources decide. Without this, a front house with units behind it is treated as a
building address and loses its residential flag even when both sources say it is a
dwelling.

On synthetic data with deliberate errors in both sources (seeds 42, 7, 123 and a 4× set):

| | without shells | after `detect_shells()` |
|---|---|---|
| decided premises correct | 99.6–99.8% | 99.8–99.9% |
| left uncertain | ~6%, mostly unconfirmed duplicates routed to review | same |
| places counted correctly, second dataset | ~93% → ~99.0% | ~93% → ~99.2–99.5% |
| places counted correctly, base (it double-counts duplicates) | ~32–36% → ~88–89% | ~32–36% → ~89–90% |
| missing tower recovered as a coverage gap | every unit | every unit |
| building addresses and shops wrongly flagged residential | all corrected | all corrected |

The base's remaining error is double counting inside duplicates that are only review
candidates.

These figures come from synthetic error patterns I designed. Real data will mix them
differently, so validate on a hand-checked sample of your own parcels.

## Shell addresses

A **shell** address stands for something other than one dwelling or tenancy:

| class | examples |
|---|---|
| `BUILDING` | `10 CORAL ST` over its units; `BUILDING 2` / `BLOCK A` / `TOWER 1` flat types |
| `SITE` | a range address over two towers; a front-street address over a block and rear townhouses; an association lot whose townhouses each sit on their own parcel |
| `COMMON_PROPERTY` | `LOBBY`, `CARPARK 1`, `SUBSTATION 1`, `BBQ`, `ANTENNA`; `BASEMENT 1`, `ROOFTOP`, `PODIUM` with no flat; building names like `COMMON PROPERTY`, `CAR PARK`, `PLANT ROOM` |
| `FLOOR` | `LEVEL 3` with no flat: a whole floor, offered as `POSSIBLE_SHELL` for review |

Private car spaces, stores and garages are someone's lot, so they are neither dwellings
nor shells.

```python
s.run()
s.attach_use(use_df)                                    # optional: compares against its flag
s.attach_geometry(cadastre_df, buildings_df)            # optional: WKT; returns its issues table
s.detect_shells()
s.shells()                                              # SHELL / POSSIBLE_SHELL rows + disagreements
s.shells(category="SHELL_FLAGGED_RESIDENTIAL")          # shells counted as dwellings
s.shell_summary()                                       # per parcel: shells by class
s.shell_buildings(pid)                                  # per footprint: what is geocoded in it
s.reconcile_use()                                       # now uses the shell verdicts
```

**Geometry input** (all optional, lon/lat WKT as Geoscape supplies it):
* cadastre: `CADASTRE_POLYGON_PID`, `WKT`. Multi-row parcels are merged and invalid
  rings repaired.
* building footprints: `BUILDING_ID` (optional), `CADASTRE_POLYGON_PID`, `WKT`. Without
  the parcel column, each footprint is linked to the parcel it overlaps most.

Each parcel is projected to local metres around its centroid, so no CRS files are
needed. For projected WKT, pass `crs="EPSG:...."` so geocodes can be placed (needs pyproj).
A GeoDataFrame's geometry column works too.

**Geocodes at the parcel centroid.** Most G-NAF geocodes sit at the parcel centroid. Such
a geocode says nothing about which building an address is in. A geocode within 2 m of
the parcel's centroid, label point or pole of inaccessibility is therefore marked
`GEOCODE_LEVEL = PROPERTY` and ignored by the building-level rules. So is a point shared
by sub-addresses of two or more inferred buildings. Those addresses still get the
parcel-level rules, which need only the footprints. `GEOCODE_LEVEL` is `FOOTPRINT` or
`OPEN` for building-level geocodes, and `OFF_PARCEL` for one more than 25 m outside its
parcel.

**Evidence** (log-odds, every weight in `Config.shell_*`):

| evidence | effect |
|---|---|
| kind of address | parent −1.5; unit −5; shop −4; car space/store −3.5; common flat type +3; BUILDING/BLOCK/TOWER flat type +2.5; basement/rooftop level with no flat +2.5; plain level with no flat 0 |
| sub-address premises below a parent | +1.6 × log2(1 + n), capped at +4.8. Premises are counted once per duplicate group, through the parcel's inferred structures and the base address (a range covers its numbers) |
| premises on neighbouring parcels sharing the parent's base | +1.4 × log2(1 + n), capped (community and multi-lot schemes) |
| nothing below a parent | −2.5 (a house or a lot address) |
| range street number | +0.8 |
| common-property building name | +4 |
| *parcel level (footprints only, no geocode needed):* | |
| parcel with no building, premises next door | +1 (association or access lot) |
| the parcel's one dwelling-sized building (≥ 60 m²) holds all its premises | +1 |
| more dwelling-sized buildings than premises on the parcel (≤ 6 premises) | −2 (the parent may be a house of its own) |
| a few premises (≤ 3) spread over several buildings | −1 (a front house? goes to review) |
| *building level (only geocodes not at the parcel centroid; supersedes the parcel-level guess):* | |
| parent geocoded in a footprint holding its sub-addresses | +1 |
| parent alone in its own footprint, its sub-addresses in other footprints | −3 (a front house) |
| parent on open ground; a house-sized footprint holds no address while its sub-addresses are in other footprints | −1.5 (a front house geocoded at the frontage: goes to review) |
| non-dwelling sub-address in a footprint under 25 m² | +1.5 (kiosk, substation) |

`SITE` vs `BUILDING` comes from how many footprints, inferred structures or neighbouring
parcels a parent's sub-addresses occupy. Duplicate-group members share one verdict.

Verdicts: `SHELL` (P ≥ 0.8), `POSSIBLE_SHELL` (P ≥ 0.5, review), `NOT_SHELL`. Each row
also gets a `CATEGORY` comparing the verdict with `ADDRESS_TYPE`
(`Config.shell_address_types`, default `SHELL_PARENT`; set it to your shell values) and
the second dataset:

| category | meaning |
|---|---|
| `SHELL_FLAGGED_RESIDENTIAL` | shell counted as a residential dwelling by `ADDRESS_TYPE` or the second dataset |
| `SHELL_NOT_TYPED` | shell whose `ADDRESS_TYPE` is not a shell type |
| `TYPED_SHELL_NOT_DETECTED` | `ADDRESS_TYPE` says shell, but it looks like a dwelling or tenancy |
| `REVIEW` | possible shell |

In the HTML report, each parcel gets section 8. It draws the parcel (dashed), the
footprints and the geocodes, coloured by verdict. Property-level geocodes are hollow.

On synthetic data (seeds 42, 1, 7, 123, 2024 ×2, 99 ×4), with deliberate traps: front
houses with rear units, vacant lots, garden sheds, private car spaces and stores. 85% of
the geocodes are parcel centroids, as in G-NAF; `make_synthetic(centroid_share=...)`
changes that.

| | text and parcel linkage only | with footprints, 85% centroid geocodes | with footprints, all centroid | with footprints, no centroid |
|---|---|---|---|---|
| precision | 0.91–0.98 | 1.00 | 1.00 | 1.00 |
| recall | 0.88–0.98 | 1.00 | 1.00 | 1.00 |
| class correct | 0.95–0.96 | 0.98–1.00 | 0.98 | 1.00 |
| non-shells left for review (4× set) | 11 | 24 | 24 | 9 |

Without geometry, the misses are 2-unit strata blocks (left as `POSSIBLE_SHELL`). The
false positives are front houses whose base address also carries rear units. The
parcel's footprints settle both. With centroid geocodes, front houses go to review
instead of being cleared: only a building-level geocode in the house's own footprint
clears one outright. These scenarios were written alongside the rules, so treat the
figures as consistency checks and review a sample of real parcels.

## How it works (per parcel)

| step | what happens |
|---|---|
| style | addresses grouped by format: flat type, building prefix or name, explicit level, base shape (single number, range, lot) |
| block | one style on one context (base address + building reference): what that scheme says is in one building |
| numbering model | `encoded` (1103 = L11 U3), `explicit+encoded` (`UNIT 1103, LEVEL 11`: the door number wins when the level field disagrees), `explicit+per-level` (`UNIT 3, LEVEL 11`), `explicit+sequential`, `prefix-level` (G01), `plain`; two schemes sharing one format and base are split apart |
| mapping | for each pair of blocks, the transform explaining most records: same identifier; level + unit; position within level; position through the building. A level offset of ±1 is only allowed as a ground-floor convention: the side with a ground floor (G01, level 0) is the one counting from 0, and the other side must take its levels from a level field or prefix, never from a door number. So `UNIT 405` never maps to `UNIT 505` |
| evidence | log-odds from coverage, layout extent (a block with a narrower floor plate nested inside another is a smaller building, not a sample; a source missing some floors keeps the full plate and is not penalised), building references (`TOWER A` = prefix `A`), base relationship, same-style prior |
| exclusivity | a block maps to one building per style unless it spans several; near-ties become `AMBIGUOUS_BUILDING` review pairs |
| structures | linked blocks form inferred buildings, drawn as stacking plans |
| groups | record pairs grouped so no group holds two different units of one scheme, and never two different encoded door numbers (`UNIT 405` / `UNIT 505`) whatever the mapping says |
| parents | site and building addresses (no flat or level) paired when they refer to the same building |

## Results on synthetic data

`make_synthetic(seed, replicates)` builds a fictional town of precincts, each reproducing
a duplicate pattern. On seeds not used for tuning (1, 7, 123, 2024 ×2, 99 ×4; up to 9,405
rows in 1.3 s):

* **Resolvable scenarios:** precision 1.000 and recall 1.000. These are single tower,
  multi-tower, corner frontage, ground-floor convention, twin towers with letters,
  mixed use, common property, community title and background.
* **Staggered coverage** (sources covering different floor ranges of one tower, e.g.
  levels 1–8 vs 3–10, plus a level field that is sometimes off by one): precision 1.000,
  recall 0.99–1.00. No pair ever joins two different door numbers; the few misses are
  small partial sources left unlinked.
* **Two scenarios are not decidable from this schema, by design:**
  * *strata-lot numbering:* `UNIT 5` vs `UNIT 105`, where the order matches but nothing
    confirms it;
  * *identical twin towers:* street numbers vs EAST/WEST TOWER names.

  They produce no groups and no false positives; their true pairs are all in
  `review_candidates()`. Overall grouped recall is about 0.89, and about 0.998 if the
  review list is included.

These are consistency checks on scenarios written alongside the engine, not accuracy on
real data. Calibrate by reviewing a sample of real parcels.

## What this schema cannot see

An earlier version (not in this repository) also used geocodes, `LEGAL_PARCEL_ID`
(strata lots), G-NAF alias links and the cadastre strata count. Those confirmed
strata-lot renumbering and told identical towers apart. With components only, such cases
go to the review list instead of being grouped. Adding any of those columns later would
let them be resolved. Geocodes and footprints are now used, but only for shell
detection, not for matching.

Shell detection does not use the G-NAF geocode type. A geocode at a building centroid and
one at a property centroid are treated alike, apart from the property-level rule above.
Adding a `GEOCODE_TYPE_CODE` column would sharpen the footprint evidence.

## Tests

```bash
python -m unittest discover -s tests -v
```
