"""Unit tests: building references, numbering-model detection, single-parcel cases."""
import unittest

import pandas as pd

from parcel_address_associator import Config, INPUT_COLUMNS
from parcel_address_associator.engine import analyse, prepare_input
from parcel_address_associator.features import building_ref, compare_brefs
from parcel_address_associator.schemes import detect_encoded


def frame(rows):
    """rows: dicts with a few columns; everything else blank."""
    df = pd.DataFrame(rows)
    for c in INPUT_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[INPUT_COLUMNS]


def unit(aid, flat, n1, n2="", ft="UNIT", pfx="", lt="", lnum="", bname="", street="HARBOUR",
         stype="STREET", parcel="P1"):
    return {"ADDR_ID": aid, "SOURCE_ID": "S" + aid, "FLAT_TYPE_CODE": ft, "FLAT_NUMBER_PREFIX": pfx,
            "FLAT_NUMBER": str(flat), "LEVEL_TYPE_CODE": lt, "LEVEL_NUMBER": str(lnum),
            "NUMBER_FIRST": str(n1), "NUMBER_LAST": str(n2), "BUILDING_NAME": bname,
            "STREET_NAME": street, "STREET_TYPE_CODE": stype, "LOCALITY_NAME": "PORT ALDER",
            "STATE_ABBREVIATION": "NSW", "POSTCODE": "2989", "CADASTRE_POLYGON_PID": parcel}


def groups_of(df):
    clean, _ = prepare_input(df)
    res = analyse(clean, Config())
    return {frozenset(f.pid for f in g.members) for r in res.values() for g in r.groups}, res


class FeatureTests(unittest.TestCase):
    def test_building_refs(self):
        self.assertEqual(building_ref("TOWER A"), ("A", "letter"))
        self.assertEqual(building_ref("BUILDING 2"), ("2", "number"))
        self.assertEqual(building_ref("EAST TOWER"), ("EAST", "name"))
        self.assertEqual(building_ref("THE ARCADIA"), ("ARCADIA", "name"))
        self.assertEqual(compare_brefs(("A", "letter"), ("A", "letter")), "equal")
        self.assertEqual(compare_brefs(("A", "letter"), ("B", "letter")), "conflict")
        self.assertEqual(compare_brefs(("B", "letter"), ("2", "number")), "ordinal")


class NumberingTests(unittest.TestCase):
    def test_level_encoded(self):
        nums = [L * 100 + u for L in range(1, 15) for u in range(1, 9)]
        self.assertTrue(detect_encoded(nums))
        self.assertTrue(detect_encoded(nums + [1, 2, 3, 4]))            # ground floor units

    def test_sequential(self):
        self.assertFalse(detect_encoded(list(range(1, 151))))           # runs through 100
        self.assertFalse(detect_encoded(list(range(1, 64))))
        self.assertFalse(detect_encoded([101, 150, 199, 260, 275]))     # unit part > 40


class ParcelCases(unittest.TestCase):
    def test_level_encoded_vs_explicit_level(self):
        rows = []
        for L in range(1, 6):
            for u in range(1, 5):
                rows.append(unit("E%d%d" % (L, u), L * 100 + u, 1, 15))
                rows.append(unit("X%d%d" % (L, u), u, 1, lt="L", lnum=L))
        groups, _ = groups_of(frame(rows))
        expected = {frozenset(("E%d%d" % (L, u), "X%d%d" % (L, u))) for L in range(1, 6) for u in range(1, 5)}
        self.assertEqual(groups, expected)

    def test_shops_never_match_units(self):
        rows = [unit("U%d" % k, k, 10) for k in range(1, 7)] + \
               [unit("S%d" % k, k, 10, ft="SHOP") for k in range(1, 7)]
        groups, _ = groups_of(frame(rows))
        self.assertEqual(groups, set())

    def test_two_towers_with_letters(self):
        rows = []
        for b in "AB":
            for L in range(1, 5):
                for u in range(1, 4):
                    rows.append(unit("P%s%d%d" % (b, L, u), L * 100 + u, 1, 3, pfx=b))
                    rows.append(unit("N%s%d%d" % (b, L, u), L * 100 + u, 1, 3, bname="TOWER " + b))
        groups, _ = groups_of(frame(rows))
        for g in groups:
            ids = sorted(g)
            self.assertEqual(ids[0][1:], ids[1][1:])                    # same tower, level, unit

    def test_identical_towers_go_to_review(self):
        rows = []
        for b, num, name in (("W", 50, "WEST TOWER"), ("E", 52, "EAST TOWER")):
            for L in range(1, 6):
                for u in range(1, 5):
                    rows.append(unit("N%s%d%d" % (b, L, u), L * 100 + u, num))
                    rows.append(unit("T%s%d%d" % (b, L, u), L * 100 + u, 50, 52, bname=name))
        groups, res = groups_of(frame(rows))
        self.assertEqual(groups, set())
        rev = res["P1"].review_pairs
        self.assertTrue(rev and all(p.relation == "AMBIGUOUS_BUILDING" for p in rev))


if __name__ == "__main__":
    unittest.main()
