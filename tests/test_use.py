"""Residential-use reconciliation tests."""
import unittest

import pandas as pd

from parcel_address_associator import AssociatorSession, make_synthetic
from parcel_address_associator.usage import parse_flag, prepare_use

from test_units import frame, unit


class Flags(unittest.TestCase):
    def test_parse_flag(self):
        for v in ("Y", "yes", "TRUE", 1, 1.0, True, "residential_dwelling"):
            self.assertIs(parse_flag(v), True, v)
        for v in ("N", "no", "false", 0, 0.0, False):
            self.assertIs(parse_flag(v), False, v)
        for v in (None, float("nan"), "maybe", ""):
            self.assertIsNone(parse_flag(v), v)

    def test_prepare_use_reports_gaps(self):
        use = pd.DataFrame({"ADDR_ID": ["A1", "A2", "A2", "A9"], "RESIDENTIAL_DWELLING": ["Y", "?", "N", "N"]})
        flags, issues = prepare_use(use, known_ids={"A1", "A2", "A3"})
        self.assertEqual(flags["A1"], True)
        self.assertIsNone(flags["A2"])                       # unparsed, first occurrence kept
        checks = set(issues.CHECK)
        for c in ("unparsed_flag", "duplicate_addr_id", "not_in_use_dataset", "not_in_address_data"):
            self.assertIn(c, checks)


class Categories(unittest.TestCase):
    def test_building_address_and_duplicates(self):
        rows = [dict(unit("P0", "", 1, 15, ft=""), ADDRESS_TYPE="RESIDENTIAL_DWELLING")]   # building address
        use = [("P0", "N")]
        for L in range(1, 5):
            for u in range(1, 5):
                a, b = "E%d%d" % (L, u), "X%d%d" % (L, u)
                rows.append(dict(unit(a, L * 100 + u, 1, 15), ADDRESS_TYPE="SHELL_CHILD"))
                rows.append(dict(unit(b, u, 1, 15, lt="L", lnum=L), ADDRESS_TYPE="SHELL_CHILD"))
                use += [(a, "Y"), (b, "N")]                  # other counts each dwelling once
        df = frame(rows)
        df["ADDRESS_TYPE"] = [r.get("ADDRESS_TYPE", "") for r in rows]
        s = AssociatorSession(df, verbose=False)
        s.validate()
        s.run()
        s.attach_use(pd.DataFrame(use, columns=["ADDR_ID", "RESIDENTIAL_DWELLING"]))
        s.reconcile_use()
        t = s.use_reconciliation().set_index("ADDR_ID")
        self.assertEqual(t.loc["P0", "CATEGORY"], "BUILDING_OR_SITE_ADDRESS")
        self.assertTrue(t.loc["P0", "CHANGE_BASE"])
        self.assertEqual(t.loc["X11", "CATEGORY"], "DUPLICATE_OF_COUNTED_DWELLING")
        self.assertEqual(t.loc["X11", "COUNTED_DWELLING_ADDR_ID"], "E11")
        self.assertEqual(t.loc["E11", "CATEGORY"], "OK")
        self.assertEqual(int((t.COUNTED_DWELLING_ADDR_ID == t.index).sum()), 16)


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df, cls.truth, use = make_synthetic(seed=7, with_use=True)
        cls.s = AssociatorSession(df, verbose=False)
        cls.s.validate()
        cls.s.run()
        cls.s.attach_use(use)
        cls.s.reconcile_use()
        cls.ev = cls.s.evaluate_use(cls.truth)

    def test_decisions_are_right(self):
        self.assertGreaterEqual(self.ev["USE_ACCURACY_DECIDED"], 0.99)
        self.assertLess(self.ev["UNCERTAIN_SHARE"], 0.15)

    def test_sources_improve(self):
        bs = self.ev["by_source"].set_index("SOURCE")
        for src in ("BASE", "OTHER"):
            self.assertGreater(bs.loc[src, "ENTITIES_RIGHT_AFTER"], bs.loc[src, "ENTITIES_RIGHT_BEFORE"], src)
        self.assertEqual(bs.loc["OTHER", "RIGHT_AFTER_BLOCK_MISSED"], "48/48")
        n, d = bs.loc["BASE", "RIGHT_AFTER_BUILDING_FLAGGED"].split("/")
        self.assertEqual(n, d)

    def test_coverage_gap_found(self):
        areas = self.s.use_areas()
        self.assertTrue((areas.VERDICT == "OTHER_COVERAGE_GAP").any())

    def test_outputs(self):
        frames = self.s.to_frames()
        for k in ("use_reconciliation", "use_areas", "use_by_parcel", "source_reliability"):
            self.assertGreater(len(frames[k]), 0, k)
        focus = self.s.use_reconciliation(direction="BASE_ONLY")
        self.assertTrue((focus.BASE_RESIDENTIAL == True).all())   # noqa: E712
        self.assertIn("7. Residential use", self.s.show_parcel(self.s.use_summary().CADASTRE_POLYGON_PID.iloc[0]))

    def test_rerun_clears_use(self):
        s = AssociatorSession(make_synthetic(seed=7)[0], verbose=False)
        s.validate()
        s.run()
        s.attach_use(pd.DataFrame({"ADDR_ID": s.clean.ADDR_ID, "RESIDENTIAL_DWELLING": "N"}))
        s.reconcile_use()
        s.run()
        self.assertEqual(s.use, {})


if __name__ == "__main__":
    unittest.main()
