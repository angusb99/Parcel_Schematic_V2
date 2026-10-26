"""End-to-end tests through the notebook session on a seed not used for tuning."""
import unittest
from pathlib import Path

import pandas as pd

from parcel_address_associator import AssociatorSession, Config, INPUT_COLUMNS, make_synthetic
from parcel_address_associator.engine import prepare_input


class Validation(unittest.TestCase):
    def test_repairs_and_reports(self):
        df, _ = make_synthetic(seed=3)
        m = df.head(6).astype(object)
        m["LEVEL_NUMBER"] = pd.to_numeric(m["LEVEL_NUMBER"], errors="coerce")
        m.loc[0, ["FLAT_NUMBER_PREFIX", "FLAT_NUMBER"]] = ["", 101.0]
        m.loc[1, ["FLAT_NUMBER_PREFIX", "FLAT_NUMBER"]] = ["", "G01"]
        m.loc[2, "FLAT_TYPE_CODE"] = "APARTMENT"
        m.loc[3, "STREET_TYPE_CODE"] = "ST"
        m.loc[4, "ADDR_ID"] = m.loc[5, "ADDR_ID"]
        m = m.drop(columns=["SOURCE_ID"])
        clean, issues = prepare_input(m)
        checks = set(issues["CHECK"])
        for c in ("missing_column", "duplicate_addr_id", "code_normalised", "identifier_split"):
            self.assertIn(c, checks)
        self.assertEqual(len(clean), 5)
        self.assertEqual(clean.loc[0, "FLAT_NUMBER"], "101")
        self.assertEqual(tuple(clean.loc[1, ["FLAT_NUMBER_PREFIX", "FLAT_NUMBER"]]), ("G", "1"))
        self.assertEqual(clean.loc[2, "FLAT_TYPE_CODE"], "APT")
        self.assertEqual(clean.loc[3, "STREET_TYPE_CODE"], "STREET")
        self.assertFalse(clean["LEVEL_NUMBER"].str.contains(r"\.").any())

    def test_ids_keep_their_case(self):
        df, _ = make_synthetic(seed=3)
        df["ADDR_ID"] = df["ADDR_ID"].str.lower()
        clean, _ = prepare_input(df)
        self.assertTrue(clean["ADDR_ID"].str.islower().all())
        self.assertTrue(clean["CADASTRE_POLYGON_PID"].str.startswith("cpl").all())

    def test_missing_required_column(self):
        df, _ = make_synthetic(seed=3)
        with self.assertRaises(ValueError):
            prepare_input(df.drop(columns=["CADASTRE_POLYGON_PID"]))


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.truth = make_synthetic(seed=7)
        cls.s = AssociatorSession(cls.df, verbose=False)
        cls.s.validate()
        cls.s.run()
        cls.ev = cls.s.evaluate(cls.truth)

    def test_accuracy(self):
        self.assertEqual(self.ev["overall"]["FP"], 0)
        bs = self.ev["by_scenario"].set_index("SCENARIO")
        for sc in ("single_tower", "multi_tower", "corner_frontage", "level_convention", "twin_towers",
                   "mixed_use", "background"):
            self.assertGreaterEqual(bs.loc[sc, "RECALL"], 0.95, sc)

    def test_unconfirmable_cases_are_reviewed_not_grouped(self):
        bs = self.ev["by_scenario"].set_index("SCENARIO")
        for sc in ("strata_lot_numbering", "twin_towers_unresolved"):
            self.assertEqual(bs.loc[sc, "FP"], 0, sc)
            self.assertGreater(bs.loc[sc, "REVIEW_TRUE"], 0, sc)
        self.assertGreaterEqual(self.ev["review"]["RECALL_IF_REVIEWED"], 0.99)

    def test_outputs(self):
        frames = self.s.to_frames()
        for name in ("parcel_summary", "styles", "blocks", "mappings", "structures", "addresses",
                     "duplicate_pairs", "duplicate_groups", "review_candidates"):
            self.assertGreater(len(frames[name]), 0, name)
        g = frames["duplicate_groups"]
        self.assertTrue((g["SIZE"] >= 2).all())
        a = frames["addresses"]
        self.assertTrue(set(a["ADDR_ID"]) <= set(self.df["ADDR_ID"]))
        dup = a[a["DUPLICATE_OF"] != ""]
        self.assertTrue(dup["DUPLICATE_OF"].isin(set(g["PREFERRED_ADDR_ID"])).all())

    def test_save_html(self):
        import os
        import tempfile
        d = tempfile.mkdtemp(prefix="paa_html_")
        pid = self.s.interesting_parcels(1)[0]
        path = self.s.save_html(os.path.join(d, "nested", "one.html"), parcels=[pid])
        self.assertTrue(os.path.isfile(path))
        doc = Path(path).read_text(encoding="utf-8")
        self.assertTrue(doc.startswith("<!doctype html>"))
        self.assertEqual(doc.count("<section"), 1)
        self.assertIn("<svg", doc)
        self.assertNotIn("<script", doc)
        full = Path(self.s.save_html(os.path.join(d, "all.html"), max_parcels=5)).read_text(encoding="utf-8")
        self.assertEqual(full.count("<section"), 5)

    def test_views(self):
        pid = self.s.interesting_parcels(1)[0]
        self.assertIn("<svg", self.s.show_parcel(pid))
        self.assertIn("STYLES", self.s.schematic(pid))
        self.assertEqual(list(self.s.overview().columns), ["METRIC", "VALUE"])
        aid = self.s.addresses().query("GROUP_ID != ''").ADDR_ID.iloc[0]
        self.assertIn("GROUP", self.s.explain(aid))

    def test_rerun_subset(self):
        pid = self.s.interesting_parcels(1)[0]
        before = len(self.s.duplicate_groups())
        self.s.run(parcels=[pid], config=Config().with_(band_low=0.99))
        after = len(self.s.duplicate_groups())
        self.s.run(config=Config())
        self.assertLess(after, before)
        self.assertEqual(len(self.s.duplicate_groups()), before)


if __name__ == "__main__":
    unittest.main()
