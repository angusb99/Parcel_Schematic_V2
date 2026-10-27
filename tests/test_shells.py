"""Shell detection: text rules, geometry rules, session wiring, synthetic accuracy."""
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from parcel_address_associator import AssociatorSession, make_synthetic
from parcel_address_associator.engine import prepare_input
from parcel_address_associator.geometry import metres_per_degree, prepare_geometry
from test_units import frame, unit

LON0, LAT0 = 151.0, -33.5
KX, KY = metres_per_degree(LAT0)


def lonlat(x, y):
    return LON0 + x / KX, LAT0 + y / KY


def box(x0, y0, x1, y1):
    ring = [lonlat(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))]
    return "POLYGON ((%s))" % ", ".join("%.9f %.9f" % p for p in ring)


def at(row, x, y):
    row["LONGITUDE"], row["LATITUDE"] = lonlat(x, y)
    return row


def parent(aid, n1, n2="", bname="", parcel="P1", street="HARBOUR"):
    r = unit(aid, "", n1, n2, ft="", bname=bname, parcel=parcel, street=street)
    r["FLAT_NUMBER"] = ""
    return r


def shells_of(rows, cadastre=None, buildings=None):
    df = frame(rows)
    for c in ("LATITUDE", "LONGITUDE"):
        df[c] = [r.get(c, float("nan")) for r in rows]
    s = AssociatorSession(df, verbose=False)
    s.validate()
    s.run()
    if cadastre is not None or buildings is not None:
        s.attach_geometry(cadastre, buildings)
    s.detect_shells()
    t = s.shell["addresses"].set_index("ADDR_ID")
    return t, s


class TextRules(unittest.TestCase):
    def test_house_is_not_a_shell(self):
        t, _ = shells_of([parent("H", 5), parent("H2", 7, parcel="P2")])
        self.assertEqual(t.loc["H", "VERDICT"], "NOT_SHELL")

    def test_building_over_units(self):
        rows = [parent("B", 10)] + [unit("U%d" % k, 100 + k, 10) for k in range(1, 7)]
        t, _ = shells_of(rows)
        self.assertEqual((t.loc["B", "VERDICT"], t.loc["B", "SHELL_CLASS"]), ("SHELL", "BUILDING"))
        self.assertTrue((t.loc[["U%d" % k for k in range(1, 7)], "VERDICT"] == "NOT_SHELL").all())

    def test_common_property_types(self):
        rows = [parent("B", 10)] + [unit("U%d" % k, 100 + k, 10) for k in range(1, 5)]
        rows += [unit("LOB", "", 10, ft="LBBY"), unit("CP1", 1, 10, ft="CARP"),
                 unit("BS1", "", 10, ft="", lt="B", lnum=1), unit("CS1", 1, 10, ft="CARS"),
                 parent("CPN", 10, bname="COMMON PROPERTY")]
        t, s = shells_of(rows)
        for aid in ("LOB", "CP1", "BS1", "CPN"):
            self.assertEqual((t.loc[aid, "VERDICT"], t.loc[aid, "SHELL_CLASS"]), ("SHELL", "COMMON_PROPERTY"), aid)
        self.assertEqual(t.loc["CS1", "VERDICT"], "NOT_SHELL")          # a private car space
        # the common-property name is not a duplicate of the building address
        self.assertFalse(any({"B", "CPN"} <= {f.pid for f in g.members} for g in s.results["P1"].groups))

    def test_whole_floor_is_for_review(self):
        rows = [parent("B", 10)] + [unit("U%d" % k, 200 + k, 10) for k in range(1, 5)]
        rows.append(unit("L1", "", 10, ft="", lt="L", lnum=1))
        t, _ = shells_of(rows)
        self.assertEqual((t.loc["L1", "VERDICT"], t.loc["L1", "SHELL_CLASS"]), ("POSSIBLE_SHELL", "FLOOR"))

    def test_shell_flagged_residential(self):
        rows = [parent("B", 10)] + [unit("U%d" % k, 100 + k, 10) for k in range(1, 7)]
        rows[0]["ADDRESS_TYPE"] = "RESIDENTIAL_DWELLING"
        for r in rows[1:]:
            r["ADDRESS_TYPE"] = "SHELL_CHILD"
        df = frame(rows)
        df["ADDRESS_TYPE"] = [r["ADDRESS_TYPE"] for r in rows]
        s = AssociatorSession(df, verbose=False)
        s.run()
        s.detect_shells()
        t = s.shell["addresses"].set_index("ADDR_ID")
        self.assertEqual(t.loc["B", "CATEGORY"], "SHELL_FLAGGED_RESIDENTIAL")


class GeometryRules(unittest.TestCase):
    def front_house(self, house_xy):
        """15 x 40 lot: house footprint at the front, three units in a rear block."""
        cad = pd.DataFrame({"CADASTRE_POLYGON_PID": ["P1"], "WKT": [box(0, 0, 15, 40)]})
        bld = pd.DataFrame({"BUILDING_ID": ["HOUSE", "REAR"], "CADASTRE_POLYGON_PID": ["P1", "P1"],
                            "WKT": [box(3, 5, 12, 17), box(2, 24, 13, 36)]})
        rows = [at(parent("H", 5), *house_xy)] + [at(unit("U%d" % k, k, 5), 7.5, 30) for k in range(1, 4)]
        return rows, cad, bld

    def test_front_house_is_not_a_shell(self):
        rows, cad, bld = self.front_house((7.5, 11))
        t, _ = shells_of(rows)
        self.assertEqual(t.loc["H", "VERDICT"], "SHELL")                   # text alone cannot tell
        t, _ = shells_of(rows, cad, bld)
        self.assertEqual(t.loc["H", "VERDICT"], "NOT_SHELL")
        self.assertIn("front house", t.loc["H", "EVIDENCE"])

    def test_front_house_geocoded_at_frontage_goes_to_review(self):
        rows, cad, bld = self.front_house((7.5, 1.5))
        t, _ = shells_of(rows, cad, bld)
        self.assertEqual(t.loc["H", "VERDICT"], "POSSIBLE_SHELL")

    def test_centroid_geocodes_carry_no_building_evidence(self):
        # every geocode at the parcel centroid (most G-NAF records): only the footprints count
        rows, cad, bld = self.front_house((7.5, 20))
        for r in rows:
            at(r, 7.5, 20)
        t, s = shells_of(rows, cad, bld)
        self.assertTrue((t["GEOCODE_LEVEL"] == "PROPERTY").all())
        self.assertEqual(t.loc["H", "VERDICT"], "POSSIBLE_SHELL")          # 3 units over 2 buildings
        self.assertIn("spread over 2 buildings", t.loc["H", "EVIDENCE"])
        self.assertNotIn("geocode", t.loc["H", "EVIDENCE"])
        self.assertEqual(s.shell_buildings()["N_BUILDING_LEVEL_GEOCODES"].sum(), 0)

    def test_strata_block_is_a_building_shell(self):
        cad = pd.DataFrame({"CADASTRE_POLYGON_PID": ["P1"], "WKT": [box(0, 0, 15, 40)]})
        bld = pd.DataFrame({"CADASTRE_POLYGON_PID": ["P1"], "WKT": [box(2, 6, 13, 34)]})
        rows = [at(parent("B", 5), 7.5, 20)] + [at(unit("U%d" % k, k, 5), 7.5, 20) for k in range(1, 3)]
        t, _ = shells_of(rows, cad, bld)
        self.assertEqual((t.loc["B", "VERDICT"], t.loc["B", "SHELL_CLASS"]), ("SHELL", "BUILDING"))

    def test_association_lot_with_units_next_door(self):
        cads = [("A", box(12, 0, 19, 70))] + [("L%d" % k, box(0, k * 14, 12, k * 14 + 14)) for k in range(5)]
        cad = pd.DataFrame(cads, columns=["CADASTRE_POLYGON_PID", "WKT"])
        bld = pd.DataFrame({"CADASTRE_POLYGON_PID": ["L%d" % k for k in range(5)],
                            "WKT": [box(2, k * 14 + 2, 10, k * 14 + 12) for k in range(5)]})
        rows = [at(parent("S", 1, parcel="A", street="WARATAH"), 15.5, 35)]
        rows += [at(unit("U%d" % k, k + 1, 1, parcel="L%d" % k, street="WARATAH"), 6, k * 14 + 7) for k in range(5)]
        t, _ = shells_of(rows, cad, bld)
        self.assertEqual((t.loc["S", "VERDICT"], t.loc["S", "SHELL_CLASS"]), ("SHELL", "SITE"))
        self.assertEqual(t.loc["S", "N_CROSS_PARCEL_PREMISES"], 5)
        self.assertIn("no building", t.loc["S", "EVIDENCE"])

    def test_geometry_issues(self):
        multi = pd.DataFrame({"CADASTRE_POLYGON_PID": ["P1", "P1", "P2", "P3"],
                              "WKT": [box(0, 0, 10, 10), box(10, 0, 20, 10), "NOT WKT",
                                      "POLYGON ((0 0, 10 10, 10 0, 0 10, 0 0))"]})
        g = prepare_geometry(multi, None)
        checks = set(g.issues["CHECK"])
        self.assertTrue({"parcel_rows_merged", "unparsed_wkt", "geometry_repaired"} <= checks)
        self.assertAlmostEqual(g.parcels["P1"].area, 200.0, delta=0.5)
        proj = pd.DataFrame({"CADASTRE_POLYGON_PID": ["P1"],
                             "WKT": ["POLYGON ((330000 6250000, 330010 6250000, 330010 6250010, 330000 6250000))"]})
        self.assertIn("projected_without_crs", set(prepare_geometry(proj, None).issues["CHECK"]))

    def test_geocode_columns(self):
        rows = [parent("H", 5), parent("J", 7)]
        df = frame(rows)
        df["LATITUDE"], df["LONGITUDE"] = [151.2, 151.3], [-33.9, -33.8]       # swapped
        clean, issues = prepare_input(df)
        self.assertIn("coordinates_swapped", set(issues["CHECK"]))
        self.assertTrue((clean["LATITUDE"] < 0).all())


class Synthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.truth, cls.use, cls.geo = make_synthetic(seed=5, with_use=True, with_geometry=True)

    def session(self, with_geometry):
        s = AssociatorSession(self.df, verbose=False)
        s.validate()
        s.run()
        s.attach_use(self.use)
        if with_geometry:
            s.attach_geometry(self.geo["cadastre"], self.geo["buildings"])
        return s.detect_shells()

    def test_accuracy_with_geometry(self):
        o = self.session(True).evaluate_shells(self.truth)["overall"]
        self.assertGreaterEqual(o["PRECISION"], 0.97)
        self.assertGreaterEqual(o["RECALL"], 0.95)
        self.assertGreaterEqual(o["CLASS_ACCURACY"], 0.95)

    def test_accuracy_text_only(self):
        o = self.session(False).evaluate_shells(self.truth)["overall"]
        self.assertGreaterEqual(o["PRECISION"], 0.85)
        self.assertGreaterEqual(o["RECALL"], 0.85)

    def test_session_outputs(self):
        s = self.session(True)
        self.assertTrue(len(s.shells()) and (s.shells()["VERDICT"] != "NOT_SHELL").any())
        self.assertEqual(set(s.shells(verdict="SHELL")["VERDICT"]), {"SHELL"})
        self.assertIn("SHELL", s.explain(s.shells(verdict="SHELL").ADDR_ID.iloc[0]))
        self.assertTrue({"shells", "shell_buildings", "shells_by_parcel"} <= set(s.to_frames()))
        d = tempfile.mkdtemp(prefix="paa_shell_")
        pid = s.shell_summary().CADASTRE_POLYGON_PID.iloc[0]
        doc = Path(s.save_html(os.path.join(d, "r.html"), parcels=[pid])).read_text(encoding="utf-8")
        self.assertIn("8. Shell addresses", doc)
        self.assertIn('aria-label="parcel plan', doc)
        # use reconciliation follows the shell verdicts
        s.reconcile_use()
        u = s.use_reconciliation().set_index("ADDR_ID")
        sh = s.shells(verdict="SHELL").ADDR_ID
        self.assertTrue((u.loc[sh, "CONTEXT"] == "building").all())

    def test_detect_needs_run(self):
        s = AssociatorSession(self.df, verbose=False)
        with self.assertRaises(RuntimeError):
            s.detect_shells()


if __name__ == "__main__":
    unittest.main()
