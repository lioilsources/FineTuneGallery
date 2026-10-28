"""python -m unittest discover -s taxo/tests  (needs only pyyaml)."""

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import conformal  # noqa: E402
import consensus  # noqa: E402
import gen_schema  # noqa: E402
import taxolib  # noqa: E402
import temperature  # noqa: E402
import wd14map  # noqa: E402

TAX = taxolib.load()


class TestTaxonomy(unittest.TestCase):
    def test_generated_files_are_current(self):
        self.assertEqual(gen_schema.main(["--check"]), 0, "run python taxo/gen_schema.py")

    def test_minor_values_never_filterable(self):
        age = TAX.node("person.age_group")
        for v in TAX.minor_block["values"]:
            self.assertFalse(age.value(v).filterable)

    def test_vlm_schema_enums_match_yaml(self):
        s = gen_schema.vlm_schema(TAX)
        hc = s["properties"]["person"]["anyOf"][0]["properties"]["hair"]["properties"]["color"]
        self.assertEqual(hc["enum"], TAX.node("person.hair.color").value_ids + ["unknown"])
        self.assertIn({"type": "null"}, s["properties"]["person"]["anyOf"])


class TestWD14Map(unittest.TestCase):
    def test_hair_and_people(self):
        lab = wd14map.map_tags(TAX, {"1girl": 0.98, "blonde hair": 0.9, "long_hair": 0.8, "glasses": 0.6, "smile": 0.5})
        self.assertEqual(lab["subject"]["value"], ["people"])
        self.assertEqual(lab["person.hair.color"]["value"], "blonde")
        self.assertEqual(lab["person.hair.length"]["value"], "long")
        self.assertEqual(lab["person.clothing.type"]["value"], ["glasses"])
        self.assertEqual(lab["person.gender_presentation"]["value"], ["feminine"])

    def test_person_nodes_need_people(self):
        lab = wd14map.map_tags(TAX, {"no humans": 0.9, "scenery": 0.8, "blonde hair": 0.5})
        self.assertEqual(lab["subject"]["value"], ["environment"])
        self.assertNotIn("person.hair.color", lab)

    def test_minor_signal(self):
        self.assertGreater(wd14map.minor_signal(TAX, {"aged down": 0.5}), 0.4)
        self.assertEqual(wd14map.minor_signal(TAX, {"1girl": 0.9}), 0.0)


class TestTemperature(unittest.TestCase):
    def test_wd14_weighted(self):
        t = temperature.wd14_temperature(TAX, {"general": 0.0, "sensitive": 0.0, "questionable": 0.0, "explicit": 0.9})
        self.assertAlmostEqual(t, 1.0)
        t = temperature.wd14_temperature(TAX, {"general": 0.5, "sensitive": 0.5, "questionable": 0, "explicit": 0})
        self.assertAlmostEqual(t, 0.1665, places=3)

    def test_disagreement_is_unsure(self):
        e = temperature.ensemble(TAX, {"wd14": 0.05, "falconsai": 0.9, "adamcodd": 0.1})
        self.assertTrue(e["unsure"])
        e = temperature.ensemble(TAX, {"wd14": 0.05, "falconsai": 0.02, "adamcodd": 0.01})
        self.assertFalse(e["unsure"])
        self.assertEqual(e["band"], "safe")


class TestConsensus(unittest.TestCase):
    def test_single(self):
        n = TAX.node("person.hair.color")
        c = consensus.node_consensus(n, {"wd14": {"value": "blonde"}, "siglip": {"value": "blonde"}})
        self.assertEqual((c["value"], c["unsure"]), ("blonde", False))
        c = consensus.node_consensus(n, {"wd14": {"value": "blonde"}, "siglip": {"value": "brown"}})
        self.assertTrue(c["unsure"])
        c = consensus.node_consensus(n, {"wd14": {"value": "blonde"}})
        self.assertTrue(c["unsure"])

    def test_ordinal_adjacent_agrees(self):
        n = TAX.node("person.hair.length")
        c = consensus.node_consensus(n, {"wd14": {"value": "long"}, "siglip": {"value": "very_long"}})
        self.assertFalse(c["unsure"])
        c = consensus.node_consensus(n, {"wd14": {"value": "short"}, "siglip": {"value": "very_long"}})
        self.assertTrue(c["unsure"])

    def test_multi(self):
        n = TAX.node("subject")
        c = consensus.node_consensus(n, {"wd14": {"value": ["people", "environment"]}, "siglip": {"value": ["people"]}})
        self.assertEqual(c["value"], ["people"])
        self.assertFalse(c["unsure"])

    def test_minor_vectors(self):
        fx = json.loads((HERE.parent / "testdata" / "minor_vectors.json").read_text())
        for case in fx["cases"]:
            with self.subTest(case["name"]):
                got = consensus.minor_decision(TAX, case["labels"], case["wd14_signal"], case["ensemble"],
                                               case["wd14_temp"], fx["thresholds"])
                self.assertEqual(got, case["want"])


class TestConformal(unittest.TestCase):
    def test_synthetic_guarantees_hold_out_of_sample(self):
        for seed in range(1, 6):
            cal = conformal.synth(3000, seed)
            res = conformal.fit(cal, TAX, "synthetic")
            ev = conformal.evaluate(res["thresholds"], conformal.synth(20000, seed + 100))
            self.assertTrue(res["valid"])
            self.assertGreaterEqual(ev["explicit_recall"], 0.985, seed)
            self.assertGreaterEqual(ev["safe_precision"], 0.975, seed)
            th = res["thresholds"]
            self.assertLess(th["safe_max"], th["suggestive_max"])
            self.assertLess(th["suggestive_max"], th["nsfw_max"])

    def test_too_few_explicit_is_invalid(self):
        cal = [(0.9, "explicit")] * 50 + [(0.05, "safe")] * 500
        res = conformal.fit(cal, TAX, "synthetic")
        self.assertFalse(res["explicit_guard_valid"])
        self.assertIsNone(res["thresholds"]["explicit_guard"])
        self.assertFalse(res["valid"])

    def test_wilson(self):
        self.assertLess(conformal.wilson_lower(98, 100), 0.98)
        self.assertGreater(conformal.wilson_lower(9990, 10000), 0.98)


if __name__ == "__main__":
    unittest.main()
