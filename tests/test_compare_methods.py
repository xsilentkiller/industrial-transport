"""Side-by-side comparison script (experiments/compare_methods.py)."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from experiments import compare_methods as cm
from experiments.make_instances import build_instance


def tiny_instances(root: Path) -> list[Path]:
    paths = []
    for seed in (11, 12):
        out = root / f"tiny_{seed}"
        build_instance(f"tiny_{seed}", 30, seed, out)
        paths.append(out)
    return paths


class TestCompareMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.instances = tiny_instances(root / "inst")
        cls.out = root / "out"
        cls.argv = ["--name", "t", "--out", str(cls.out), "--instances", *map(str, cls.instances),
                    "--seeds", "1", "--budget", "150", "--workers", "1",
                    "--methods", "current", "C_cluster", "--ablation-instances", "1"]
        cls.folder = cm.main(cls.argv)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rows(self):
        with open(self.folder / "runs.csv", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_outputs_written(self):
        for name in ("runs.csv", "summary.csv", "curves.json", "report.html"):
            self.assertTrue((self.folder / name).exists(), name)
        # 2 instances x 1 seed x 2 methods + ablation 1 instance x 1 seed x 2 methods
        rows = self.rows()
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(r["error"] == "" for r in rows), [r["error"] for r in rows])
        html = (self.folder / "report.html").read_text(encoding="utf-8")
        self.assertIn(cm.LABELS["current"], html)
        self.assertIn(cm.LABELS["C_cluster"], html)

    def test_resume_skips_done_runs(self):
        before = len(self.rows())
        cm.main(self.argv)
        self.assertEqual(len(self.rows()), before)

    def test_failing_method_is_recorded(self):
        cm.METHODS["boom"] = lambda data, seed, budget, freeze: 1 / 0
        try:
            argv = [a if a != "t" else "t2" for a in self.argv]
            argv = argv[: argv.index("--methods") + 1] + ["boom", "current"] + \
                argv[argv.index("--ablation-instances"):]
            folder = cm.main(argv)
            with open(folder / "runs.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            boom = [r for r in rows if r["method"] == "boom"]
            self.assertTrue(boom and all("ZeroDivisionError" in r["error"] for r in boom))
            self.assertTrue((folder / "report.html").exists())
        finally:
            cm.METHODS.pop("boom", None)


    def _job(self, method):
        return {"method": method, "instance": self.instances[0].name, "path": str(self.instances[0]),
                "seed": 1, "freeze": False, "budget": 60}

    def test_variant_method_runs(self):
        from experiments.b_variants import VARIANTS
        self.assertIn("B_ring", VARIANTS)
        row, curve = cm.run_job(self._job("B_ring"))
        self.assertEqual(row["error"], "")
        self.assertTrue(curve)

    def test_unknown_method_row_error(self):
        row, _ = cm.run_job(self._job("nope"))
        self.assertIn("KeyError", row["error"])
        with self.assertRaises(KeyError):
            cm.resolve_method("nope")


if __name__ == "__main__":
    unittest.main()
