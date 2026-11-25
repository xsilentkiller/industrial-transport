"""Run every test in tests/ once and print a per-file summary.

    python run_tests.py          # all tests, short output + summary table
    python run_tests.py -v       # also list every test as it runs
    python main.py --mode tests  # the same

Exit code 0 = everything passed, 1 = at least one failure or error.
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# What each test file checks and on which data (shown in the summary).
ABOUT = {
    "test_cost": "trip cost by hand, overtime, overflow fee      | data/base + synthetic",
    "test_day_window": "working day, 2x overtime, day-end rules       | synthetic + data/base",
    "test_decoder": "tour -> trips, coverage, 10-50, capacity      | data/base",
    "test_convergence": "PSO fixes: fast 2-opt, scoring, seeding       | data/base",
    "test_report": "report files and charts                       | data/base",
    "test_instances": "straight-line instance generator              | data/base",
    "test_road": "road mode, GA cross-check, OSM data, pipeline | data/road_ga + data/osm + synthetic",
}


class _Collect(unittest.TextTestResult):
    """Remembers the outcome of every test, keyed by its test file."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcomes: list[tuple[str, str]] = []

    def _file(self, test) -> str:
        return test.id().split(".")[1] if test.id().startswith("tests.") else test.id().split(".")[0]

    def addSuccess(self, test):
        super().addSuccess(test)
        self.outcomes.append((self._file(test), "pass"))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.outcomes.append((self._file(test), "fail"))

    def addError(self, test, err):
        super().addError(test, err)
        self.outcomes.append((self._file(test), "fail"))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.outcomes.append((self._file(test), "skip"))


def run_all(verbose: bool = False) -> int:
    # Tests use paths relative to the project root (e.g. "data/base") and
    # import src/ and experiments/, so run from the root wherever we start.
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    suite = unittest.TestLoader().discover("tests")
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1, resultclass=_Collect)
    t0 = time.perf_counter()
    result = runner.run(suite)
    elapsed = time.perf_counter() - t0

    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "fail": 0, "skip": 0})
    for name, outcome in result.outcomes:
        counts[name][outcome] += 1

    print("\n" + "=" * 100)
    print(f"{'Test file':18s} {'tests':>5s} {'pass':>5s} {'fail':>5s} {'skip':>5s}  what it checks | data")
    print("-" * 100)
    for name in sorted(counts):
        c = counts[name]
        total = sum(c.values())
        print(f"{name:18s} {total:5d} {c['pass']:5d} {c['fail']:5d} {c['skip']:5d}  {ABOUT.get(name, '')}")
    print("-" * 100)
    n_pass = sum(c["pass"] for c in counts.values())
    n_fail = sum(c["fail"] for c in counts.values())
    n_skip = sum(c["skip"] for c in counts.values())
    print(f"{'TOTAL':18s} {result.testsRun:5d} {n_pass:5d} {n_fail:5d} {n_skip:5d}  in {elapsed:.1f} s")
    print("ALL TESTS PASSED" if result.wasSuccessful() else "SOME TESTS FAILED (details above)")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_all(verbose="-v" in sys.argv[1:]))
