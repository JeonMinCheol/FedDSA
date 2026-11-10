import json
from pathlib import Path
import tempfile
import unittest

from group_feddsa_experiments import group_summary, refresh


class GroupSummaryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def run_log(self, seed, rounds):
        path = self.folder / f"seed{seed}.jsonl"
        path.write_text("\n".join(json.dumps({"seed": seed, "round": r,
                                             "test_accuracy": 0.4 + seed / 100})
                                  for r in rounds) + "\n")
        return {"seed": seed, "metrics": str(path)}

    def test_incomplete_seed_never_becomes_a_final_group_mean(self):
        group = {"id": "fixture", "runs": [self.run_log(0, range(450, 501, 10)),
                                           self.run_log(1, [450, 460, 470, 480, 490])]}
        result = group_summary(group)
        self.assertIsNone(result["round500_accuracy"])
        self.assertIsNone(result["last50_accuracy"])
        group["runs"][1] = self.run_log(1, range(450, 501, 10))
        result = group_summary(group)
        self.assertAlmostEqual(result["round500_accuracy"], 0.405)
        self.assertAlmostEqual(result["last50_accuracy"], 0.405)

    def test_mixed_seeds_and_repeated_rounds_are_rejected(self):
        run = self.run_log(1, [500])
        run["seed"] = 0
        with self.assertRaises(ValueError):
            group_summary({"id": "fixture", "runs": [run]})
        run = self.run_log(0, [500, 500])
        with self.assertRaises(ValueError):
            group_summary({"id": "fixture", "runs": [run]})

    def test_missing_late_evaluation_is_not_a_complete_late_mean(self):
        result = group_summary({"id": "fixture", "runs": [self.run_log(0, [500])]})
        self.assertEqual(result["round500_accuracy"], 0.4)
        self.assertIsNone(result["last50_accuracy"])

    def test_refresh_keeps_real_files_and_shared_reference_has_no_extra_directory(self):
        output = self.folder / "groups"
        group = output / "fixture"
        group.mkdir(parents=True)
        run = self.run_log(0, [500])
        metrics = group / "seed0.metrics.jsonl"
        Path(run["metrics"]).rename(metrics)
        stdout = group / "seed0.out"
        stdout.write_text("training log")
        config = self.folder / "groups.json"
        config.write_text(json.dumps({"output_dir": str(output), "note": "fixture",
            "aliases": {"reference": "fixture"}, "groups": [{"id": "fixture", "label": "fixture",
            "protocol": "fixture", "runs": [{"job_id": 42, "seed": 0,
            "stdout": str(stdout), "metrics": str(metrics)}]}]}))
        original = metrics.read_bytes()
        refresh(config)
        refresh(config)
        self.assertEqual(metrics.read_bytes(), original)
        self.assertFalse(any(p.is_symlink() for p in output.rglob("*")))
        self.assertFalse((output / "reference").exists())


if __name__ == "__main__":
    unittest.main()
