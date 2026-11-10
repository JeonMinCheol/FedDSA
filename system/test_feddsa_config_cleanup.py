import argparse
import ast
import os
from pathlib import Path
import re
import subprocess
import unittest


SYSTEM = Path(__file__).resolve().parent


def current_parser():
    # Check the actual CLI without importing unrelated algorithm backends.
    tree = ast.parse((SYSTEM / "main.py").read_text())
    parser = argparse.ArgumentParser()
    namespace = {"parser": parser, "str2bool": lambda value: value.lower() == "true"}
    for node in sorted(ast.walk(tree), key=lambda node: getattr(node, "lineno", 0)):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "experimental"
            for target in node.targets
        ):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"), namespace)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == "add_argument":
                exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"), namespace)
    return parser, namespace["experimental"]


class ConfigCleanupTest(unittest.TestCase):
    def test_retired_options_absent_and_experiments_labeled(self):
        parser, experimental = current_parser()
        for name in parser._option_string_actions:
            self.assertFalse(name.startswith(("--open_world", "--domain_shift", "--cohort_shift")))
        for name in ("--linear_head", "--kappa", "--learning_rate_decay",
                     "--learning_rate_decay_gamma", "--data_loader_num_workers",
                     "--stability_guard", "--trust_region", "--margin_gate",
                     "--head_mode", "--projector_mode", "--anchor_mode",
                     "--utility_server_gate"):
            self.assertNotIn(name, parser._option_string_actions)
        self.assertTrue(experimental._group_actions)
        for action in experimental._group_actions:
            self.assertTrue(action.help.startswith("[EXPERIMENTAL]"), action.dest)
        self.assertIn("실험용", parser.format_help())

    def test_all_launcher_modes_parse_without_starting_training(self):
        parser, _ = current_parser()
        script = (SYSTEM / "submit_feddsa_shift_compare.sbatch").read_text()
        launch = 'exec "${PYTHON_BIN}" -u main.py "${COMMON[@]}"'
        self.assertEqual(script.count(launch), 1)
        script = script.replace(launch, "printf '%s\\0' \"${COMMON[@]}\"")
        modes = re.findall(r'\$\{MODE\}" == "([a-z_]+)"', script)
        self.assertIn("utility_single", modes)
        for mode in modes:
            with self.subTest(mode=mode):
                result = subprocess.run(
                    ["bash", "-s"], input=script, text=True, capture_output=True,
                    env={**os.environ, "MODE": mode}, check=True,
                )
                args = parser.parse_args(result.stdout.rstrip("\0").split("\0"))
                self.assertEqual(args.global_rounds, 500)


if __name__ == "__main__":
    unittest.main()
