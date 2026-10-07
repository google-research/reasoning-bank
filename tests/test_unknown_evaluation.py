# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real CLI regressions using synthetic trajectories and provider doubles."""

import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


WEB = Path(__file__).resolve().parents[1] / "WebArena"
MODEL = "gemini-2.5-flash"


class UnknownEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(contextlib.chdir(self.root))
        self.stack.enter_context(patch.object(sys, "path", [str(WEB), *sys.path]))
        Path("config_files").mkdir()
        for tid in (21, 22):
            Path(f"config_files/{tid}.json").write_text(json.dumps({
                "task_id": tid, "sites": ["shopping"],
                "intent": "synthetic offline query", "intent_template_id": 1,
            }))
        Path("results/webarena.21").mkdir(parents=True)
        Path("results/webarena.21/summary_info.json").write_text('{"cum_reward": 1}')
        self.bank = Path("bank.jsonl")
        self.bank.write_text('{"task_id": "prior", "memory_items": ["existing"]}\n')
        self.before = self.bank.read_bytes()
        self.client = Mock()
        self.client.one_step_chat.return_value = ("synthetic memory", {})
        self.client_factory = Mock(return_value=self.client)
        self.evaluator = Mock()
        self.evaluator.return_value = ({"status": "success", "thoughts": "fixture"}, {})
        eval_module = types.ModuleType("autoeval.evaluator")
        eval_module.Evaluator = Mock(return_value=self.evaluator)
        eval_clients = types.ModuleType("autoeval.clients")
        eval_clients.CLIENT_DICT = {MODEL: self.client_factory}
        clients = types.ModuleType("utils.clients")
        clients.CLIENT_DICT = {MODEL: self.client_factory}
        self.stack.enter_context(patch.dict(sys.modules, {
            "autoeval.evaluator": eval_module, "autoeval.clients": eval_clients,
            "utils.clients": clients,
        }))

    def cli(self, name, *arguments):
        with patch.object(sys, "argv", [name, *arguments]):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    runpy.run_path(str(WEB / name), run_name="__main__")
                except SystemExit as result:
                    return result.code
        return 0

    def evaluate(self):
        return self.cli("autoeval/evaluate_trajectory.py", "--result_dir", "results/webarena.21",
                        "--model", MODEL, "--log_dir", "logs")

    def induce(self):
        return self.cli("induce_memory.py", "--result_dir", "results", "--task", "webarena.21",
                        "--criteria", "autoeval", "--model", MODEL,
                        "--output_path", str(self.bank))

    def test_evaluator_error_preserves_diagnostic_and_exits_nonzero(self):
        self.evaluator.side_effect = RuntimeError("synthetic provider failure")
        code = self.evaluate()
        artifact = json.loads(Path(f"results/webarena.21/{MODEL}_autoeval.json").read_text())
        self.assertIsNone(artifact[0]["rm"])
        self.assertEqual(artifact[0]["uid"], 21)
        self.assertNotEqual(code, 0)

    def test_unrecognized_evaluator_status_is_unknown(self):
        self.evaluator.return_value = ({"status": "pending", "thoughts": "fixture"}, {})
        self.assertNotEqual(self.evaluate(), 0)
        artifact = json.loads(Path(f"results/webarena.21/{MODEL}_autoeval.json").read_text())
        self.assertIsNone(artifact[0]["rm"])

    def test_valid_success_and_failure_judgments_exit_zero(self):
        for status, expected in (("success", True), ("failure", False)):
            with self.subTest(status=status):
                self.evaluator.return_value = ({"status": status, "thoughts": "fixture"}, {})
                self.assertEqual(self.evaluate(), 0)
                artifact = json.loads(Path(f"results/webarena.21/{MODEL}_autoeval.json").read_text())
                self.assertIs(artifact[0]["rm"], expected)

    def test_unknown_judgment_never_creates_client_or_appends_memory(self):
        for reward in (None, "false", 2):
            with self.subTest(reward=reward):
                Path(f"results/webarena.21/{MODEL}_autoeval.json").write_text(json.dumps([
                    {"rm": reward, "thoughts": None, "uid": 21},
                ]))
                with self.assertRaisesRegex(ValueError, "correctness signal"):
                    self.induce()
                self.client_factory.assert_not_called()
                self.assertEqual(self.bank.read_bytes(), self.before)

    def test_known_judgments_keep_memory_status_and_legacy_numeric_values(self):
        for reward, expected in ((True, "success"), (False, "fail"), (1, "success"), (0, "fail")):
            with self.subTest(reward=reward):
                self.bank.write_bytes(self.before)
                Path(f"results/webarena.21/{MODEL}_autoeval.json").write_text(json.dumps([
                    {"rm": reward, "thoughts": "fixture", "uid": 21},
                ]))
                self.assertEqual(self.induce(), 0)
                appended = json.loads(self.bank.read_text().splitlines()[-1])
                self.assertEqual(appended["task_id"], "21")
                self.assertEqual(appended["status"], expected)
                self.assertEqual(appended["memory_items"], ["synthetic memory"])

    def test_real_evaluator_error_stops_pipeline_before_induction_and_later_tasks(self):
        self.evaluator.side_effect = RuntimeError("synthetic provider failure")
        commands = []

        def start(command):
            commands.append(command)
            code = self.evaluate() if "autoeval.evaluate_trajectory" in command else 0
            process = Mock()
            process.wait.return_value = code
            return process

        with patch("subprocess.Popen", side_effect=start):
            code = self.cli("pipeline_memory.py", "--website", "shopping", "--output_dir", "results")
        self.assertNotEqual(code, 0)
        self.assertEqual(len(commands), 2)
        self.assertIn("webarena.21", commands[0])
        self.assertEqual(self.bank.read_bytes(), self.before)


if __name__ == "__main__":
    unittest.main()
