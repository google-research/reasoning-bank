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

"""Offline subprocess-boundary tests for the WebArena memory pipeline."""

import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


PIPELINE = Path(__file__).resolve().parents[1] / "WebArena" / "pipeline_memory.py"


class PipelineFailurePropagationTest(unittest.TestCase):
    def run_pipeline(self, exit_codes, memory_mode="reasoningbank", existing=False):
        root = Path(self.enter_context(tempfile.TemporaryDirectory()))
        (root / "config_files").mkdir()
        (root / "config_files/21.json").write_text(json.dumps({
            "task_id": 21,
            "sites": ["shopping"],
        }))
        if existing:
            result_dir = root / "results/webarena.21"
            result_dir.mkdir(parents=True)
            (result_dir / "summary_info.json").write_text("old result")
            (result_dir / "gemini-2.5-flash_autoeval.json").write_text("old evaluation")

        processes = []

        def start(command):
            process = Mock()
            process.wait.return_value = exit_codes[len(processes)]
            processes.append(command)
            return process

        stderr = io.StringIO()
        with contextlib.chdir(root), patch.object(sys, "argv", [
            "pipeline_memory.py", "--website", "shopping", "--output_dir", "results",
            "--memory_mode", memory_mode,
        ]), patch("subprocess.Popen", side_effect=start), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as result:
                runpy.run_path(str(PIPELINE), run_name="__main__")
        return result.exception.code, processes, stderr.getvalue()

    def test_success_runs_all_three_stages(self):
        code, commands, stderr = self.run_pipeline([0, 0, 0])
        self.assertEqual(code, 0)
        self.assertEqual(len(commands), 3)
        self.assertIn("run.py", commands[0])
        self.assertIn("autoeval.evaluate_trajectory", commands[1])
        self.assertIn("induce_memory.py", commands[2])
        self.assertNotIn("[ERROR]", stderr)

    def test_actor_failure_stops_before_evaluation_and_induction(self):
        code, commands, stderr = self.run_pipeline([7, 0, 0], existing=True)
        self.assertEqual(code, 7)
        self.assertEqual(len(commands), 1)
        self.assertIn("run.py", commands[0])
        self.assertIn("webarena.21 actor", stderr)
        self.assertIn("7", stderr)

    def test_evaluator_failure_stops_before_induction(self):
        code, commands, stderr = self.run_pipeline([0, 9, 0])
        self.assertEqual(code, 9)
        self.assertEqual(len(commands), 2)
        self.assertIn("autoeval.evaluate_trajectory", commands[1])
        self.assertIn("webarena.21 evaluator", stderr)
        self.assertIn("9", stderr)

    def test_induction_failure_returns_nonzero_after_three_stages(self):
        code, commands, stderr = self.run_pipeline([0, 0, 11])
        self.assertEqual(code, 11)
        self.assertEqual(len(commands), 3)
        self.assertIn("induce_memory.py", commands[2])
        self.assertIn("webarena.21 inducer", stderr)
        self.assertIn("11", stderr)

    def test_no_memory_has_only_actor_and_evaluator(self):
        code, commands, stderr = self.run_pipeline([0, 0], memory_mode="no_memory", existing=True)
        self.assertEqual(code, 0)
        self.assertEqual(len(commands), 2)
        self.assertNotIn("induce_memory.py", commands[-1])
        self.assertNotIn("[ERROR]", stderr)

    def test_failure_blocks_later_tasks(self):
        root = Path(self.enter_context(tempfile.TemporaryDirectory()))
        (root / "config_files").mkdir()
        for task_id in (21, 22):
            (root / f"config_files/{task_id}.json").write_text(json.dumps({
                "task_id": task_id,
                "sites": ["shopping"],
            }))
        commands = []

        def start(command):
            process = Mock()
            process.wait.return_value = 13
            commands.append(command)
            return process

        with contextlib.chdir(root), patch.object(sys, "argv", [
            "pipeline_memory.py", "--website", "shopping", "--output_dir", "results",
        ]), patch("subprocess.Popen", side_effect=start), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                runpy.run_path(str(PIPELINE), run_name="__main__")
        self.assertEqual(result.exception.code, 13)
        self.assertEqual(len(commands), 1)
        self.assertIn("webarena.21", commands[0])

    def setUp(self):
        self.addCleanup(self._cleanup_context)
        self._contexts = []

    def enter_context(self, context):
        self._contexts.append(context)
        return context.__enter__()

    def _cleanup_context(self):
        for context in reversed(self._contexts):
            context.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
