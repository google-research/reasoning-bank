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

"""Offline CLI regressions: real entrypoints/registries, fake external services."""

import contextlib
import importlib
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
MODELS = ("gemini-2.5-flash", "gemini-2.5-pro", "claude-3-7-sonnet@20250219")
MODES = ("reasoningbank", "awm", "synapse", "no_memory")


def sdk_stubs():
    """Allow the actual client modules to import without installing SDKs."""
    modules = {name: types.ModuleType(name) for name in (
        "openai", "numpy", "PIL", "PIL.Image", "google", "google.genai",
        "google.genai.types", "anthropic",
    )}
    modules["openai"].OpenAI = Mock(side_effect=AssertionError("Unexpected API call"))
    modules["openai"].ChatCompletion = type("ChatCompletion", (), {})
    modules["numpy"].ndarray = type("ndarray", (), {})
    modules["PIL.Image"].Image = type("Image", (), {})
    modules["PIL"].Image = modules["PIL.Image"]
    modules["google"].genai = modules["google.genai"]
    modules["google.genai"].Client = Mock(side_effect=AssertionError("Unexpected API call"))
    for name in ("HttpOptions", "GenerateContentConfig"):
        setattr(modules["google.genai.types"], name, Mock())
    modules["anthropic"].AnthropicVertex = Mock(side_effect=AssertionError("Unexpected API call"))
    return modules


class MemoryModelContractTest(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(contextlib.chdir(root))
        self.stack.enter_context(patch.object(sys, "path", [str(WEB), *sys.path]))
        self.stack.enter_context(patch.dict(sys.modules, sdk_stubs()))
        self.clients = importlib.import_module("utils.clients")
        self.eval_clients = importlib.import_module("autoeval.clients")
        Path("config_files").mkdir()
        Path("config_files/21.json").write_text(json.dumps({
            "task_id": 21, "sites": ["shopping"], "intent": "fixture query",
            "intent_template_id": 1,
        }))
        Path("results/webarena.21").mkdir(parents=True)
        Path("results/webarena.21/summary_info.json").write_text('{"cum_reward": 1}')

    def cli(self, name, *args):
        with patch.object(sys, "argv", [name, *args]):
            with contextlib.redirect_stdout(io.StringIO()):
                return runpy.run_path(str(WEB / name), run_name="__main__")

    def test_pipeline_passes_model_to_every_stage(self):
        for model in MODELS:
            for mode in MODES:
                with self.subTest(model=model, mode=mode), patch("subprocess.Popen") as launch:
                    launch.return_value.wait.return_value = 0
                    self.cli("pipeline_memory.py", "--website", "shopping",
                             "--output_dir", "results", "--model", model, "--memory_mode", mode)
                    commands = [call.args[0] for call in launch.call_args_list]
                    self.assertEqual(len(commands), 2 if mode == "no_memory" else 3)
                    self.assertEqual(commands[0][commands[0].index("--model_name") + 1], model)
                    self.assertEqual(commands[1][commands[1].index("--model") + 1], model)
                    self.assertIn(model, self.eval_clients.CLIENT_DICT)
                    if mode != "no_memory":
                        self.assertEqual(commands[2][commands[2].index("--model") + 1], model)
                        self.assertIn(model, self.clients.CLIENT_DICT)

    def test_unsupported_pipeline_models_never_start_a_subprocess(self):
        for model in ("google/gemma-3-12b-it", "unknown-model"):
            for mode in MODES:
                with self.subTest(model=model, mode=mode), patch("subprocess.Popen") as launch:
                    error_text = io.StringIO()
                    with contextlib.redirect_stderr(error_text):
                        with self.assertRaises(SystemExit) as error:
                            self.cli("pipeline_memory.py", "--website", "shopping",
                                     "--output_dir", "results", "--model", model, "--memory_mode", mode)
                    self.assertEqual(error.exception.code, 2)
                    self.assertIn("invalid choice", error_text.getvalue())
                    self.assertIn(model, error_text.getvalue())
                    launch.assert_not_called()

    def test_induction_cli_selects_existing_client_and_writes_memory(self):
        for supplied in (*MODELS, "gpt-4", "gpt-3.5", "gpt-3.5-turbo"):
            canonical = "gpt-3.5-turbo" if supplied == "gpt-3.5" else supplied
            with self.subTest(model=supplied):
                client_class = self.clients.CLIENT_DICT[canonical]
                Path(f"results/webarena.21/{canonical}_autoeval.json").write_text(
                    '[{"rm": 1, "thoughts": "fixture reason"}]')
                with patch.object(client_class, "one_step_chat", autospec=True,
                                  return_value=("fixture memory", None)) as chat:
                    self.cli("induce_memory.py", "--result_dir", "results", "--task", "webarena.21",
                             "--output_path", "memory.jsonl", "--model", supplied)
                    self.assertEqual(chat.call_args.args[0].model_name, canonical)
                    self.assertIn("fixture reason", chat.call_args.args[1])
                    self.assertEqual(chat.call_args.kwargs["temperature"], 1.0)
                    self.assertEqual(json.loads(Path("memory.jsonl").read_text().splitlines()[-1])[
                        "memory_items"], ["fixture memory"])

    def test_evaluator_cli_accepts_pipeline_models_and_selects_client(self):
        evaluator = importlib.import_module("autoeval.evaluator")
        for model in MODELS:
            with self.subTest(model=model), patch.object(evaluator.Evaluator, "__call__", autospec=True,
                    return_value=({"status": "success", "thoughts": "fixture"}, None)) as evaluate:
                self.cli("autoeval/evaluate_trajectory.py", "--result_dir", "results/webarena.21",
                         "--model", model, "--log_dir", "logs")
                actual = evaluate.call_args.args[0].lm_clients[model]
                self.assertIsInstance(actual, self.eval_clients.CLIENT_DICT[model])
                self.assertEqual(actual.model_name, model)
                self.assertTrue(json.loads(Path(f"results/webarena.21/{model}_autoeval.json").read_text())[0]["rm"])

    def test_default_flash_remains_unchanged(self):
        with patch("subprocess.Popen") as launch:
            self.cli("pipeline_memory.py", "--website", "shopping", "--output_dir", "results")
            self.assertEqual(launch.call_count, 3)
            for call in launch.call_args_list:
                self.assertIn("gemini-2.5-flash", call.args[0])
        Path("results/webarena.21/gemini-2.5-flash_autoeval.json").write_text('[{"rm": 1}]')
        with patch.object(self.clients.GEMINI_Client, "one_step_chat", autospec=True,
                          return_value=("default memory", None)) as chat:
            self.cli("induce_memory.py", "--result_dir", "results", "--task", "webarena.21",
                     "--output_path", "default.jsonl")
            self.assertEqual(chat.call_args.args[0].model_name, "gemini-2.5-flash")

    def test_legacy_gpt4o_synapse_cli_remains_accepted(self):
        self.cli("induce_memory.py", "--result_dir", "results", "--task", "webarena.21",
                 "--output_path", "synapse.jsonl", "--model", "gpt-4o",
                 "--criteria", "gt", "--memory_mode", "synapse")
        self.assertIn("fixture query", Path("synapse.jsonl").read_text())

    def test_inducer_rejects_unknown_model(self):
        error_text = io.StringIO()
        with contextlib.redirect_stderr(error_text):
            with self.assertRaises(SystemExit) as error:
                self.cli("induce_memory.py", "--output_path", "unexpected.jsonl",
                         "--model", "unknown-model")
        self.assertEqual(error.exception.code, 2)
        self.assertIn("invalid choice", error_text.getvalue())
        self.assertFalse(Path("unexpected.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
