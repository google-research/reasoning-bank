# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import gzip
import pickle
import json
import argparse
import sys
import traceback
from autoeval.evaluator import Evaluator
from autoeval.clients import CLIENT_DICT


def _extract_think_from_output(ai: dict) -> str:
    """Extract think text from agent info, falling back to chat_messages."""
    think = ai.get("think", "")
    if think:
        return think
    # Model may put reasoning outside <think> tags — extract text before <action>
    msgs = ai.get("chat_messages", [])
    if len(msgs) >= 3:
        output = str(msgs[2])
        action_idx = output.find("<action>")
        if action_idx > 0:
            raw = output[:action_idx].strip()
            raw = re.sub(r"</?think>", "", raw).strip()
            if raw:
                return raw
    return think


def extract_think_and_action(folder: str) -> tuple[list[str], list[str], list[str]]:
    """Extract think/action/obs triples from step pkl files."""
    step_files = sorted(
        [f for f in os.listdir(folder) if re.match(r"step_\d+\.pkl\.gz", f)],
        key=lambda f: int(re.findall(r"\d+", f)[0])
    )
    think_list = []
    action_list = []
    obs_list = []
    for f in step_files:
        try:
            with gzip.open(os.path.join(folder, f), 'rb') as fh:
                data = pickle.load(fh)
            ai = data.agent_info
            think = _extract_think_from_output(ai)
            action = ai.get("action", "")
            if not action:  # skip empty action steps
                continue
            think_list.append(think)
            action_list.append(action)
            # Extract actual page content from observation for autoeval
            obs = data.obs
            axtree = obs.get("axtree_txt", "") if isinstance(obs, dict) else ""
            obs_list.append(axtree)
        except Exception:
            continue
    return think_list, action_list, obs_list

def extract_response(action):
    s, e = action.index("(")+1, action.index(")")
    return action[s: e]


def process_sample(
    idx, traj_info, log_save_path,
    model, eval_version,
):
    clients = {model: CLIENT_DICT[model](model_name=model)}
    evaluator = Evaluator(clients, log_save_path=log_save_path + "/trajs")
    try:
        out, _ = evaluator(traj_info, model, eval_version)
        status = out["status"].strip().lower()
        if status not in ("success", "failure", "fail"):
            raise ValueError(f"Invalid evaluation status: {status!r}")
        eval_result = status == "success"
        return [{
                "idx": idx,
                "gt": traj_info["eval"],
                "rm": eval_result,
                "thoughts": out["thoughts"],
                "uid": traj_info["traj_name"],
        }]
    except Exception as e:
        print(f"Error on {idx}, {e}")
        print(traceback.format_exc())
        return [{
            "idx": idx,
            "gt": traj_info["eval"],
            "rm": None,
            "thoughts": None,
            "uid": traj_info["traj_name"],
        }]


def main():
    # load task config
    task_id = args.result_dir.split('/')[-1].split(".")[1]
    config_path = os.path.join("config_files", f"{task_id}.json")
    config = json.load(open(config_path))

    # load trajectory from pkl step files
    think_list, action_list, obs_list = extract_think_and_action(args.result_dir)
    actions = [act for act in action_list]
    if action_list and "send_msg_to_user" in action_list[-1]:
        response = extract_response(action_list[-1])
    else:
        response = ""

    # load summary info
    summary_path = os.path.join(args.result_dir, "summary_info.json")
    summary = json.load(open(summary_path, 'r'))

    # collect traj info
    image_paths = [
        os.path.join(args.result_dir, f) for f in os.listdir(args.result_dir)
        if f.startswith("screenshot_step_") and (f.endswith(".jpg") or f.endswith(".png"))
    ]
    image_paths = sorted(image_paths, key=lambda x: int(x.split('/')[-1].split("_")[-1].split(".")[0]))
    # Use actual page content (axtree_txt) as captions for autoeval.
    # The agent's think text is often empty, leaving the evaluator blind.
    # The axtree gives the evaluator real webpage content to verify answers against.
    # Truncate to avoid exceeding token limits.
    MAX_OBS_CHARS = 40000
    captions_for_eval = [obs[:MAX_OBS_CHARS] if obs else think
                         for obs, think in zip(obs_list, think_list)]

    traj_info = {
        "intent": config["intent"],
        "response": response,
        "captions": captions_for_eval,
        "thinks": think_list,
        "actions": actions,
        "traj_name": config["task_id"],
        "image_paths": image_paths,
        "images": image_paths,
        "eval": summary["cum_reward"]
    }

    # evaluate trajectory
    log_save_path = os.path.join(args.log_dir, args.result_dir.split('/')[-1])
    print("Log Save Path:", log_save_path)
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)
        os.makedirs(log_save_path + "/trajs")
    eval_info = process_sample(
        idx=config["task_id"], traj_info=traj_info,
        log_save_path=log_save_path,
        model=args.model, eval_version=args.prompt,
    )
    output_eval_path = os.path.join(args.result_dir, f"{args.model}_autoeval.json")
    json.dump(eval_info, open(output_eval_path, 'w'))
    # Keep the diagnostic artifact, but do not report an unknown judgment as
    # successful evaluation to the pipeline.
    return 0 if all(isinstance(row["rm"], bool) for row in eval_info) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Path to the result directory, e.g., 'webarena.0'.")
    # autoeval
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                        choices=['gemini-2.5-flash', 'claude-3-7-sonnet@20250219', 'gemini-2.5-pro', "google/gemma-3-12b-it"])
    parser.add_argument("--prompt", type=str, default="text",
                        choices=["text", "vision"])
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Path to the output directory.")

    args = parser.parse_args()

    if args.model == "gpt-4o" and args.prompt != "vision":
        print(f"Waring: use vision prompt by default for {args.model}.")
        args.prompt = "vision"

    sys.exit(main())
