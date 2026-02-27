# ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory

## 📜 Overview

<p align="center">
    <img src="assets/reasoningbank.png" width="80%" alt="intro_case">
</p>

We introduce ReasoningBank, a memory mechanism for agents that learns from both 
successful and failed trajectories, with reasoning stored as memory content.

<p align="center">
    <img src="assets/method.png" width="100%" alt="intro_case">
</p>

Building upon this memory formulation, we propose memory-aware test-time scaling,
which leverages the bidirectional synergy between memory and test-time scaling,
establishing experience-driven memory as another scaling dimension for agent
systems.


## 📂 Code Setup
We release code for `SWE-Bench` (software engineering) and `WebArena` (web-browsing),
as in corresponding directories.

Before we start, please install required packages by running `pip install -r requirements.txt`.

### 0. LLM Configuration
Currently we support three model families: 
- **GPT**: To use GPT models (`gpt-3.5-turbo`, `gpt-4`, `gpt-4o`), you need to set your OpenAI API key as an environment variable:
  ```bash
  export OPENAI_API_KEY="your-openai-api-key"
  ```

- **Gemini & Claude**: To use Gemini models (`gemini-2.5-flash`, `gemini-2.5-pro`) or Claude (`claude-3-7-sonnet@20250219`) on Vertex AI, you need to configure Google Cloud authentication.
  1.  Install the Google Cloud CLI and log in to set up Application Default Credentials (ADC):
      ```bash
      gcloud auth application-default login
      ```
  2.  Set your project and location as environment variables, as they are required by clients like the one for Claude on Vertex AI:
      ```bash
      export GOOGLE_CLOUD_PROJECT="your-project-id"
      export GOOGLE_CLOUD_LOCATION="your-region"
      export GOOGLE_GENAI_USE_VERTEXAI="True"
      ```


### 1. WebArena
#### Docker Configuration
Make sure to correctly install `browsergym` following the [official documentation](https://github.com/ServiceNow/BrowserGym).

The next step is to download and config docker environment for WebArena. Please refer to [this tutorial](https://github.com/gasse/webarena-setup/tree/main/webarena),
executing the scripts follow the numerical order of file names. Before executing,
make sure to config the address of each website in corresponding scripts as
instructed correspondingly.

#### Directory Structure

* `WebArena/agents/`: implementation for web agents integrating with browsergym
* `WebArena/autoeval/`: llm-as-a-judge for obtaining correctness signal for trajectories
* `WebArena/config_files/`: data processing for webarena tasks
* `WebArena/prompt/`: instructions used across the implementation

#### Data preprocessing

Download raw test files from [here](https://github.com/web-arena-x/webarena/blob/main/config_files/test.raw.json) and put it to `config_files`.

Run `generate_config_files.py` to process raw test data to config files as input.


#### Run the code

Run directly with ReasoningBank: `bash run.sh`, config `model`, `output_dir`, and
`website`, and `memory_mode` accordingly.

To run with scaling setting, please refer to
`pipeline_scaling.py` and `induce_scaling.py`.

### 2. SWE-Bench
We built upon [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent). First, install it from source by `pip install -e .` under the directory of `./third_party` This will install the dependencies as specified in `pyproject.toml`.

The script `SWE-Bench/run.sh` provides direct running command, which will generate
result files in the output directory. Before running, make sure the
configuration for VertexAI is properly configured as instructed in `run.sh`.

For evaluation, please refer to `sb-cli` command in the [official documentation](https://mini-swe-agent.com/latest/usage/swebench/). 

## Acknowledgement
We adopt code from the following code repositories. We sincerely appreciate these
great work/codebases:

- [Agent-workflow-memory](https://github.com/zorazrw/agent-workflow-memory)
- [webarena](https://github.com/web-arena-x/webarena)
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)

## 📚 Citation
If you find this work useful, please kindly cite our paper:
```
@inproceedings{
  ouyang2026reasoningbank,
  title={ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory},
  author={Ouyang, Siru and Yan, Jun and Hsu, I-Hung and Chen, Yanfei and Jiang, Ke and Wang, Zifeng and Han, Rujun and Le, Long T and Daruki, Samira and Tang, Xiangru and Tirumalashetty, Vishy and Lee,
  George and Rofouei, Mahsan and Lin, Hangfei and Han, Jiawei and Lee, Chen-Yu and Pfister, Tomas},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=jL7fwchScm}
}
```

## Disclaimer

This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).

This project is intended for demonstration purposes only. It is not intended for use in a production environment.