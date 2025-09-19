# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os

# disable tensorflow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
import time

import openai
import torch
from tqdm import trange
from pathlib import Path

from api.run import LLMClient
from text2analysis import Text2AnalysisDataset

CODE_GENERATION_EXAMPLE=[
        {
            "role":"user",
            "content":"""
PROBLEM:
I want you to act as a data analytics scientist. I will give you a tables, an user query, and you need to generate Python code to answer the user query. You should follow the rules:
1. Return the Python code.
2. There is already a DataFrame variable for `tables` in the environment, so you should not reassign `tables=pd.read_html(...)` in the code.
3. Stores the final result in the variable `result`. I will directly utilize the 'result' variable for assessing the subsequent code outcomes.

Given the HTML tables:
<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th>Time</th>\n      <th>Type</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <td>12:23</td>\n      <td>a</td>\n    </tr>\n    <tr>\n      <td>3:23</td>\n      <td>a</td>\n    </tr>\n    <tr>\n      <td>22:14</td>\n      <td>b</td>\n    </tr>\n    <tr>\n      <td>None</td>\n      <td>c</td>\n    </tr>\n  </tbody>\n</table>

The user query is:
I have a dataframe, sort it, and then delete the earliest row."""
        },
        {
            "role":"assistant",
            "content":"""
```python
import pandas as pd

# Convert the Time column to a time format that can be compared.
table['Time'] = pd.to_datetime(table['Time'], errors='coerce').dt.time

# Sort the DataFrame.
table = table.sort_values(by='Time')

# Delete the first row.
result = table.iloc[1:]
```"""
        }]

def normalize(raw_text: str):
    # replace the html symbols '<' '>' '&'
    if "&lt;" in raw_text:
        raw_text = raw_text.replace("&lt;", "<")
    if "&gt;" in raw_text:
        raw_text = raw_text.replace("&gt;", ">")
    if "&amp;" in raw_text:
        raw_text = raw_text.replace("&amp;", "&")

    return raw_text


llm_client = LLMClient()


def gpt4_call(
        prefix: str,system=None,example=None, suffix: str = None, batch_size: int = 1, args=None
):
    """
    gpt4 call
    """
    response = None

    while response is None:
        try:
            """codes only"""
            response=llm_client.run_llm(prefix,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["</code>", "# SOLUTION END"],
                logprobs=1,
                n=batch_size,)
            for batch_i in range(batch_size):
                if "choices" in response and len(response["choices"][batch_i]["text"]) > 0:
                    response["choices"][batch_i]["text"] = normalize(
                        response["choices"][batch_i]["text"]
                    )
            if response is not None and "choices" not in response:
                print(response)
            return response
        except (openai.error.RateLimitError, openai.error.APIConnectionError) as e:
            print(type(e), e)
            time.sleep(20)

def gpt4_inference(output_dir: Path, prefix: str, suffix: str, args):
    """
    Calls OpenAI API to get Codex responses and save prediction text and logprobs
    """
    # if outputs are already cached, we skip this sample
    for batch_start in range(0, args.num_samples, args.batch_size):
        all_cached = True
        for batch_i in range(min(args.num_samples - batch_start, args.batch_size)):
            if not all(
                    [
                        not args.overwrite_output_dir,
                        (output_dir / f"{batch_start + batch_i}.py").is_file(),
                        (output_dir / f"{batch_start + batch_i}.json").is_file(),
                    ]
            ):
                all_cached = False
        if all_cached:
            continue

        response = gpt4_call(
            prefix,
            example=CODE_GENERATION_EXAMPLE,
            suffix= None,
            batch_size=min(args.num_samples - batch_start, args.batch_size),
            args=args
        )

        # storing responses
        for batch_i in range(min(args.num_samples - batch_start, args.batch_size)):
            with open(
                    output_dir / f"{batch_start + batch_i}.py", "w", encoding="UTF-8"
            ) as f:
                f.write(response["choices"][batch_i]["text"] if "choices" in response else "")
                
            result = dict()
            result["trg_prediction"] = response["choices"][batch_i]["text"] if "choices" in response else ""
            result["logprobs"] = response["choices"][batch_i]["logprobs"][
                "token_logprobs"
            ] if "choices" in response else []
            result["tokens"] = response["choices"][batch_i]["logprobs"]["tokens"] if "choices" in response else []
            result["prompt"] = prefix

            with open(output_dir / f"{batch_start + batch_i}.json", "w") as f:
                json.dump(result, f)


def model_inference(output_dir: Path, prefix: str, suffix: str = None,args=None):
    """
    provide an interface for different models to use this inference framework
    """
    global model, tokenizer
    os.makedirs(output_dir, exist_ok=True)
    gpt4_inference(output_dir, prefix, suffix,args)


def inference(text2analysis, args):
    """
    A function for model inference
    """
    for problem_id in trange(len(text2analysis)):
        prefix = text2analysis[problem_id].prompt
        suffix = ""
        model_inference(
            args.output_dir
            / args.model
            / ("q" + str(problem_id)),
            prefix,
            suffix,
            args
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt4",
        choices=["gpt4","gpt-35-turbo-16k-0613","gpt-4-turbo","gpt-4-32k-0613"],
        help="Type of Codex Model to run",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./codex_greedy_outputs",
        help="Path to which the Codex responses will be cached at",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="Tables/text2analysis_1220.csv",
        help="Path to the downloaded Text2analysis data",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of Codex samples to draw for one question.",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        default=False,
        help="By default will skip cached samples; Turn on this flag to overwrite the existing caches.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature of the Codex sampling distribtuion.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p cutoff of the Codex sampling distribtuion",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Number of maximum tokens for Codex to generate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Number of requests to issue at one time",
    )
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.source_dir = Path(args.source_dir)
    text2analysis = Text2AnalysisDataset(source_dir=args.source_dir)
    print("loaded dataset")
    inference(text2analysis, args)
