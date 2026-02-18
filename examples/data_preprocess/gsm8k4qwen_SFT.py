# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the GSM8k dataset to parquet format for Qwen3 SFT
"""

import argparse
import os
# import re

import datasets


# def extract_solution(solution_str):
#     solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
#     assert solution is not None
#     final_solution = solution.group(0)
#     final_solution = final_solution.split("#### ")[1].replace(",", "")
#     return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default=None, help="The save directory for the preprocessed dataset.")
    
    args = parser.parse_args()
    
    data_source = "openai/gsm8k"
    dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    
    system_prompt = (
        "You are a helpful math tutor. Solve the problem step by step, "
        "showing your reasoning clearly. End with the final answer on a new line "
        "in the format: #### <number>"
    )
    
    # USE THIS FOR THE BASE MODELS, LIKE "Llama-3.1-8B"
    # def make_map_fn(split):
    #     def process_fn(example, idx):
    #         q = example["question"]
    #         a = example["answer"]

    #         prompt = f"{system_prompt}\n\nQuestion: {q}\nAnswer:"
    #         response = a

    #         return {
    #             "prompt": prompt,
    #             "response": response,
    #             # optional debugging / bookkeeping
    #             "extra_info": {"split": split, "index": idx},
    #         }
    #     return process_fn
    
    
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            """Convert a GSM8K example into a chat-formatted conversation."""
            messages = [
                {
                    "role": "system", 
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": example["question"]
                },
                {
                    "role": "assistant", 
                    "content": example["answer"]
                },
            ]
            return {
                "messages": messages,
                "extra_info": {"split": split, "index": idx}
            }#, "enable_thinking": False,}
        return process_fn

    # TODO: CHECK IF THIS WITHOUT MULTITURN WILL WORK BETTER/DIFFERENTLY
    # # add a row to each data item that represents a unique id
    # def make_map_fn(split):
    #     def process_fn(example, idx):
    #         question_raw = example.pop("question")

    #         answer_raw = example.pop("answer")
    #         solution = extract_solution(answer_raw)
    #         data = {
    #             "data_source": data_source,
    #             "prompt": [
    #                 {   
    #                     "role": "system",
    #                     "content": system_prompt,
    #                 },
    #                 {   
    #                     "role": "user",
    #                     "content": question_raw,
    #                 }
    #             ],
    #             "ability": "math",
    #             "reward_model": {"style": "rule", "ground_truth": solution},
    #             "extra_info": {
    #                 "split": split,
    #                 "index": idx,
    #                 "answer": answer_raw,
    #                 "question": question_raw,
    #             },
    #         }
    
    
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    train_dataset.to_parquet(os.path.join(args.local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(args.local_save_dir, "test.parquet"))
