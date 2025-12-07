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

import re
from typing import Any
import json
import string

from sentence_transformers import SentenceTransformer, util
sentence_tranformers_model = SentenceTransformer('all-MiniLM-L6-v2')  # light weight



def normalize(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, ground_truth):
    prediction_tokens = normalize(prediction).split()
    ground_truth_tokens = normalize(ground_truth).split()

    common = set(prediction_tokens) & set(ground_truth_tokens)
    num_same = len(common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def compute_similarity(prediction, ground_truth, sentence_tranformers_model=sentence_tranformers_model):
    emb1 = sentence_tranformers_model.encode(prediction, convert_to_tensor=True)
    emb2 = sentence_tranformers_model.encode(ground_truth, convert_to_tensor=True)
    cosine_score = util.cos_sim(emb1, emb2)
    return float(cosine_score)

def accuracy_reward(response: str, ground_truth: str) -> float:
    try:
        # Extract answer from solution if it has think/answer tags
        if '<answer>' in ground_truth:
            if '<search>' in response:
                reward = 0.0
            else:
                sol_match = re.search(r'<answer>(.*?)</answer>', ground_truth)
                ground_truth = sol_match.group(1).strip() if sol_match else ground_truth.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', response)
                student_answer = content_match.group(1).strip() if content_match else response.strip()
                
                # Compare the extracted answers
                reward = compute_f1(student_answer, ground_truth)
            
        elif '<search>' in ground_truth:
            if '<answer>' in response:
                reward = 0.0
            else:
                sol_match = re.search(r'<search>(.*?)</search>', ground_truth)
                ground_truth = sol_match.group(1).strip() if sol_match else ground_truth.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<search>(.*?)</search>', response)
                student_answer = content_match.group(1).strip() if content_match else response.strip()
                
                # Compare the extracted answers
                reward = compute_similarity(student_answer, ground_truth)

                if reward<0.5:
                    reward = 0.0
        return reward    
    except Exception:
        pass
    
    return 0.0


def format_reward(response: str) -> float:
    """Reward function that checks if the completion has a specific format."""
    pattern_answer = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern_search = r"<think>.*?</think>\s*<search>.*?</search>"
    #pattern_answer = r"<answer>.*?</answer>"
    #pattern_search = r"<search>.*?</search>"

    if response.count("<answer>")>=2 or response.count("<search>")>=2 or response.count("<think>")>=2:
        return 0.0
        # print('1111111111111111', content.count("<answer>"), content.count("<search>"), content.count("<think>"))
    elif '<answer>' in response and '</answer>' in response:
        match_answer = re.match(pattern_answer, response, re.DOTALL)
        match_search = re.match(pattern_search, response, re.DOTALL)
        #print(match_answer,match_search)
        if match_answer and not match_search:
            return 1.0
        else:
            return 0.0
            # print('2222222222222222', match_answer, match_search, content)
    elif '<search>' in response and '</search>' in response:
        match_answer = re.match(pattern_answer, response, re.DOTALL)
        match_search = re.match(pattern_search, response, re.DOTALL)
        #print(match_answer,match_search)
        if match_search and not match_answer:
            return 1.0
        else:
            return 0.0
            # print('3333333333333333', match_answer, match_search)
    else:
        return 0.0
        # print('4444444444444444', content)

def compute_score(reward_input: dict[str, Any], format_weight: float = 0.5) -> dict[str, float]:
    if not isinstance(reward_input, dict):
        raise ValueError("Please use `reward_type=sequential` for r1v reward function.")
    #print(reward_input["response"])
    format_score = format_reward(reward_input["response"])
    accuracy_score = accuracy_reward(reward_input["response"], reward_input["ground_truth"])
    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }