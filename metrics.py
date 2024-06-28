#!/usr/bin/env python3
import string
from typing import List

import regex

def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        if isinstance(text, list):
            text = ", ".join(text)
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def normalize_tm_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_punc(s))

def normalize_math_answer(s) -> str:
    if not isinstance(s, str):
        return str(s)
    else:
        return s.replace(",", "")
    
    


def normalize_answer_based_on_task(answer, task):
    if "text" in task:
        return normalize_tm_answer(answer)
    elif "math" in task:
        return normalize_math_answer(answer)
    elif "safety" in task:
        return normalize_answer(answer)
    else:
        return normalize_answer(answer)

def best_subspan_em(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)

    normalized_ground_truth = normalize_answer(ground_truth)
    if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0

def best_subspan_em_instance(predictions, ground_truths, task):
    normalized_prediction =normalize_answer_based_on_task(predictions[-1], task)
    normalized_ground_truth = normalize_answer_based_on_task(ground_truths[-1], task)
    
    if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0

def best_subspan_em_instruction(predictions, ground_truths, task):
    ins_acc = best_subspan_em_per_instruction(predictions, ground_truths, task)
    acc_sum = 0
    for depth, acc in enumerate(ins_acc):
        acc_sum += acc
        
    return acc_sum / len(ground_truths)

def best_subspan_em_per_instruction(predictions, ground_truths, task):
    acc = []
    for i, prediction in enumerate(predictions):
        if i >= len(ground_truths):
            break
        normalized_prediction = normalize_answer_based_on_task(prediction, task)
        normalized_ground_truth = normalize_answer_based_on_task(ground_truths[i], task)
    
        if normalized_ground_truth.lower() in normalized_prediction.lower():
                acc.append(1.0)
        else:
            acc.append(0.0)
    return acc

def instruction_depth(predictions, ground_truths, task):
    ins_acc = best_subspan_em_per_instruction(predictions, ground_truths, task)
    for depth, acc in enumerate(ins_acc):
        if acc < 1.0:
            return depth
    return depth + 1


if __name__ == "__main__":
    text = " if, Normalized_ground_truth in normalized_prediction: "
    print(normalize_answer_based_on_task(text, "qa"))