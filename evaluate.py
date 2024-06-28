import argparse

#!/usr/bin/env python3
"""Given a data file with LM QA predictions, evaluate the predictions.
"""
import argparse
import json
import logging
import sys
from copy import deepcopy
import os

from tqdm import tqdm
from xopen import xopen
import numpy as np

import wandb
import re

from metrics import best_subspan_em_instance, best_subspan_em_instruction, \
    instruction_depth, normalize_answer_based_on_task, best_subspan_em_per_instruction

logger = logging.getLogger(__name__)

METRICS = [
    (best_subspan_em_instance, "instance_acc"),
    (best_subspan_em_instruction, "instruction_acc"),
    (instruction_depth, "instruction_depth"),
    (best_subspan_em_per_instruction, "acc_per_step")
]

def get_config_from_input_path(input_path):
    input_paths = input_path.split("/")
    dataset = input_paths[-3]
    task = re.search('_(.+?).jsonl', input_paths[-1]).group(1)
    model_name = input_paths[-2]
    
    return dataset, task, model_name

def get_shown_model_name(model_name):
    if "llama-2-7b" in model_name.lower():
        return "LLaMA2-7B-Chat"
    elif "llama-2-70b" in model_name.lower():
        return "LLaMA2-70B-Chat"
    elif "llama-3-8B" in model_name.lower():
        return "LLaMA3-8B-Instruct"
    elif "llama-3-70B" in model_name.lower():
        return "LLaMA3-70B-Instruct"
    elif "gpt-4" in model_name.lower():
        return "GPT4"
    elif "mistral" in model_name.lower():
        return "Mistral-7B-Instruct"
    else:
        return model_name

def eval_per_file(
    response_path,
    output_dir,
):
    dataset, task, model_name = get_config_from_input_path(response_path)
    shown_model_name = get_shown_model_name(model_name)
    
    config={
        "model_name": shown_model_name,
        "task": task,
        }
    
    run = wandb.init(
        # set the wandb project where this run will be logged
        project= "sif-2024",
        name= shown_model_name,
        # track hyperparameters and run metadata
        config= config
    )     
    
    all_examples = []
    all_example_metrics = []
    with xopen(response_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            all_examples.append(input_example)
    
    # Compute normal metrics in parallel, if applicable
    logger.info("Computing metrics")
    all_example_metrics = []
    for example in tqdm(all_examples):
        example_metric = get_metrics_for_example(example, METRICS, task)
        if example_metric != None:
            all_example_metrics.append(example_metric)

    # Average metrics across examples
    metric = {}
    for (_, metric_name) in METRICS:
        metric_values = [example_metrics[metric_name] for (example_metrics, _) in all_example_metrics]
        if isinstance(metric_values[0], list):
            sum_metric_value = []
            num_metric_value = []
            for metric_value in metric_values:
                for i, v in enumerate(metric_value):
                    if i < len(sum_metric_value):
                        sum_metric_value[i] += v
                        num_metric_value[i] += 1
                    else:
                        sum_metric_value.append(v)
                        num_metric_value.append(1)
            average_metric_value = np.divide(sum_metric_value, num_metric_value)
            
            if metric_name == "acc_per_step" and "qa" in task:
                average_metric_value = np.round(average_metric_value * 100, 2)
                
                qa_average_metric_value = average_metric_value[::2]
                table_qa = wandb.Table(data= list(zip(range(1, len(qa_average_metric_value) + 1), 
                                                      qa_average_metric_value)), 
                                    columns=["step", "accuracy"])
                line_plot_qa = wandb.plot.line(table_qa, x='step', y='accuracy', title=f'{task}_question')
                wandb.log({f'{task}_q_acc_per_step': line_plot_qa})
                
                tm_average_metric_value = average_metric_value[1::2]
                table_tm = wandb.Table(data= list(zip(range(1, len(tm_average_metric_value) + 1), 
                                                      tm_average_metric_value)), 
                                    columns=["step", "accuracy"])
                line_plot_tm = wandb.plot.line(table_tm, x='step', y='accuracy', 
                                               title=f'{task}_modification')
                wandb.log({f'{task}_tm_acc_per_step': line_plot_tm})
                 
            elif metric_name == "acc_per_step":
                average_metric_value = np.round(average_metric_value * 100, 2)
                table = wandb.Table(data= list(zip(range(1, len(average_metric_value) + 1), average_metric_value)), 
                                    columns=["step", "accuracy"])
                line_plot = wandb.plot.line(table, x='step', y='accuracy', title=f'{task}')
                wandb.log({f'{task}_acc_per_step': line_plot})
            
        else:
            average_metric_value = np.mean(metric_values, axis = 0)
            if "depth" not in metric_name:
                average_metric_value = np.round(average_metric_value * 100, 2)
        logger.info(f"{metric_name}: {average_metric_value}")
        metric[f"{metric_name}"] = str(average_metric_value)
      
    
    
    output_dir = os.path.join(output_dir, dataset, model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    json_object = {}
    for (example_metrics, example) in all_example_metrics:
                example_with_metrics = deepcopy(example)
                for metric_name, metric_value in example_metrics.items():
                    example_with_metrics[f"metric_{metric_name}"] = metric_value
                json_object[example['id']] = example_with_metrics
    
    with xopen(os.path.join(output_dir, f"{task}.json"), "w") as f:
            json.dump(json_object, f, indent=4)
            f.write("\n" + json.dumps(metric) + "\n")
           
    
    result_table = wandb.Table(columns=["model_name"] + list(metric.keys()), 
                                 data=[[shown_model_name] + list(metric.values())]
                                 )
    table_name = f"{task}"

    run.log({table_name: result_table})
   
        
    for k, v in metric.items():
        wandb.log(metric)
    
    wandb.finish()


def preprocess_example(example):
    gold_answers = []
    num_answer = 0
    for i in range(1, 7):
        if f"answer_{i}" not in example:
            break
        gold_answers.append(example[f"answer_{i}"])
        num_answer += 1
        
    if "instruction_answer" in example:
        gold_answers.append(example["instruction_answer"])
        
    model_answers = []
    # response = example['response'].replace("assistant\n\n", "")
    # if smart_json and "{" in example['response'] and "}" in example['response']:
    #     response = "{" + re.findall("{([\s\S]*?)}", example['response'])[0] + "}"
    
    response = example['response'].replace("assistant\n\nHere are the responses to each instruction:\n\n", "")
    
    try:
        data = json.loads(response)
        for k, v in data.items():
            model_answers.append(v)
                    
    except:
        print("Answer not in json format. Parse with instruction number")
              
        if "Instruction_1" in response:
            responses = re.split(r'"Instruction_\d":', response)
            if len(responses) < 2:
                responses = re.split(r'Instruction_\d:', response)
            for r in responses[1:]:
                model_answers.append(r)
        
        else:
           print("Error parsing model answer")
           
    while len(model_answers) < num_answer:
        model_answers.append("")
        
    model_answers = [answer if answer is not None else "" for answer in model_answers]
    return gold_answers, model_answers


def get_metrics_for_example(example, metrics, task):
    gold_answers, model_answers = preprocess_example(example)

    # NOTE: we take everything up to the first newline, since otherwise models could hack
    # the metric by simply copying te input context (as the gold answer is guaranteed
    # to occur in the input context).
    # model_answer = model_answer.split("\n")[0].strip()
    if example['id'] == 20067:
        print("Test")
    example_metrics = {}
    for (metric, metric_name) in metrics:
        example_metrics[metric_name] = metric(predictions=model_answers, ground_truths=gold_answers, task=task)
    
    example["process_answer"] = [normalize_answer_based_on_task(gold_answer, task) 
                                 for gold_answer in gold_answers]
    
    example["process_response"] = [normalize_answer_based_on_task(model_answer, task) 
                                   for model_answer in model_answers]
    
    return (example_metrics, example)


def main(response_dir, output_dir):
    for filename in sorted(os.listdir(response_dir)):
        response_path = os.path.join(response_dir, filename)
        print("running evaluation on", response_path)
        eval_per_file(response_path, output_dir)
    
    print("Finish running all files")

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--response_path", help="Path to data with model response.", 
                        default="/projects/0/gusr0608/generation-with-lms/responses/sif_final/claude-3")
    parser.add_argument("--output_dir", help="Path to output results.", 
                        default="/projects/0/gusr0608/generation-with-lms/results")
    
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(
        args.response_path,
        args.output_dir,
    )
    
    logger.info("finished running %s", sys.argv[0])
    


   