import argparse
import dataclasses
import json
import logging
import pathlib
import random
import sys
from copy import deepcopy
import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from xopen import xopen

logger = logging.getLogger(__name__)
random.seed(0)

def input_preprocess(input):
    data = {}
    if "task" in input:
        data["task"] = input["task"]
    elif "context" in input:
         data["task"] = 'In the following, you will be provided with a context and multiple instructions. Please follow the instructions one-by-one and answer the questions without any explanation. Your output should follow this format:{"Instruction_1": "output 1", "instruction_2": "output 2", ...}'
    else:
        data["task"] = 'In the following, you will be provided with multiple instructions. Please follow the instructions one-by-one and answer the questions without any explanation. Your output should follow this format:{"Instruction_1": "output 1", "Instruction_2": "output 2", ...}'
    
    if "context" in input:
        data["context"] = input["context"]
    instructions = []
    for i in range(1, 7):
        if f"instruction_{i}" not in input or \
            len(input[f"instruction_{i}"]) == 0:
                break
        
        instruction_content= input[f"instruction_{i}"]
        
        instructions.append(f"Instruction_{i}. {instruction_content}")
    
    data["instructions"] = "\n".join(instructions)
    return data
        

def main(
    input_dir,
    model_name,
    temperature,
    top_p,
    num_gpus,
    max_new_tokens,
    max_prompt_length,
    hf_cache_path,
    output_dir,
    gpu_memory_utilization = 1
):
    
    logger.info("Loading model")
    model = LLM(
        model=model_name,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
        download_dir=hf_cache_path,
        max_model_len=max_prompt_length,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True
    )

    
    logger.info("Loading tokenizer")
    tokenizer = model.get_tokenizer()
    
    # LLMs are not trained to continue from pad tokens, your input needs to be left-padded.
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token  # to avoid an error
    
    for filename in sorted(os.listdir(input_dir)):
        input_path = os.path.join(input_dir, filename)
        print("Processing file:", input_path)
        dataset_name = input_dir.split("/")[-1]
        if os.path.isfile(input_path):
            output_name = model_name.split("/")[-1] + "_" +  input_path.split("/")[-1].split(".")[0] + ".jsonl"
            output_path = os.path.join(output_dir, dataset_name, model_name.split("/")[-1], output_name)
            run_eval_per_dataset(
                input_path,
                model_name,
                temperature,
                top_p,
                max_new_tokens,
                max_prompt_length,
                output_path,
                tokenizer,
                model,
            )
    
    
def create_prompt(data, format='alpaca'):
    system_prompt = "You are a helpful and honest assistant. Please, respond concisely and truthfully. "
    task = data["task"] + "\n" if "task" in data else ""
    context = "Context:\n" + data["context"] + "\n" if "context" in data else ""
    
    instructions = data["instructions"]
    
    if "llama-3" in format.lower():
        prompt = [{"role": "system", "content": system_prompt}]
        prompt.append({"role": "user", "content": f"{task}{context}{instructions}\n"})
    
    elif "llama-2" in format.lower():
        prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{task}{context}{instructions} [/INST]"
    
    elif "mistral" in format.lower():
        prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{task}{context}{instructions}[/INST]"
    
    else:
        prompt = f"### Instruction:\n{task}{instructions}\n\n### {context}### Response:"
    
    return prompt


def run_eval_per_dataset(
    input_path,
    model_name,
    temperature,
    top_p,
    max_new_tokens,
    max_prompt_length,
    output_path,
    tokenizer,
    model
):
    
    # Create directory for output path if it doesn't exist.
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    examples = []
    prompts = []
    
    logger.info(f"Loaded {len(prompts)} prompts to process")
    
    
    # Fetch all of the prompts
    with xopen(input_path, 'r') as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            
            data = input_preprocess(input_example)
            prompt = create_prompt(data, format = model_name)
            
            if "llama-3" in model_name.lower():
                prompt = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    )
            
            prompt_length = len(tokenizer(prompt)["input_ids"])
            if max_prompt_length < prompt_length:
                logger.info(
                    f"Skipping prompt {prompt[:100]}... with length {prompt_length}, which "
                    f"is greater than maximum prompt length {max_prompt_length}"
                )
                continue
            
            prompts.append(prompt)
            examples.append(deepcopy(input_example))

    
    # Get responses for all of the prompts
    if not torch.cuda.is_available():
        raise ValueError("Unable to find CUDA device with torch. Please use a CUDA device to run this script.")
    
    # sampling_params = ""
    if "llama-3" in model_name.lower():
        sampling_params = SamplingParams(temperature=temperature, 
                                         top_p=top_p, 
                                         max_tokens=max_new_tokens, 
                                         stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
    else:
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
    
    raw_responses = model.generate(prompts, sampling_params)
    responses = [tokenizer.decode(output.outputs[0].token_ids, skip_special_tokens=True).strip() for output in raw_responses]

    with xopen(output_path, "w") as f:
        for example, prompt, response in zip(examples, prompts, responses):
            # output_example = deepcopy(example)
        
            output_example = {}
            output_example["id"] = example["id"]
            output_example["response"] = response
            output_example["prompt"] = prompt
            
            for i in range(1, 7):
                if f"answer_{i}" not in example:
                    break
                output_example[f"answer_{i}"] = example[f"answer_{i}"]
             
            if "instruction_answer" in example:
                output_example["instruction_answer"] = example["instruction_answer"]
            
            f.write(json.dumps(output_example) + "\n")



if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", help="Path to data with questions and documents to use.", 
                        default="../sifo_datasets")
    parser.add_argument(
        "--model",
        help="Model to use in generating responses",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument("--temperature", help="Temperature to use in generation", type=float, default=0.0)
    parser.add_argument("--top-p", help="Top-p to use in generation", type=float, default=1.0)
   
    parser.add_argument("--num-gpus", help="Number of GPUs to use", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", help="gpu_memory_utilization between [0, 1]", type=float, default=0.98)
    parser.add_argument("--hf-cache-path", help="Path to huggingface cache to use.")
    parser.add_argument("--output-dir", help="Path to write output file of generated responses", 
                        default="../results")
    parser.add_argument(
        "--max-new-tokens",
        help="Maximum number of new tokens to generate",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--max-prompt-length",
        help="Maximum number of tokens in the prompt. Longer prompts will be skipped.",
        type=int,
        default=2048,
    )
    
    
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_dir,
        args.model,
        args.temperature,
        args.top_p,
        args.num_gpus,
        args.max_new_tokens,
        args.max_prompt_length,
        args.hf_cache_path,
        args.output_dir,
        args.gpu_memory_utilization
    )
    logger.info("finished running %s", sys.argv[0])
