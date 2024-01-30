import os
import numpy as np
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from datasets import load_from_disk
from matplotlib import pyplot as plt
from utils import *

# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
from peft import PeftModel  # For some reason this import must be after os.environ['CUDA_VISIBLE_DEVICES']


def eval_model(model, tokenizer, dataset, base_model=None):
    model = model.eval()

    generated_texts, reference_texts = [], []
    for feature in tqdm(dataset):

        prompt = feature['prompt']
        gt = feature['answer']

        inputs = tokenizer(
            prompt, return_tensors='pt',
            padding=False, max_length=4096
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        
        res = model.generate(
            **inputs, 
            use_cache=True
        )
        output = tokenizer.decode(res[0], skip_special_tokens=True)
        answer = re.sub(r'.*\[/INST\]\s*', '', output, flags=re.DOTALL)

        generated_texts.append(answer)
        reference_texts.append(gt)

        # print("GENERATED: ", answer)
        # print("REFERENCE: ", gt)

    metrics = calc_metrics(reference_texts, generated_texts)
    return metrics, generated_texts


def main(args):

    print("Arguments: ")
    print('-' * 55)
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print('-' * 55)

    # Load Model and Tokenizer
    model_name = parse_model_name(args.base_model_name, from_remote=True) 
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,   
    )
    base_model.model_parellal = True

    if args.eval_model == 'base':
        model = base_model
    elif args.eval_model == 'peft':
        model = PeftModel.from_pretrained(
            base_model,
            args.model_path
            )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load dataset
    dataset = load_from_disk(args.dataset_path)

    # Evaluate model
    metrics, generated_texts = eval_model(model, tokenizer, dataset['test'])

    # Print results
    print("Results: ")
    print(metrics)

    # Save results
    if args.store_results:
        df = dataset['test'].to_pandas()
        df['generated'] = generated_texts
        df.to_csv('./results/' + f"{args.eval_model}_{args.dataset_path.split('/')[-1]}.csv", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default="llama2")
    parser.add_argument("--model_path", type=str, default="FinGPT/fingpt-forecaster_dow30_llama2-7b_lora")
    parser.add_argument("--dataset_path", type=str, default="data/fingpt-forecaster-crypto-20230131-20231231-1-4-08")
    parser.add_argument("--eval_model", type=str, choices=['base', 'peft'], default="peft")
    parser.add_argument("--store_results", type=bool, default=True)
    args = parser.parse_args()

    main(args)

