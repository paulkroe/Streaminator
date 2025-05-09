# main.py
import os
import re
import time
import json
import random
import torch
import textwrap
import wandb
import argparse
from string import Template
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from stream import StreamManager
from tests import GSM8KAnswerChecker
from data import load_random_gsm8k
from experiments.experiments import run_experiments  # Import the experiments module
import pynvml


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and evaluate math problem solutions")
    parser.add_argument("--mode", type=str, choices=["pipeline", "experiments"], default="pipeline",
                        help="Mode to run: pipeline or experiments")
    parser.add_argument("--num_samples", type=int, default=32, help="Number of GSM8K examples to sample")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                       help="Name of the LLM model to use")
    parser.add_argument("--stream_width", type=int, default=16, help="Stream width for generation")
    parser.add_argument("--max_length", type=int, default=250, help="Maximum length for generation")
    parser.add_argument("--num_completions", type=int, default=2, help="Number of completions to generate")
    parser.add_argument("--no_kv_cache", action="store_false", default=True, help="Whether to use KV cache")
    parser.add_argument("--no_continuous_batching", action="store_false", default=True, help="Whether to use continuous batching")
    parser.add_argument("--no_spec_decoding", action="store_false", default=True, help="Whether to use speculative decoding")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Whether to use wandb")
    parser.add_argument("--ngram_order", type=int, default=3, help="Order of the n-gram model")
    parser.add_argument("--no_prompt_training", action="store_false", default=True, help="Whether to train the NGram on the prompt")
    parser.add_argument("--no_generation_training", action="store_false", default=True, help="Whether to train the NGram on the generations")
    return parser.parse_args()


def run_pipeline(args):
    """
    Run the Streaminator pipeline for generating and evaluating solutions.
    """
    print(f"Running pipeline mode with model: {args.model_name}")

    # Load examples
    examples = load_random_gsm8k(num_samples=args.num_samples, seed=42)

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 3) Set up the stream manager with your parameters.
    if args.use_wandb:
        print("Initializing wandb")
        wandb.init(
            project="pipeline-profiling",
            entity="multi-answer-spec-decoding",
            config={
                "model_name": args.model_name,
                "num_samples": args.num_samples,
                "stream_width": args.stream_width,
                "max_length": args.max_length,
                "num_completions": args.num_completions,
                "kv_cache": args.no_kv_cache,
                "continuous_batching": args.no_continuous_batching,
                "spec_decoding": args.no_spec_decoding,
                "ngram_order": args.ngram_order,
                "no_prompt_training": args.no_prompt_training,
                "no_generation_training": args.no_prompt_training,
                "seed": args.seed,
            }
        )

    stream_manager = StreamManager(
        model=model,
        tokenizer=tokenizer,
        stream_width=args.stream_width,
        max_length=args.max_length,
        use_kv_cache=args.no_kv_cache,
        continuous_batching=args.no_continuous_batching,
        spec_decoding=args.no_spec_decoding,
        no_prompt_training=args.no_prompt_training,
        no_generation_training=args.no_generation_training,
        ngram_order=args.ngram_order,
        logger=wandb if args.use_wandb else None
    )

    print(f"Stream manager initialized: stream_width={args.stream_width}, max_length={args.max_length}")
    print(f"continuous_batching={args.no_continuous_batching}, kv_cache={args.no_kv_cache}, spec_decoding={args.no_spec_decoding}")
    print(f"ngram_order={args.ngram_order}, no_prompt_training={args.no_prompt_training}, no_generation_training={args.no_generation_training}")

    # 4) Define your prompt template.
    template_text = textwrap.dedent("""
           {{#system}}
        You are a renowned mathematician known for your flawless accuracy and clarity. You solve math problems step by step,
        using well-structured logic.
        Always follow this exact response format:
        1. Put your step-by-step calculation process inside <think> tags, explaining each step clearly.
        2. Provide the final answer in a <boxed> tag, using a clear and simplified format.

        Below are two examples. You must never deviate from this format.
        Example 1:
        {{#user}}
        Lucy has 18 apples. She gives 4 apples to her friend. She then doubles the number of apples she has. How many apples does Lucy have left?
        {{#assistant}}
        <think>
        1. Subtract the apples Lucy gave away: 18 - 4 = 14
        2. Double the remaining apples: 14 * 2 = 28
        </think>
        \\boxed{28}

        Example 2:
        {{#user}}
        What is the value of (3 + 5) * 2?
        {{#assistant}}
        <think>
        1. Calculate the expression inside parentheses: 3 + 5 = 8
        2. Multiply the result by 2: 8 × 2 = 16
        </think>
        \\boxed{16}
        {{#user}}
        Now solve the following problem:
        $question
        {{#assistant}}
        """)
    prompt_template = Template(template_text)

    # 5) Build prompts and record mapping from prompt_text -> gold answer.
    prompt_map = {}
    for example in examples:
        question = example["question"]
        gold_answer = example["answer"]
        prompt_text_instance = prompt_template.substitute(question=question)
        prompt_map[prompt_text_instance] = gold_answer

    # 6) Enqueue all prompts with desired completions.
    for prompt in prompt_map.keys():
        stream_manager.enqueue_prompt(prompt, args.num_completions)

    # Run the generation loop
    stream_manager.run_generation_loop()

    # Evaluate results
    results_for_eval = {
        prompt: {
            "generations": stream_manager.results.get(prompt, []),
            "ground_truth": answer
        }
        for prompt, answer in prompt_map.items()
    }
    evaluation = GSM8KAnswerChecker.eval(results_for_eval)

    # Print evaluation metrics
    print(f"Evaluation results: {evaluation}")


def main():
    """
    Main entry point for the Streaminator pipeline or experiments.
    """
    args = parse_args()

    if args.mode == "experiments":
        print("Running experiments mode...")
        run_experiments()
    else:
        print("Running pipeline mode...")
        run_pipeline(args)


if __name__ == "__main__":
    main()