# main.py
import os
import re
import time
import json
import random
import torch
import wandb
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from stream import StreamManager
from tests import GSM8KAnswerChecker

from utils import load_examples, get_prompt_template, build_prompt_map
from experiments import run_experiments
import pynvml
import gc

def parse_args():
    parser = argparse.ArgumentParser(description="Generate and evaluate math problem solutions or run experiments.")
    parser.add_argument("--mode", choices=["pipeline", "experiments"], default="pipeline",
                        help="Choose whether to run the main pipeline or experiments.")
    parser.add_argument("--num_samples", type=int, default=32, help="Number of GSM8K examples to sample")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                       help="Name of the LLM model to use")
    parser.add_argument("--stream_width", type=int, default=16, help="Stream width for generation")
    parser.add_argument("--max_length", type=int, default=250, help="Maximum length for generation")
    parser.add_argument("--num_completions", type=int, default=8, help="Number of completions to generate")
    parser.add_argument("--use_kv_cache", action="store_true", default=True, help="Whether to use KV cache")
    parser.add_argument("--continuous_batching", action="store_true", default=True, help="Whether to use continuous batching")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", default=True, help="Whether to use wandb")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()

    # Check if the user wants to run experiments
    if args.mode == "experiments":
        run_experiments()
        return

    # Load examples
    examples = load_examples(num_samples=args.num_samples, seed=args.seed)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Initialize StreamManager
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
                "use_kv_cache": args.use_kv_cache,
                "continuous_batching": args.continuous_batching,
                "seed": args.seed,
            }
        )

    stream_manager = StreamManager(
        model,
        tokenizer,
        stream_width=args.stream_width,
        max_length=args.max_length,
        use_kv_cache=args.use_kv_cache,
        continuous_batching=args.continuous_batching,
        logger=wandb if args.use_wandb else None
    )

    print(f"Stream manager initialized: stream_width={args.stream_width}, max_length={args.max_length}")
    print(f"continuous_batching={args.continuous_batching}, use_kv_cache={args.use_kv_cache}")

    # Get prompt template and build prompt map
    prompt_template = get_prompt_template()
    prompt_map = build_prompt_map(examples, prompt_template)

    # Enqueue prompts
    for prompt in prompt_map.keys():
        stream_manager.enqueue_prompt(prompt, args.num_completions)

    # Run the generation loop
    start = time.time()
    stream_manager.run_generation_loop()
    end = time.time()
    print(f"Generation took {end - start:.2f} seconds.")
    if args.use_wandb:
        wandb.log({"total_generation_time_sec": end - start})

       # Count total number of generated tokens (excluding prompt)
    total_generated_tokens = sum(
        len(tokenizer.encode(gen, add_special_tokens=False))
        for completions in stream_manager.results.values()
        for gen in completions
    )

    tokens_per_sec = total_generated_tokens / (end - start)
    print(f"Total generated tokens: {total_generated_tokens}")
    print(f"Tokens per second: {tokens_per_sec:.2f}")
    if args.use_wandb:
        wandb.log({"total_generated_tokens": total_generated_tokens, "tokens_per_second": tokens_per_sec})

    # 8) Convert StreamManager results into a structure for answer-checking.
    results_for_eval = {
       prompt: {
           "generations": stream_manager.results.get(prompt, []),
            "ground_truth": gold_answer
        }
        for prompt, gold_answer in prompt_map.items()
   }

    # Evaluate results and log metrics (same as before)
    # 9) Evaluate the results with GSM8KAnswerChecker.
    evaluation = GSM8KAnswerChecker.eval(results_for_eval)

    # 10) Compute overall stats
    num_questions = len(evaluation)
    num_pass_n = 0
    num_match_n = 0
    num_total_generations = 0
    num_correct_generations = 0

    for entry in evaluation:
        evaluated_answers = entry["answers"]
        eval_metrics = entry["evaluation"]
        generations = [a["text"] for a in evaluated_answers]
        if eval_metrics["pass@n"]:
            num_pass_n += 1
        if eval_metrics["match@n"]:
            num_match_n += 1
        num_total_generations += len(generations)
        num_correct_generations += sum(
            answer["answer_eval"]["correct"] for answer in evaluated_answers
        )

    overall_pass_n = num_pass_n / num_questions if num_questions else 0.0
    overall_match_n = num_match_n / num_questions if num_questions else 0.0
    overall_correct_fraction = num_correct_generations / num_total_generations if num_total_generations else 0.0

    print(f"\nOverall pass@n rate:     {overall_pass_n:.3f}")
    print(f"Overall match@n rate:    {overall_match_n:.3f}")
    print(f"Correct generations:     {overall_correct_fraction:.3f} of all completions")

    if args.use_wandb:
        wandb.log({
            "overall_pass@n_rate": overall_pass_n,
            "overall_match@n_rate": overall_match_n,
            "overall_correct_fraction": overall_correct_fraction
        })

    # 11) Compute and log average completion length
    all_completion_lengths = [
        len(tokenizer.encode(g, add_special_tokens=False))
        for completions in stream_manager.results.values()
        for g in completions
    ]
    avg_completion_length = (
        sum(all_completion_lengths) / len(all_completion_lengths)
        if all_completion_lengths else 0.0
    )
    print(f"Average completion length (in tokens): {avg_completion_length:.2f}")
    if args.use_wandb:
        wandb.log({"avg_completion_length_tokens": avg_completion_length})

    with open("evaluation_results.json", "w") as f:
        json.dump(results_for_eval, f, indent=2)
    print("Evaluation results saved to evaluation_results.json")


if __name__ == "__main__":
    main()
    pynvml.nvmlShutdown()

    # Free Memory
    torch.cuda.empty_cache()
    gc.collect()
