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
import pynvml
import gc

def parse_args():
    parser = argparse.ArgumentParser(description="Generate and evaluate math problem solutions")
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
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # 1) Load a random sample of GSM8K examples from Hugging Face.
    examples = load_random_gsm8k(num_samples=args.num_samples, seed=args.seed)

    # 2) Load model and tokenizer.
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
                "seed": args.seed,
            }
        )

    stream_manager = StreamManager(
        model,
        tokenizer,
        stream_width=args.stream_width,
        max_length=args.max_length,
        use_kv_cache=args.no_kv_cache,
        continuous_batching=args.no_continuous_batching,
        spec_decoding=args.no_spec_decoding,
        logger=wandb if args.use_wandb else None
    )

    print(f"Stream manager initialized: stream_width={args.stream_width}, max_length={args.max_length}")
    print(f"continuous_batching={args.no_continuous_batching}, kv_cache={args.no_kv_cache}, spec_decoding={args.no_spec_decoding}")

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

    # 7) Run the generation loop.
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
        metrics = {
            "overall_pass@n_rate": overall_pass_n,
            "overall_match@n_rate": overall_match_n,
            "overall_correct_fraction": overall_correct_fraction,
        }

        if args.no_spec_decoding:
            token_acc = {
                stream_manager.tokenizer.decode([tok_id]): count
                for tok_id, count in stream_manager.acceptance_dict.items()
            }
            token_table = wandb.Table(
                columns=["token", "accepted_count"],
                data=list(token_acc.items())
            )
            wandb.log({"token_accuracy_table": token_table})

            level_metrics = {
                f"level_{level}_acceptance": (
                    stream_manager.completion_level_acceptance[level]
                    / stream_manager.completion_level_count[level]
                )
                for level in stream_manager.completion_level_acceptance.keys()
            }
            metrics.update(level_metrics)

        metrics.update({"avg_completion_length": avg_completion_length})
        wandb.log(metrics)

    with open("evaluation_results.json", "w") as f:
        json.dump(results_for_eval, f, indent=2)
    print("Evaluation results saved to evaluation_results.json")
    
if __name__ == "__main__":
    main()
    pynvml.nvmlShutdown()

    # Free Memory
    torch.cuda.empty_cache()
    gc.collect()