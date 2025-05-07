# main.py
import os
import time
import torch
import wandb
import argparse
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from stream import StreamManager
from utils.data_loader import load_random_gsm8k
from utils.prompt_utils import get_prompt_template, build_prompt_map
from utils.evaluation import GSM8KAnswerChecker
from experiments.experiments import run_experiments
from utils.logging_utils import Logger


def parse_args():
    """
    Parse command-line arguments for the Streaminator pipeline or experiments.
    """
    parser = argparse.ArgumentParser(description="Generate and evaluate math problem solutions or run experiments.")
    parser.add_argument("--mode", choices=["pipeline", "experiments"], default="pipeline", help="Run mode.")
    parser.add_argument("--num_samples", type=int, default=32, help="Number of examples to sample.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name.")
    parser.add_argument("--stream_width", type=int, default=16, help="Stream width for generation.")
    parser.add_argument("--max_length", type=int, default=250, help="Maximum length for generation.")
    parser.add_argument("--num_completions", type=int, default=8, help="Number of completions to generate.")
    parser.add_argument("--use_kv_cache", action="store_true", default=True, help="Enable KV cache.")
    parser.add_argument("--continuous_batching", action="store_true", default=True, help="Enable continuous batching.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_wandb", action="store_true", default=True, help="Enable WandB logging.")
    return parser.parse_args()


def initialize_model_and_tokenizer(model_name):
    """
    Initialize the model and tokenizer.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        tuple: The loaded model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer


def initialize_wandb(args):
    """
    Initialize WandB logging if enabled.

    Args:
        args (Namespace): Parsed command-line arguments.
    """
    if args.use_wandb:
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


def run_pipeline(args):
    """
    Run the main pipeline for generating and evaluating solutions.

    Args:
        args (Namespace): Parsed command-line arguments.
    """
    # Load examples
    examples = load_random_gsm8k(num_samples=args.num_samples, seed=args.seed)

    # Initialize model and tokenizer
    model, tokenizer = initialize_model_and_tokenizer(args.model_name)

    # Initialize WandB
    initialize_wandb(args)

    # Initialize Logger
    logger = Logger(enable_wandb=args.use_wandb, debug=False)

    # Initialize StreamManager
    stream_manager = StreamManager(
        model=model,
        tokenizer=tokenizer,
        stream_width=args.stream_width,
        max_length=args.max_length,
        use_kv_cache=args.use_kv_cache,
        continuous_batching=args.continuous_batching,
        logger=logger  # Pass the Logger instance
    )

    # Prepare prompts
    prompt_template = get_prompt_template()
    prompt_map = build_prompt_map(examples, prompt_template)

    # Enqueue prompts
    for prompt in prompt_map.keys():
        stream_manager.enqueue_prompt(prompt, args.num_completions)

    # Run the generation loop
    start_time = time.time()
    stream_manager.run_generation_loop()
    end_time = time.time()

    # Log generation stats
    total_time = end_time - start_time
    print(f"Generation took {total_time:.2f} seconds.")
    if args.use_wandb:
        wandb.log({"total_generation_time_sec": total_time})

    # Evaluate results
    evaluate_results(stream_manager, prompt_map, tokenizer, total_time, args)


def evaluate_results(stream_manager, prompt_map, tokenizer, total_time, args):
    """
    Evaluate the generated results and log metrics.

    Args:
        stream_manager (StreamManager): The StreamManager instance.
        prompt_map (dict): Mapping of prompts to ground truth answers.
        tokenizer (AutoTokenizer): The tokenizer used for encoding.
        total_time (float): Total time taken for generation.
        args (Namespace): Parsed command-line arguments.
    """
    # Convert results for evaluation
    results_for_eval = {
        prompt: {
            "generations": stream_manager.results.get(prompt, []),
            "ground_truth": gold_answer
        }
        for prompt, gold_answer in prompt_map.items()
    }

    # Evaluate results
    evaluation = GSM8KAnswerChecker.eval(results_for_eval)

    # Compute overall stats
    num_questions = len(evaluation)
    num_pass_n = sum(1 for entry in evaluation if entry["evaluation"]["pass@n"])
    num_match_n = sum(1 for entry in evaluation if entry["evaluation"]["match@n"])
    num_total_generations = sum(len(entry["answers"]) for entry in evaluation)
    num_correct_generations = sum(
        sum(answer["answer_eval"]["correct"] for answer in entry["answers"])
        for entry in evaluation
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

    # Compute and log average completion length
    all_completion_lengths = [
        len(tokenizer.encode(g, add_special_tokens=False))
        for completions in stream_manager.results.values()
        for g in completions
    ]
    avg_completion_length = sum(all_completion_lengths) / len(all_completion_lengths) if all_completion_lengths else 0.0
    print(f"Average completion length: {avg_completion_length:.2f} tokens.")
    if args.use_wandb:
        wandb.log({"avg_completion_length": avg_completion_length})


def main():
    """
    Main entry point for the Streaminator pipeline or experiments.
    """
    args = parse_args()

    if args.mode == "experiments":
        run_experiments()
    else:
        run_pipeline(args)

    # Clean up resources
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
