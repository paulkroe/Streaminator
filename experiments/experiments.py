import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from stream import StreamManager
from tests import GSM8KAnswerChecker
from utils.data_loader import load_random_gsm8k
from utils.prompt_utils import get_prompt_template, build_prompt_map  # Reuse functions from main.py
from .plotting import plot_results
from utils.logging_utils import Logger

def run_experiment(config):
    logger = Logger(enable_wandb=True, debug=False)

    # Load examples and prompt template
    examples = load_random_gsm8k(num_samples=config["num_prompts"], seed=42)
    prompt_template = get_prompt_template()
    prompt_map = build_prompt_map(examples, prompt_template)

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype=torch.float16)
    model.to(device)
    model.eval()

    # Initialize the StreamManager with the given configuration
    stream_manager = StreamManager(
        model=model,
        tokenizer=tokenizer,
        stream_width=config["stream_width"],
        max_length=config["max_length"],
        use_kv_cache=config["use_kv_cache"],
        continuous_batching=config["continuous_batching"],
        logger=logger
    )

    # Enqueue prompts
    for prompt in prompt_map.keys():
        stream_manager.enqueue_prompt(prompt, config["num_completions"])

    # Start timing
    start_time = time.time()
    stream_manager.run_generation_loop()
    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    throughput = (config["num_prompts"] * config["num_completions"]) / total_time
    total_generated_tokens = sum(
        len(tokenizer.encode(gen, add_special_tokens=False))
        for completions in stream_manager.results.values()
        for gen in completions
    )
    tokens_per_second = total_generated_tokens / total_time

    # Evaluate results
    results_for_eval = {
        prompt: {
            "generations": stream_manager.results.get(prompt, []),
            "ground_truth": prompt_map[prompt],  # Use ground truth from prompt map
        }
        for prompt in prompt_map.keys()
    }
    evaluation = GSM8KAnswerChecker.eval(results_for_eval)

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

    overall_pass_n = float(num_pass_n) / float(num_questions) if num_questions else 0.0
    overall_match_n = float(num_match_n) / float(num_questions) if num_questions else 0.0
    overall_correct_fraction = float(num_correct_generations) / float(num_total_generations) if num_total_generations else 0.0
    avg_completion_length = (
        float(total_generated_tokens) / float(num_total_generations) if num_total_generations else 0.0
    )

    print(f"Config: {config}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Throughput: {throughput:.2f} sequences/s")
    print(f"Tokens per Second: {tokens_per_second:.2f}")
    print(f"Overall pass@n rate: {overall_pass_n:.3f}")
    print(f"Overall match@n rate: {overall_match_n:.3f}")
    print(f"Correct generations fraction: {overall_correct_fraction:.3f}")
    print(f"Average completion length (in tokens): {avg_completion_length:.2f}")

    # Log results
    logger.log_metrics({"experiment_config": config})

    # Shutdown logger
    logger.shutdown()

    return {
        "config": config,
        "throughput": throughput,
        "tokens_per_second": tokens_per_second,
        "overall_pass_n": overall_pass_n,
        "overall_match_n": overall_match_n,
        "overall_correct_fraction": overall_correct_fraction,
        "avg_completion_length": avg_completion_length,
    }


def run_experiments():
    # Define experiment configurations
    experiments = [
        {"stream_width": 8, "max_length": 250, "use_kv_cache": False, "continuous_batching": False, "num_prompts": 10, "num_completions": 8, "model_name": "meta-llama/Llama-3.2-1B-Instruct"},
        {"stream_width": 8, "max_length": 250, "use_kv_cache": True, "continuous_batching": False, "num_prompts": 10, "num_completions": 8, "model_name": "meta-llama/Llama-3.2-1B-Instruct"},
        {"stream_width": 8, "max_length": 250, "use_kv_cache": False, "continuous_batching": True, "num_prompts": 10, "num_completions": 8, "model_name": "meta-llama/Llama-3.2-1B-Instruct"},
        {"stream_width": 8, "max_length": 250, "use_kv_cache": True, "continuous_batching": True, "num_prompts": 10, "num_completions": 8, "model_name": "meta-llama/Llama-3.2-1B-Instruct"},
    ]

    results = []
    for config in experiments:
        result = run_experiment(config)
        results.append(result)

    # Plot results
    plot_results(results)

