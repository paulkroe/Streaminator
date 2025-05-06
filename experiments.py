import time
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from stream import StreamManager
from tests import GSM8KAnswerChecker
from utils import load_examples, get_prompt_template, build_prompt_map  # Reuse functions from main.py


def run_experiment(config):
    # Load examples and prompt template
    examples = load_examples(num_samples=config["num_prompts"], seed=42)
    prompt_template = get_prompt_template()
    prompt_map = build_prompt_map(examples, prompt_template)

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Replace with your actual model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
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
        debug=False,
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
        {"stream_width": 8, "max_length": 250, "use_kv_cache": False, "continuous_batching": False, "num_prompts": 10, "num_completions": 8},
        {"stream_width": 8, "max_length": 250, "use_kv_cache": True, "continuous_batching": False, "num_prompts": 10, "num_completions": 8},
        {"stream_width": 8, "max_length": 250, "use_kv_cache": False, "continuous_batching": True, "num_prompts": 10, "num_completions": 8},
        {"stream_width": 8, "max_length": 250, "use_kv_cache": True, "continuous_batching": True, "num_prompts": 10, "num_completions": 8},
    ]

    results = []
    for config in experiments:
        result = run_experiment(config)
        results.append(result)

    # Plot results
    plot_results(results)


def plot_results(results):
    # Extract data for plotting
    configs = [str(res["config"]) for res in results]
    throughput = [res["throughput"] for res in results]
    tokens_per_second = [res["tokens_per_second"] for res in results]
    overall_pass_n = [res["overall_pass_n"] for res in results]
    overall_match_n = [res["overall_match_n"] for res in results]
    overall_correct_fraction = [res["overall_correct_fraction"] for res in results]
    avg_completion_length = [res["avg_completion_length"] for res in results]

       # Map configurations to simplified labels
    xtick_labels = []
    for res in results:
        use_kv_cache = res["config"]["use_kv_cache"]
        continuous_batching = res["config"]["continuous_batching"]
        if use_kv_cache and continuous_batching:
            xtick_labels.append("both")
        elif use_kv_cache:
            xtick_labels.append("kv caching")
        elif continuous_batching:
            xtick_labels.append("continuous batching")
        else:
            xtick_labels.append("neither")

    # Create subplots
    fig, axs = plt.subplots(3, 2, figsize=(18, 14), constrained_layout=True)
    fig.suptitle("Experiment Results: stream_width=8, max_length=250, num_prompts=10, num_completions=8")

    # Plot throughput
    axs[0, 0].bar(range(len(configs)), throughput, color="blue")
    axs[0, 0].set_title("Throughput (sequences/s)", fontsize=12)
    axs[0, 0].set_xticks(range(len(configs)))
    axs[0, 0].set_xticklabels(xtick_labels, rotation=30, ha="right", fontsize=10)

    # Plot tokens per second
    axs[0, 1].bar(range(len(configs)), tokens_per_second, color="green")
    axs[0, 1].set_title("Tokens per Second", fontsize=12)
    axs[0, 1].set_xticks(range(len(configs)))
    axs[0, 1].set_xticklabels(xtick_labels, rotation=30, ha="right", fontsize=10)

    # Plot pass@n rate
    axs[1, 0].bar(range(len(configs)), overall_pass_n, color="orange")
    axs[1, 0].set_title("Pass@N Rate", fontsize=12)
    axs[1, 0].set_xticks(range(len(configs)))
    axs[1, 0].set_xticklabels(xtick_labels, rotation=30, ha="right", fontsize=10)

    # Plot match@n rate
    axs[1, 1].bar(range(len(configs)), overall_match_n, color="red")
    axs[1, 1].set_title("Match@N Rate", fontsize=12)
    axs[1, 1].set_xticks(range(len(configs)))
    axs[1, 1].set_xticklabels(xtick_labels, rotation=30, ha="right", fontsize=10)

    # Plot correct generations fraction
    axs[2, 0].bar(range(len(configs)), overall_correct_fraction, color="purple")
    axs[2, 0].set_title("Correct Generations Fraction", fontsize=12)
    axs[2, 0].set_xticks(range(len(configs)))
    axs[2, 0].set_xticklabels(xtick_labels, rotation=30, ha="right", fontsize=10)

    # Plot average completion length
    axs[2, 1].bar(range(len(configs)), avg_completion_length, color="cyan")
    axs[2, 1].set_title("Average Completion Length (tokens)", fontsize=12)
    axs[2, 1].set_xticks(range(len(configs)))
    axs[2, 1].set_xticklabels(xtick_labels, rotation=30, ha="right", fontsize=10)

    # Adjust layout and spacing
    # plt.subplots_adjust(hspace=0.5, wspace=0.4)  # Adjust spacing between subplots
    # plt.tight_layout(rect=[0, 0, 1, 0.95])  # Apply tight layout with adjusted margins
    # plt.show()  # Show the plot
    plt.savefig("experiment_results.png", dpi=300)  # Save the figure as a PNG file
    plt.close(fig)  # Close the figure to free up memory