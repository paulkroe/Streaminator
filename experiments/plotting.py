import matplotlib.pyplot as plt

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