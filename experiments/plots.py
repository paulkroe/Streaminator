import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
plt.style.use('ggplot')

def create_profiling_plot(data):
    print('creating profiling plot')
    # Filter for the components we want to compare
    components = ['ngram_call', 'generate_step_with_kv', '_accept_speculative', '_refill_active_seqs']
    filtered_data = data[data['component'].isin(components)]
    
    # Create mapping for prettier names
    name_mapping = {
        'ngram_call': 'Ngram Forward',
        'generate_step_with_kv': 'Base Model Forward',
        '_accept_speculative': 'Accept Speculation',
        '_refill_active_seqs': 'Refill Batches'
    }
    
    # Sort by total time for better visualization
    filtered_data = filtered_data.sort_values('fraction_of_total_time')
    
    # Create the horizontal bar plot
    plt.figure(figsize=(20, 8))
    # Use a softer blue color palette
    colors = ['#8172B3']
    bars = plt.barh(filtered_data['component'].map(name_mapping), 
                   filtered_data['fraction_of_total_time'],
                   color=colors)
    
    # Add value labels on the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{(width* 100):.2f}%', 
                ha='left', va='center', fontsize=10)
    
    plt.xlabel('Fraction of Total Time (%)')
    plt.title('Profiling Comparison for different components')
    plt.tight_layout()

def create_gpu_plots(gpu_mem_data, stream_data):
    # Create figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    
    # Get colors from Set1 colormap
    colors = plt.cm.Set1.colors
    
    # Plot GPU Memory Usage
    ax1.plot(gpu_mem_data['Step'], 
             gpu_mem_data['no-cont-batching - gpu memory usage (%)'],
             label='No Continuous Batching',
             color=colors[0])
    ax1.plot(gpu_mem_data['Step'],
             gpu_mem_data['cont-batching - gpu memory usage (%)'],
             label='Continuous Batching',
             color=colors[1])
    
    ax1.set_xlabel('Step')
    ax1.set_ylabel('GPU Memory Usage (%)')
    ax1.set_title('GPU Memory Usage Comparison')
    ax1.legend()
    
    # Plot Stream Utilization
    ax2.plot(stream_data['Step'], 
             stream_data['no-cont-batching - stream utilization (%)'],
             label='No Continuous Batching',
             color=colors[0])
    ax2.plot(stream_data['Step'],
             stream_data['cont-batching - stream utilization (%)'],
             label='Continuous Batching',
             color=colors[1])
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Stream Utilization (%)')
    ax2.set_title('Stream Utilization Comparison')
    ax2.legend()
    
    plt.tight_layout()

def create_levels_plot(data_list):
    prompt_gen = []
    prompt_gen = []
    prompt = []
    gen = []
    
    for lv, df in enumerate(data_list):
        # Get the last non-empty value for each metric
        prompt_gen.append(df[f'prompt generation - level_{lv}_acceptance'].dropna().iloc[-1] if not df[f'prompt generation - level_{lv}_acceptance'].dropna().empty else 0)
        prompt.append(df[f'prompt - level_{lv}_acceptance'].dropna().iloc[-1] if not df[f'prompt - level_{lv}_acceptance'].dropna().empty else 0)
        if lv == 0:
            gen.append(0)
        else:
            gen.append(df[f'generation - level_{lv}_acceptance'].dropna().iloc[-1] if not df[f'generation - level_{lv}_acceptance'].dropna().empty else 0)
    
    # Create figure and axis
    plt.figure(figsize=(20, 8))
    
    # Set up the x positions for the bars
    x = np.arange(len(data_list))  # Create an array of indices for the levels
    width = 0.25  # Width of each bar
    
    # Create arrays for bar positions and heights
    prompt_x = []
    prompt_heights = []
    gen_x = []
    gen_heights = []
    prompt_gen_x = []
    prompt_gen_heights = []
    
    # Prepare data for plotting
    for i in range(len(data_list)):
        if i == 0:  # First generation
            prompt_x.append(x[i] - width/2)
            prompt_heights.append(prompt[i])
            prompt_gen_x.append(x[i] + width/2)
            prompt_gen_heights.append(prompt_gen[i])
        else:
            prompt_x.append(x[i] - width)
            prompt_heights.append(prompt[i])
            gen_x.append(x[i])
            gen_heights.append(gen[i])
            prompt_gen_x.append(x[i] + width)
            prompt_gen_heights.append(prompt_gen[i])
    
    # Plot the bars with consistent colors
    plt.bar(prompt_x, prompt_heights, width, label='Ngram trained on Prompt')
    if len(gen_x) > 0:  # Only plot gen bars if there are any
        plt.bar(gen_x, gen_heights, width, label='Ngram trained on Generation')
    plt.bar(prompt_gen_x, prompt_gen_heights, width, label='Ngram trained on Prompt & Generation')
    
    # Customize the plot
    plt.xlabel('Acceptance Rate')
    plt.ylabel('Generation Number')
    plt.title('Ngram Acceptance Across Across Generations')
    plt.xticks(x, [f'Generation {lv+1}' for lv in range(len(data_list))])
    plt.legend()
    plt.tight_layout()

def accepted_tokens_distribution_plot(data):
    data = data.sort_values(by='accepted_count', ascending=False)
    data = data.head(50)
    # Replace tokens that are exactly a single space with "<blank>"
    data['token'] = data['token'].apply(lambda x: "<blank>" if x == " " else x)
    X = data['token']
    Y = data['accepted_count']
    plt.figure(figsize=(20, 8))  # Increased width to accommodate vertical labels
    plt.bar(X, Y)
    plt.title('Accepted Token Counts')
    plt.xlabel('Token')
    plt.ylabel('Count')
    plt.xticks(rotation=90)  # Rotate x-axis labels vertically
    plt.tight_layout()  # Adjust layout to prevent label cutoff

if __name__ == "__main__":
    # Load the data
    data = pd.read_csv("data/zipfs.csv")

    # Plot the data
    accepted_tokens_distribution_plot(data)
    plt.savefig("plots/accepted_tokens_distribution.png")

    lv_data = []
    for i in range(8):
        lv_data.append(pd.read_csv(f"data/lv{i}.csv"))
    
    create_levels_plot(lv_data)
    plt.savefig("plots/levels.png")

    profiling_data = pd.read_csv("data/profiling.csv")
    create_profiling_plot(profiling_data)
    plt.savefig("plots/profiling.png")

    stream_usage_data = pd.read_csv("data/stream_util.csv")
    gpu_usage_data = pd.read_csv("data/gpu_memory.csv")
    create_gpu_plots(gpu_usage_data, stream_usage_data)
    plt.savefig("plots/gpu_metrics.png")
