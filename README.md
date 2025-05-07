# Streaminator: Multi-Answer Speculative Decoding

Streaminator is a speculative decoding pipeline designed to generate multiple high-quality answers for mathematical problems. It leverages advanced techniques like KV caching, continuous batching, and speculative decoding to optimize performance and accuracy. The project is built to evaluate and improve the performance of language models on datasets like GSM8K.

---

## Features

- **Speculative Decoding**: Generate multiple answers for a single prompt with high efficiency.
- **KV Caching**: Optimize memory usage and speed by reusing key-value caches during decoding.
- **Continuous Batching**: Dynamically batch prompts to maximize GPU utilization.
- **Customizable Pipeline**: Easily configure the pipeline with command-line arguments to enable or disable specific optimizations.
- **Evaluation Metrics**: Measure performance using metrics like `pass@N`, `match@N`, and `correct_generation_fraction`.
- **GPU Profiling**: Log GPU utilization, memory usage, and generation performance.
- **WandB Integration**: Log experiments and metrics to [Weights & Biases](https://wandb.ai/) for visualization and analysis.

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- CUDA-enabled GPU (for optimal performance)
- Required Python libraries (see `requirements.txt`)

### Installation

```
git clone https://github.com/paulkroe/Streaminator.git
cd Streaminator
```

#### 1. Running the Pipeline

The pipeline generates and evaluates answers for mathematical problems using a language model.

```
python3 main.py --mode pipeline --model_name <model_name> --num_samples <num_samples>
```

**Key Arguments**
**--mode**: Choose between pipeline (default) or experiments.

**--model_name**: Name of the Hugging Face model to use (e.g., meta-llama/Llama-3.2-1B-Instruct).

**--num_samples**: Number of examples to sample from the dataset.

**--stream_width**: Number of sequences to process in parallel (default: 16).

**--max_length**: Maximum length of generated sequences (default: 250).

**--num_completions**: Number of completions to generate per prompt (default: 8).

**--use_kv_cache**: Enable KV caching for faster generation.

**--continuous_batching**: Enable continuous batching for better GPU utilization.

**--use_wandb**: Enable logging to WandB.

#### 2. Running the Pipeline
Run a series of experiments with different configurations to evaluate the impact of optimizations.

```
python3 main.py --mode experiments
```

Example Command

```
python3 main.py --mode pipeline --model_name meta-llama/Llama-3.2-1B-Instruct --num_samples 32 --stream_width 1
```

## Development Roadmap
### Short-Term Goals
Ensure the pipeline achieves a pass@8 score of 75% on GSM8K.

Profile GPU performance and optimize memory usage.

### Long-Term Goals
Support additional models via the KV cache wrapper.

Implement a scheduler for prompt batching.

Optimize the prompt sampler for parallelism.

### Key Features and Optimizations
#### 1. KV Caching
Reuse key-value caches during decoding to reduce memory usage and improve speed.

Implemented in kv_cache_manager.py and integrated into StreamManager.

#### 2. Continuous Batching
Dynamically batch prompts to maximize GPU utilization.

Enabled with the --continuous_batching flag.

#### 3. Speculative Decoding
Generate multiple answers for a single prompt using speculative techniques.

Managed by StreamManager and PromptSampler.

#### 4. Evaluation Metrics
pass@N: Fraction of prompts with at least one correct answer in the top N completions.

match@N: Fraction of prompts with at least half of the top N completions correct.

correct_generation_fraction: Fraction of all generated completions that are correct.

#### 5. GPU Profiling
Log GPU utilization, memory usage, and generation performance.

Implemented in logging_utils.py and integrated into StreamManager.

Metrics
Tokens per Second: Measure the speed of token generation.

Pass@8 Score: Target score of 75% on GSM8K with LLaMA 3.2 1B.

## Project Structure

```
Streaminator/
├── main.py               # Entry point for the pipeline and experiments
├── stream/
│   ├── stream_manager.py     # Core logic for speculative decoding
│   ├── prompt_sampler.py     # Logic for sampling tokens during generation
│   ├── kv_cache_manager.py   # Utilities for managing KV caches
│   ├── kv_cache_wrapper.py   # Wrapper for KV cache operations
│   ├── sequence.py           # Sequence management during generation
│   └── __init__.py           # Module initialization
├── utils/
│   ├── data_loader.py        # Functions for loading datasets
│   ├── prompt_utils.py       # Utilities for prompt templates and mapping
│   ├── evaluation.py         # Evaluation logic and metrics
│   ├── logging_utils.py      # Logging utilities (e.g., GPU stats, WandB)
│   └── __init__.py           # Module initialization
├── experiments/
│   ├── experiments.py        # Experiment configurations and execution
│   ├── plotting.py           # Plotting results
│   └── __init__.py           # Module initialization
├── tests/
│   ├── test_stream_manager.py  # Unit tests for StreamManager
│   ├── test_utils.py           # Unit tests for utility functions
│   └── __init__.py             # Module initialization
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── wandb/                  # WandB logs and experiment results
```

## Acknowledgments
Hugging Face Transformers for the model and tokenizer utilities.

Weights & Biases for experiment tracking and visualization.

The GSM8K dataset for providing challenging mathematical problems.