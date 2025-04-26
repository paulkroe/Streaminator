# main.py
import os
import re
import time
import json
import random
import torch
import textwrap
import wandb
from string import Template
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from stream import StreamManager
from tests import GSM8KAnswerChecker
from data import load_random_gsm8k
import pynvml
import gc

def main():
    # 1) Load a random sample of GSM8K examples from Hugging Face.
    num_samples = 32
    examples = load_random_gsm8k(num_samples=num_samples, seed=42)

    # 2) Load model and tokenizer.
    model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Update to your actual model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 3) Set up the stream manager with your parameters.
    stream_width = 16
    max_length = 250
    use_kv_cache = True
    continuous_batching = True
    num_completions = 2
    
    # N-gram speculative decoding parameters
    use_ngram_specdec = True
    ngram_n = 3
    speculative_k = 5

    wandb.init(
        project="pipeline-profiling",
        entity="multi-answer-spec-decoding",
        config={
            "model_name": model_name,
            "num_samples": num_samples,
            "stream_width": stream_width,
            "max_length": max_length,
            "num_completions": num_completions,
            "use_kv_cache": use_kv_cache,
            "continuous_batching": continuous_batching,
            "use_ngram_specdec": use_ngram_specdec,
            "ngram_n": ngram_n,
            "speculative_k": speculative_k,
        }
    )

    stream_manager = StreamManager(
        model,
        tokenizer,
        stream_width=stream_width,
        max_length=max_length,
        use_kv_cache=use_kv_cache,
        continuous_batching=continuous_batching,
        use_ngram_specdec=use_ngram_specdec,
        ngram_n=ngram_n,
        speculative_k=speculative_k,
        logger=wandb
    )

    print(f"Stream manager initialized: stream_width={stream_width}, max_length={max_length}")
    print(f"continuous_batching={continuous_batching}, use_kv_cache={use_kv_cache}")
    print(f"N-gram speculative decoding: enabled={use_ngram_specdec}, n={ngram_n}, k={speculative_k}")

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
        stream_manager.enqueue_prompt(prompt, num_completions)

    # 7) Run the generation loop.
    start = time.time()
    stream_manager.run_generation_loop()
    end = time.time()
    print(f"Generation took {end - start:.2f} seconds.")
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
    wandb.log({"avg_completion_length_tokens": avg_completion_length})

    # Log n-gram speculative decoding statistics
    if use_ngram_specdec:
        # Calculate average n-gram model size per prompt
        ngram_model_sizes = []
        for prompt_text, ngram_model in stream_manager.prompt_ngram_models.items():
            # Count total entries in the n-gram counts dictionary
            total_entries = sum(len(contexts) for contexts in ngram_model.counts.values())
            ngram_model_sizes.append(total_entries)
        
        avg_ngram_model_size = sum(ngram_model_sizes) / len(ngram_model_sizes) if ngram_model_sizes else 0
        print(f"Average n-gram model entries per prompt: {avg_ngram_model_size:.2f}")
        
        wandb.log({
            "avg_ngram_model_size": avg_ngram_model_size,
            "ngram_n": ngram_n,
            "speculative_k": speculative_k
        })

    with open("evaluation_results.json", "w") as f:
        json.dump(results_for_eval, f, indent=2)
    print("Evaluation results saved to evaluation_results.json")
    
if __name__ == "__main__":
    main()
    pynvml.nvmlShutdown()

    # Free Memory
    torch.cuda.empty_cache()
    gc.collect()
