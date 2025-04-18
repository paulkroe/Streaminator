# main.py
import os
import re
import time
import json
import random
import torch
import textwrap
from string import Template
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from stream import StreamManager
from tests import GSM8KAnswerChecker
from data import load_random_gsm8k

def main():
    # 1) Load a random sample of GSM8K examples from Hugging Face.
    num_samples = 25
    examples = load_random_gsm8k(num_samples=num_samples, seed=42)

    # 2) Load model and tokenizer.
    model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Update to your actual model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 3) Set up the stream manager with your parameters.
    stream_width = 12
    max_length = 300
    use_kv_cache = True
    continuous_batching = True
    num_completions = 8

    stream_manager = StreamManager(
        model,
        tokenizer,
        stream_width=stream_width,
        max_length=max_length,
        use_kv_cache=use_kv_cache,
        continuous_batching=continuous_batching
    )
    print(f"Stream manager initialized: stream_width={stream_width}, max_length={max_length}")
    print(f"continuous_batching={continuous_batching}, use_kv_cache={use_kv_cache}")

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

    # 8) Convert StreamManager results into a structure for answer-checking.
    #    Each prompt in prompt_map => { "generations": [...], "ground_truth": ... }
    results_for_eval = {}
    for prompt, gold_answer in prompt_map.items():
        results_for_eval[prompt] = {
            "generations": stream_manager.results.get(prompt, []),
            "ground_truth": gold_answer
        }

    # 9) Evaluate the results with GSM8KAnswerChecker.
    evaluation = GSM8KAnswerChecker.eval(results_for_eval)

    # 10) Print out the evaluation for each prompt.
    num_questions = len(evaluation)
    correct_questions = 0
    for entry in evaluation:
        prompt_str = entry["prompt"]
        ground_truth = entry["ground_truth"]
        accuracy = entry["evaluation"]["accuracy"]
        pass_n = entry["evaluation"]["pass@n"]
        match_n = entry["evaluation"]["match@n"]
        if pass_n:
            correct_questions += 1

        print("--------------------------------------------------------")
        print("Prompt:", prompt_str[:100], "...")
        print("Ground Truth:", ground_truth)
        print(f"Accuracy of completions: {accuracy:.2f}")
        print(f"Pass@n = {pass_n}, Match@n = {match_n}")
        print("--------------------------------------------------------")

    overall_pass_rate = correct_questions / num_questions if num_questions > 0 else 0.0
    print(f"\nOverall pass@n rate = {overall_pass_rate:.3f} (i.e., fraction of prompts with at least one correct)")

if __name__ == "__main__":
    main()