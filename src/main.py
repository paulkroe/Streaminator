# main.py
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from stream_manager import StreamManager
import textwrap
from string import Template

def main():
    start = time.time()
    # Load model and tokenizer.
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Set up the stream manager.
    stream_width = 8  # Adjust based on your GPU memory and throughput targets.
    max_length = 250   # Maximum tokens to generate per completion.
    stream_manager = StreamManager(model,
        tokenizer,
        stream_width=stream_width,
        max_length=max_length,
        use_kv_cache=True,
        continuous_batching=True
    )


    template_text = textwrap.dedent("""
        {{#system}}
        You are a helpful assistant solving math problems. You solve problems step by step using the following format:
        1. Put your step-by-step solution inside <think> tags, explaining each step clearly.
        2. Verify your final answer whenever possible.
        3. Provide the final answer in a <boxed> tag in a simplified and clear format.

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
        1. Add 3 and 5 to get 8.
        2. Multiply the result by 2: 8 * 2 = 16
        </think>
        \\boxed{16}
        {{#user}}
        $question
        {{#assistant}}
        """
    )
    prompt_template = Template(template_text)
    # Enqueue some example prompts.
    prompts = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
        "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?"
        # "What is the capital of France?",
        # "What is the square root of 144?",
        # "What is the name of the Ocean between Chinea and the USA?"
    ]
    for idx, p in enumerate(prompts):
        prompts[idx] = prompt_template.substitute(question=p)

    num_completions = 3  # Generate 3 completions per prompt.
    for prompt in prompts:
        stream_manager.enqueue_prompt(prompt, num_completions)

    # Start the continuous generation loop.
    stream_manager.run_generation_loop()
    end = time.time()
    print(end-start)   

if __name__ == "__main__":
    main()