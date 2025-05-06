from data import load_random_gsm8k
from string import Template
import textwrap

def load_examples(num_samples, seed):
    """Load a random sample of GSM8K examples."""
    return load_random_gsm8k(num_samples=num_samples, seed=seed)


def get_prompt_template():
    """Return the prompt template."""
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
    return Template(template_text)


def build_prompt_map(examples, prompt_template):
    """Build a mapping from prompt text to gold answers."""
    prompt_map = {}
    for example in examples:
        question = example["question"]
        gold_answer = example["answer"]
        prompt_text_instance = prompt_template.substitute(question=question)
        prompt_map[prompt_text_instance] = gold_answer
    return prompt_map
