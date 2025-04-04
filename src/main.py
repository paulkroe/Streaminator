# main.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from stream_manager import StreamManager

def main():
    # Load model and tokenizer.
    model_name = "gpt2"  # Use an appropriate model for your use-case.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Set up the stream manager.
    stream_width = 8  # Adjust based on your GPU memory and throughput targets.
    max_length = 50   # Maximum tokens to generate per completion.
    stream_manager = StreamManager(model, tokenizer, stream_width=stream_width, max_length=max_length)

    # Enqueue some example prompts.
    prompts = [
        "Once upon a time",
        "In a galaxy far far away",
        "The quick brown fox",
        "To be or not to be",
        "In the midst of chaos",
    ]
    num_completions = 3  # Generate 3 completions per prompt.
    for prompt in prompts:
        stream_manager.enqueue_prompt(prompt, num_completions)

    # Start the continuous generation loop.
    stream_manager.run_generation_loop()

if __name__ == "__main__":
    main()