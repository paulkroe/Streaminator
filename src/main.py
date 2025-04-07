# main.py
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from stream_manager import StreamManager

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
    max_length = 50   # Maximum tokens to generate per completion.
    stream_manager = StreamManager(model,
        tokenizer,
        stream_width=stream_width,
        max_length=max_length,
        use_kv_cache=True,
        continuous_batching=False
    )

    # Enqueue some example prompts.
    prompts = [
        "Once upon a time",
        "In a galaxy far far away",
        "The quick brown fox",
        "To be or not to be",
        "In the midst of chaos",
        "Answer the following question with one word: What is the capital of Germany?",
        "The capital of France is ",
        "The capital of China is ",
        "The capital of Ukraine is "
    ]
    num_completions = 3  # Generate 3 completions per prompt.
    for prompt in prompts:
        stream_manager.enqueue_prompt(prompt, num_completions)

    # Start the continuous generation loop.
    stream_manager.run_generation_loop()
    end = time.time()
    print(end-start)   

if __name__ == "__main__":
    main()