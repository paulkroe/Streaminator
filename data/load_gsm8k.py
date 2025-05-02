from datasets import load_dataset
import random
import os
import pickle

def load_random_gsm8k(num_samples=100, seed=42, cache_file="gsm8k_cache.pkl"):
    """
    Loads the 'main' split of openai/gsm8k from Hugging Face,
    then randomly selects `num_samples` examples from the train subset.
    If the data is already cached, it loads from the cache instead.
    Each example is a dict with "question" and "answer" keys.
    """
    if os.path.exists(cache_file):
        print(f"Loading cached GSM8K data from {cache_file} ...")
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
        return cached_data

    print("Loading GSM8K from Hugging Face (openai/gsm8k) ...")
    ds = load_dataset("openai/gsm8k", "main", ignore_verifications=True)
    train_ds = ds["train"]
    total_train_size = len(train_ds)
    print(f"GSM8K train set size: {total_train_size}")

    # For reproducibility, set a random seed. Then pick random indices.
    random.seed(seed)
    indices = random.sample(range(total_train_size), num_samples)

    examples = []
    for idx in indices:
        row = train_ds[idx]
        question = row["question"]
        answer = row["answer"]
        examples.append({"question": question, "answer": answer})

    # Cache the data for future use
    print(f"Caching GSM8K data to {cache_file} ...")
    with open(cache_file, "wb") as f:
        pickle.dump(examples, f)

    return examples