import random
import os
import pickle
from datasets import load_dataset

def load_random_gsm8k(num_samples=100, seed=42, cache_file="gsm8k_cache.pkl"):
    """
    Load a random sample of GSM8K examples. If cached data exists, use it.
    Otherwise, download the dataset and cache it.

    Args:
        num_samples (int): Number of examples to sample.
        seed (int): Random seed for reproducibility.
        cache_file (str): Path to the cache file.

    Returns:
        list: A list of sampled examples, each containing "question" and "answer".
    """
    if os.path.exists(cache_file):
        print(f"Loading cached GSM8K data from {cache_file} ...")
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
        if len(cached_data) >= num_samples:
            return random.sample(cached_data, num_samples)
        else:
            print("Cached data has fewer samples than requested. Reloading data...")
            os.remove(cache_file)

    print("Loading GSM8K from Hugging Face (openai/gsm8k) ...")
    ds = load_dataset("openai/gsm8k", "main")
    train_ds = ds["train"]
    total_train_size = len(train_ds)
    print(f"GSM8K train set size: {total_train_size}")

    random.seed(seed)
    indices = random.sample(range(total_train_size), num_samples)

    examples = []
    for idx in indices:
        row = train_ds[idx]
        examples.append({"question": row["question"], "answer": row["answer"]})

    print(f"Caching GSM8K data to {cache_file} ...")
    with open(cache_file, "wb") as f:
        pickle.dump(examples, f)

    return examples