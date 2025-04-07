from datasets import load_dataset

# Load the GSM8K dataset; choose the 'main' configuration
dataset = load_dataset("gsm8k", "main")

# Access train and test splits
train_data = dataset["train"]
test_data = dataset["test"]

print(train_data[0])



# Save to files
for split, data in zip(["train", "test"], [train_data, test_data]):
    file = f"gsm8k_{split}.txt"
    with open(file, "w", encoding="utf-8") as f:
        for example in train_data:
            f.write(example["question"].strip() + "\n")
    
    print(f"Saved {split} data to {file}")