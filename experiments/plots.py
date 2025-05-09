import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.style.use('ggplot')

def create_levels_plot(data):
    propmt_gen = []
    prompt = []
    gen = []
    for lv in data:
        propmt_gen.append(lv['prompt_gen'])
        prompt.append(lv['prompt'])
        gen.append(lv['gen'])
    plt.figure(figsize=(10, 6))
    plt.bar(prompt, propmt_gen)
    plt.show()

def create_zipf_plot(data):
    data = data.sort_values(by='accepted_count', ascending=False)
    data = data.head(50)
    # Replace tokens that are exactly a single space with "<blank>"
    data['token'] = data['token'].apply(lambda x: "<blank>" if x == " " else x)
    X = data['token']
    Y = data['accepted_count']
    plt.figure(figsize=(20, 8))  # Increased width to accommodate vertical labels
    plt.bar(X, Y)
    plt.title('Accepted Token Counts')
    plt.xlabel('Token')
    plt.ylabel('Count')
    plt.xticks(rotation=90)  # Rotate x-axis labels vertically
    plt.tight_layout()  # Adjust layout to prevent label cutoff

if __name__ == "__main__":
    # Load the data
    data = pd.read_csv("data/zipfs.csv")

    # Plot the data
    create_zipf_plot(data)
    plt.savefig("plots/zipf.png")