import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

with open('dataset/caption_pair.csv', 'r') as f:
    pairs = f.readlines()
    
sum = 0

for pair in tqdm(pairs):
    human = pair.split(' / ')[0].split(': ')[1]
    pred = pair.split('/ Model: ')[1][:-2]
    texts = [human, pred]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

    # Calculate cosine similarities
    # Cosine similarities are in [-1, 1]. Higher means more similar
    cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
    print(cosine_sim_0_1)
    sum += cosine_sim_0_1

print("Average : ")
print(sum/len(pairs))