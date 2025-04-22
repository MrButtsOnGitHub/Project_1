import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import yaml
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
stemmer = PorterStemmer()

# Tokenizer and stemmer
def tokenize(sentence):
    return nltk.word_tokenize(sentence.lower())

def stem(word):
    return stemmer.stem(word)

def bag_of_words(sentence, words):
    sentence_words = [stem(w) for w in tokenize(sentence)]
    return np.array([1 if w in sentence_words else 0 for w in words])

# Load YAML dataset
with open("intents.yml", "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)

# Build vocabulary
all_words = []
tags = []
xy = []

for intent in data["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend([stem(word) for word in w])
        xy.append((w, tag))

words = sorted(set(all_words))
tags = sorted(set(tags))

# Prepare training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(" ".join(pattern_sentence), words)
    X_train.append(bag)
    y_train.append(tags.index(tag))

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

# Model
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

input_size = len(words)
hidden_size = 16
output_size = len(tags)
model = ChatbotModel(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
epochs = 200
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Chat
def chatbot():
    print("Chatbot is ready! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        bow = bag_of_words(user_input, words)
        bow_tensor = torch.tensor(bow, dtype=torch.float32).unsqueeze(0)
        output = model(bow_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)

        if confidence.item() > 0.75:
            predicted_tag = tags[predicted_index.item()]
            for intent in data["intents"]:
                if intent["tag"] == predicted_tag:
                    if predicted_tag == "random_mix_prompt":
                        mixable_intents = [i for i in data["intents"] if "setups" in i and "twists" in i]
                        if mixable_intents:
                            chosen = random.choice(mixable_intents)
                            setup = random.choice(chosen["setups"])
                            twist = random.choice(chosen["twists"])
                            print("Bot:", f"{setup} {twist}")
                        else:
                            print("Bot:", "Oops! No mixable prompts found.")
                    else:
                        print("Bot:", random.choice(intent["responses"]))
        else:
            print("Bot: I'm not sure what you mean. Can you rephrase that?")


chatbot()
