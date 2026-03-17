import numpy as np # type: ignore
import re

def soft_max(raws):
    denom = sum(np.exp(raws))
    return np.exp(raws)/denom

def generate_pairs(token_ids, window):
    l = len(token_ids)
    pairs = []
    for i, word in enumerate(token_ids):
        
        start = max(0, i-window)
        end = min(l, i+window+1)
        for j in range(start, end):
            if i == j:
                continue
            context = token_ids[j]
            pairs.append((word, context))
    return pairs


with open("short-text.txt", "r", encoding="utf-8") as file:
    clean_text = re.sub(r'[^\w\s]', '', file.read()).lower()
    
tokens = clean_text.split()

vocab = list(set(tokens))
word_to_id = {w: i for i, w in enumerate(vocab)}
id_to_word = {i: w for w, i in word_to_id.items()}

token_ids = [word_to_id[w] for w in tokens]

print(tokens)
print(token_ids)

pairs = generate_pairs(token_ids=token_ids, window=4)

V = len(vocab)
N = 100

W1 = np.random.randn(V, N)
W2 = np.random.randn(N, V)

m = 1
lr = 0.1

for epoch in range(200):
    loss = 0
    for target, context in pairs:
        raws = np.dot(W1[target], W2)
        pred = soft_max(raws)
        pred[context] -= 1
        dW2 = np.dot(W1[target].reshape(-1, 1), pred.reshape(1,-1))
        dv = np.dot(W2, pred)
        W2 -= lr*dW2
        W1[target] -= lr*dv
        # print("loss(", context, "): ", -np.log(pred[context]+1))
        loss += -np.log(pred[context]+1)
    print("epoch", epoch, "loss:", loss)

p = []
for word1 in vocab:
    for word2 in vocab:
        if word_to_id[word2] >= word_to_id[word1]:
            continue
        p.append((word1, word2, float(np.sum((W1[word_to_id[word1]]-W1[word_to_id[word2]])**2))))
s = sorted(p, key=lambda x: x[2])
for i in range(100):
    print(s[i])
print("-"*30)

for i in range(1, 100):
    print(s[-i])

