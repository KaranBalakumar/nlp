# glove_pipeline.py
import re
import torch
import pickle
import random
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### ---------------- DATA PREPROCESSING ---------------- ###

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_dataset(file_path):
    texts = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            label, text = line.strip().split(",", 1)
            labels.append(label)
            texts.append(clean_text(text))
    return texts, labels

def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in texts:
        tokens = text.split()
        counter.update(tokens)
    vocab = {word: idx for idx, (word, freq) in enumerate(counter.items()) if freq >= min_freq}
    return vocab

def build_cooccurrence(texts, vocab, window_size=5):
    cooccurrences = defaultdict(float)
    for text in texts:
        tokens = text.split()
        token_ids = [vocab[token] for token in tokens if token in vocab]
        for center_idx, center_word in enumerate(token_ids):
            start = max(0, center_idx - window_size)
            end = min(len(token_ids), center_idx + window_size + 1)
            for context_idx in range(start, end):
                if center_idx == context_idx:
                    continue
                context_word = token_ids[context_idx]
                distance = abs(center_idx - context_idx)
                increment = 1.0 / distance
                cooccurrences[(center_word, context_word)] += increment
    return cooccurrences

def save_model(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

### ---------------- GLOVE MODEL ---------------- ###

class GloVeModel:
    def __init__(self, vocab_size, embed_dim, x_max=100, alpha=0.75):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.x_max = x_max
        self.alpha = alpha
        self.W = torch.randn(vocab_size, embed_dim, device=device) * 0.01
        self.ContextW = torch.randn(vocab_size, embed_dim, device=device) * 0.01
        self.b = torch.zeros(vocab_size, device=device)
        self.Contextb = torch.zeros(vocab_size, device=device)

    def train(self, cooccurrences, epochs=50, learning_rate=0.05):
        for epoch in range(epochs):
            total_loss = 0
            for (i_idx, j_idx), x_ij in cooccurrences.items():
                w_i = self.W[i_idx]
                w_j = self.ContextW[j_idx]
                b_i = self.b[i_idx]
                b_j = self.Contextb[j_idx]
                
                weight = (x_ij / self.x_max) ** self.alpha if x_ij < self.x_max else 1.0
                pred = torch.dot(w_i, w_j) + b_i + b_j
                loss = weight * (pred - torch.log(torch.tensor(x_ij, device=device))) ** 2
                
                total_loss += loss.item()
                
                grad = 2 * weight * (pred - torch.log(torch.tensor(x_ij, device=device)))
                grad_w_i = grad * w_j
                grad_w_j = grad * w_i
                grad_b_i = grad
                grad_b_j = grad
                
                self.W[i_idx] -= learning_rate * grad_w_i
                self.ContextW[j_idx] -= learning_rate * grad_w_j
                self.b[i_idx] -= learning_rate * grad_b_i
                self.Contextb[j_idx] -= learning_rate * grad_b_j
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    def get_embeddings(self):
        return (self.W + self.ContextW).cpu().detach().numpy()

    def save(self, path):
        save_model({
            'W': self.W.cpu(),
            'ContextW': self.ContextW.cpu(),
            'b': self.b.cpu(),
            'Contextb': self.Contextb.cpu()
        }, path)

    def load(self, path):
        data = load_model(path)
        self.W = data['W'].to(device)
        self.ContextW = data['ContextW'].to(device)
        self.b = data['b'].to(device)
        self.Contextb = data['Contextb'].to(device)

### ---------------- INTRINSIC EVALUATION ---------------- ###

def compute_perplexity(model, texts, vocab):
    total_log_prob = 0
    total_words = 0
    embeddings = model.get_embeddings()
    for text in texts:
        tokens = text.split()
        token_ids = [vocab[token] for token in tokens if token in vocab]
        for center_idx, center_word in enumerate(token_ids):
            start = max(0, center_idx - 5)
            end = min(len(token_ids), center_idx + 6)
            context_ids = token_ids[start:center_idx] + token_ids[center_idx+1:end]
            for context_id in context_ids:
                w_center = embeddings[center_word]
                w_context = embeddings[context_id]
                score = np.dot(w_center, w_context)
                prob = 1 / (1 + np.exp(-score))  # Sigmoid
                total_log_prob += np.log(prob + 1e-9)
                total_words += 1
    perplexity = np.exp(-total_log_prob / total_words) if total_words > 0 else float('inf')
    return perplexity

### ---------------- EXTRINSIC EVALUATION ---------------- ###

def get_sentence_vector(text, vocab, embeddings):
    tokens = text.split()
    token_ids = [vocab[token] for token in tokens if token in vocab]
    if not token_ids:
        return np.zeros(embeddings.shape[1])
    vecs = embeddings[token_ids]
    return np.mean(vecs, axis=0)

def classify_and_evaluate(model, vocab, texts_train, labels_train, texts_test, labels_test):
    embeddings = model.get_embeddings()
    X_train = np.array([get_sentence_vector(text, vocab, embeddings) for text in texts_train])
    X_test = np.array([get_sentence_vector(text, vocab, embeddings) for text in texts_test])
    
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, labels_train)
    preds = clf.predict(X_test)
    
    acc = accuracy_score(labels_test, preds)
    prec = precision_score(labels_test, preds, average='weighted', zero_division=0)
    rec = recall_score(labels_test, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels_test, preds, average='weighted', zero_division=0)
    return acc, prec, rec, f1

### ---------------- MAIN PIPELINE ---------------- ###

if __name__ == "__main__":
    file_path = "assgn-1/datasets/english/english_30000.txt"  # Dataset file
    print("Loading dataset...")
    texts, labels = load_dataset(file_path)
    
    # Shuffle and split
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    split = int(0.8 * len(combined))
    train_data = combined[:split]
    test_data = combined[split:]
    texts_train, labels_train = zip(*train_data)
    texts_test, labels_test = zip(*test_data)
    
    print("Building vocabulary...")
    vocab = build_vocab(texts_train, min_freq=1)
    print(f"Vocabulary size: {len(vocab)}")
    
    print("Building co-occurrence matrix...")
    cooccurrences = build_cooccurrence(texts_train, vocab, window_size=5)
    print(f"Total co-occurrence pairs: {len(cooccurrences)}")
    
    print("Training GloVe model...")
    glove = GloVeModel(vocab_size=len(vocab), embed_dim=50)
    glove.train(cooccurrences, epochs=50, learning_rate=0.05)
    
    print("Saving GloVe model...")
    glove.save("glove_model.pkl")
    
    print("Performing intrinsic evaluation (Perplexity)...")
    perp = compute_perplexity(glove, texts_test, vocab)
    print(f"Perplexity on test set: {perp:.4f}")
    
    print("Performing extrinsic evaluation (Classification)...")
    acc, prec, rec, f1 = classify_and_evaluate(glove, vocab, texts_train, labels_train, texts_test, labels_test)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
