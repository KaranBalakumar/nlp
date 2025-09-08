import re
import random
import math
from collections import Counter, defaultdict


# Path to datasets directory (change this variable as needed)
DATASETS_DIR = 'datasets'  # Default: local datasets folder

# Google Colab setup (optional, auto-override DATASETS_DIR if in Colab)
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    DATASETS_DIR = '/content/drive/MyDrive/nlp/datasets'  # Change as needed for your Drive

import os
import re
import math
import random
import torch
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------
# Data Loading (Notebook style)
# -----------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# -----------------------
# Dataset Loader (Same as notebook)
# -----------------------
def load_dataset(filepath):
    data = []
    current_label = None
    current_text = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("'") and line.count("','") >= 1:
                if current_label is not None:
                    data.append((current_label, " ".join(current_text)))
                parts = line.split("','", 1)
                current_label = parts[0].strip("'")
                current_text = [parts[1].rstrip("'")] if len(parts) > 1 else []
            else:
                current_text.append(line)
        if current_label is not None:
            data.append((current_label, " ".join(current_text)))

    labels, texts = zip(*data)
    return list(labels), list(texts)

def tokenize_text(text):
    """Tokenize text same as notebook"""
    return re.findall(r"\b\w+\b", str(text).lower())

# -----------------------
# Vocabulary Building (with subwords)
# -----------------------
def build_vocab(texts, max_vocab=10000, min_count=1, max_subwords=50000):
    word_counter = Counter()
    subword_counter = Counter()
    for text in texts:
        tokens = tokenize_text(text)
        for word in tokens:
            word_counter[word] += 1
            padded = f"<{word}>"
            for n in range(3, 7):
                if len(padded) < n:
                    continue
                for i in range(len(padded) - n + 1):
                    subword_counter[padded[i:i+n]] += 1
            subword_counter[word] += 1
    # Prune
    vocab = {w: i for i, (w, c) in enumerate(word_counter.most_common(max_vocab)) if c >= min_count}
    subword_vocab = {sw: i for i, (sw, c) in enumerate(subword_counter.most_common(max_subwords)) if c >= 1}
    return vocab, subword_vocab


# -----------------------
# FastText Model (PyTorch, subwords)
# -----------------------
class FastTextPyTorch(torch.nn.Module):
    def __init__(self, vocab, subword_vocab, vector_size=100):
        super().__init__()
        self.vocab = vocab
        self.subword_vocab = subword_vocab
        self.vector_size = vector_size
        self.word_emb = torch.nn.Embedding(len(vocab), vector_size)
        self.subword_emb = torch.nn.Embedding(len(subword_vocab), vector_size)
        torch.nn.init.uniform_(self.word_emb.weight, -0.5/vector_size, 0.5/vector_size)
        torch.nn.init.uniform_(self.subword_emb.weight, -0.5/vector_size, 0.5/vector_size)

    def get_word_indices(self, word):
        idx = self.vocab.get(word, None)
        sub_idxs = self.get_subword_indices(word)
        return idx, sub_idxs

    def get_subword_indices(self, word):
        padded = f"<{word}>"
        subwords = set([word])
        for n in range(3, 7):
            if len(padded) < n: continue
            for i in range(len(padded) - n + 1):
                subwords.add(padded[i:i+n])
        return [self.subword_vocab[sw] for sw in subwords if sw in self.subword_vocab]

    def get_word_vector(self, word):
        idx, sub_idxs = self.get_word_indices(word)
        vectors = []
        if idx is not None:
            vectors.append(self.word_emb.weight[idx])
        if sub_idxs:
            vectors.append(self.subword_emb.weight[torch.tensor(sub_idxs, device=self.word_emb.weight.device)])
        if vectors:
            return torch.mean(torch.stack([v.mean(0) if v.ndim > 1 else v for v in vectors]), dim=0)
        else:
            return torch.zeros(self.vector_size, device=self.word_emb.weight.device)

    def sentence_vector(self, tokens):
        vecs = [self.get_word_vector(w) for w in tokens if w in self.vocab]
        if vecs:
            return torch.stack(vecs).mean(0)
        else:
            return torch.zeros(self.vector_size, device=self.word_emb.weight.device)

# -----------------------
# Training (Negative Sampling, batched, GPU)
# -----------------------
def train_fasttext(model, data, epochs=5, batch_size=1024, neg_samples=5, lr=0.05):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # Create training pairs (center, context) for skip-gram
    train_pairs = []
    for text in data:
        tokens = tokenize_text(text)
        # Simple approach: create pairs of adjacent words
        for i in range(len(tokens) - 1):
            if tokens[i] in model.vocab and tokens[i+1] in model.vocab:
                train_pairs.append((model.vocab[tokens[i]], model.vocab[tokens[i+1]]))
                train_pairs.append((model.vocab[tokens[i+1]], model.vocab[tokens[i]])) # Add reverse pair

    if not train_pairs:
        print("No valid training pairs generated.")
        return

    for epoch in range(epochs):
        random.shuffle(train_pairs)
        total_loss = 0.0
        for i in range(0, len(train_pairs), batch_size):
            batch = train_pairs[i:i+batch_size]
            if not batch: continue
            center, context = zip(*batch)
            center = torch.tensor(center, device=DEVICE)
            context = torch.tensor(context, device=DEVICE)

            # Positive samples
            pos_score = (model.word_emb(center) * model.word_emb(context)).sum(1)
            pos_labels = torch.ones_like(pos_score, dtype=torch.float32) # Ensure float32 for BCEWithLogitsLoss

            # Negative samples
            neg_context = torch.randint(0, model.word_emb.num_embeddings, (len(center), neg_samples), device=DEVICE)
            neg_score = (model.word_emb(center).unsqueeze(1) * model.word_emb(neg_context)).sum(2)
            neg_labels = torch.zeros_like(neg_score, dtype=torch.float32) # Ensure float32

            # Loss
            loss = loss_fn(pos_score, pos_labels) + loss_fn(neg_score, neg_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/((len(train_pairs)//batch_size)+1):.4f}")


# -----------------------
# Intrinsic Evaluation (Perplexity, batched)
# -----------------------
class NGramLanguageModel:
    def __init__(self, n=2):
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocab = set()

    def train(self, texts):
        for s in texts:
            tokens = ["<s>"] + tokenize_text(s) + ["</s>"]
            self.vocab.update(tokens)
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i+self.n])
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

    def prob(self, ngram):
        context = ngram[:-1]
        V = len(self.vocab)
        return (self.ngram_counts[ngram] + 1) / (self.context_counts[context] + V)

    def perplexity(self, texts):
        N, log_prob_sum = 0, 0.0
        V = len(self.vocab)
        for s in texts:
            tokens = ["<s>"] * (self.n - 1) + tokenize_text(s) + ["</s>"]
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i+self.n])
                p = self.prob(ngram)
                log_prob_sum += math.log(p, 2)
                N += 1
        return math.pow(2, -log_prob_sum / N) if N > 0 else float('inf')

# -----------------------
# Extrinsic Evaluation (Classification)
# -----------------------
def extrinsic_classification(train_texts, train_labels, test_texts, test_labels, n=1, method="nb"):
    """Classification same as notebook"""
    # Join tokens into sentences
    train_sentences = [" ".join(tokenize_text(t)) for t in train_texts]
    test_sentences = [" ".join(tokenize_text(t)) for t in test_texts]

    vectorizer = TfidfVectorizer(ngram_range=(1, n), max_features=8000)
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)

    if method == "nb":
        clf = MultinomialNB()
    else:
        clf = LinearSVC(class_weight="balanced")

    clf.fit(X_train, train_labels)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(test_labels, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, y_pred, average="weighted", zero_division=0
    )
    return acc, precision, recall, f1

# -----------------------
# Run Experiments (Same as notebook)
# -----------------------
def run_experiment(train_path, test_path, lang="English", method="nb",
                   fasttext_epochs=5, fasttext_min_count=1, fasttext_vector_size=100):
    """Run experiments exactly like the notebook, with FastText parameters"""
    print(f"\n===== {lang} Dataset =====")
    train_labels, train_texts = load_dataset(train_path)
    test_labels, test_texts = load_dataset(test_path)

    print(f"Loaded {len(train_texts)} training samples")
    print(f"Loaded {len(test_texts)} test samples with {len(set(test_labels))} unique labels")

    results = []

    # N-gram Language Model Evaluation
    print("\n=== N-gram Language Model Evaluation ===")
    for n in [1, 2, 3]:
        print(f"\n--- Training {n}-gram Model ---")
        model = NGramLanguageModel(n=n)
        model.train(train_texts)
        ppl = model.perplexity(test_texts)
        print(f"Perplexity: {ppl:.2f}")

    # Extrinsic Classification Evaluation (using TF-IDF)
    print("\n=== Extrinsic Classification Evaluation (TF-IDF) ===")
    for n in [1, 2, 3]:
         acc, precision, recall, f1 = extrinsic_classification(
            train_texts, train_labels, test_texts, test_labels, n=n, method=method
        )
         print(f"--- {n}-gram TF-IDF Classification ---")
         print(f"Acc: {acc:.3f}, Prec: {precision:.3f}, Rec: {recall:.3f}, F1: {f1:.3f}")
         results.append({
            "type": "tfidf",
            "n": n,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })


    # FastText Training (Skip-gram like)
    print("\n=== FastText Training ===")
    print(f"Parameters: epochs={fasttext_epochs}, min_count={fasttext_min_count}, vector_size={fasttext_vector_size}")
    vocab, subword_vocab = build_vocab(train_texts, min_count=fasttext_min_count)
    fasttext_model = FastTextPyTorch(vocab, subword_vocab, vector_size=fasttext_vector_size).to(DEVICE)
    # Pass only texts to train_fasttext as it builds its own pairs
    train_fasttext(fasttext_model, train_texts, epochs=fasttext_epochs)


    # --- Nearest Neighbor prediction using FastText embeddings ---
    print("\n=== Extrinsic Evaluation (FastText Embeddings, Classifier) ===")
    def get_sentence_embedding(text):
        tokens = tokenize_text(text)
        with torch.no_grad():
            emb = fasttext_model.sentence_vector(tokens)
        return emb.detach().cpu().numpy()

    X_train_ft = np.stack([get_sentence_embedding(t) for t in train_texts])
    X_test_ft = np.stack([get_sentence_embedding(t) for t in test_texts])

    # Use a classifier (LinearSVC) on FastText embeddings
    from sklearn.svm import LinearSVC
    clf = LinearSVC(class_weight="balanced", max_iter=2000)
    clf.fit(X_train_ft, train_labels)
    y_pred_cls = clf.predict(X_test_ft)

    acc_cls = accuracy_score(test_labels, y_pred_cls)
    precision_cls, recall_cls, f1_cls, _ = precision_recall_fscore_support(
        test_labels, y_pred_cls, average="weighted", zero_division=0
    )
    print(f"FastText Embedding (Classifier) - Acc: {acc_cls:.3f}, Prec: {precision_cls:.3f}, Rec: {recall_cls:.3f}, F1: {f1_cls:.3f}")
    results.append({
        "type": "fasttext_classifier",
        "accuracy": acc_cls,
        "precision": precision_cls,
        "recall": recall_cls,
        "f1": f1_cls
    })

    # Intrinsic Evaluation: Perplexity using FastText (simple LM over embeddings)
    print("\n=== Intrinsic Perplexity Evaluation (FastText Embeddings) ===")
    # For each test sentence, compute average cosine similarity between each word and previous context embedding
    from numpy.linalg import norm
    def fasttext_perplexity(texts):
        total_logprob = 0.0
        total_words = 0
        for sent in texts:
            tokens = tokenize_text(sent)
            if len(tokens) < 2:
                continue
            prev_emb = fasttext_model.sentence_vector(tokens[:-1]).detach().cpu().numpy()
            for i in range(1, len(tokens)):
                w_emb = fasttext_model.get_word_vector(tokens[i]).detach().cpu().numpy()
                # Use cosine similarity as a proxy for log-prob
                cos_sim = np.dot(prev_emb, w_emb) / (norm(prev_emb) * norm(w_emb) + 1e-8)
                # Map cosine similarity [-1,1] to pseudo-probability [0,1]
                prob = (cos_sim + 1) / 2
                prob = max(prob, 1e-8)
                total_logprob += np.log2(prob)
                total_words += 1
        if total_words == 0:
            return float('inf')
        ppl = 2 ** (-total_logprob / total_words)
        return ppl

    ft_ppl = fasttext_perplexity(test_texts)
    print(f"FastText Embedding Perplexity: {ft_ppl:.2f}")
    results.append({
        "type": "fasttext_perplexity",
        "perplexity": ft_ppl
    })

    return results


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"Running in Colab: {IN_COLAB}")
    print(f"DATASETS_DIR: {DATASETS_DIR}")
    # English
    run_experiment(
        train_path=os.path.join(DATASETS_DIR, "english/english_2500.txt"),
        test_path=os.path.join(DATASETS_DIR, "english/english_test.txt"),
        lang="English (Small)", method="nb", fasttext_epochs=10, fasttext_min_count=1, fasttext_vector_size=50)
    # run_experiment(
    #     train_path=os.path.join(DATASETS_DIR, "english/english_15000.txt"),
    #     test_path=os.path.join(DATASETS_DIR, "english/english_test.txt"),
    #     lang="English (Medium)", method="nb", fasttext_epochs=10, fasttext_min_count=2, fasttext_vector_size=50)
    # run_experiment(
    #     train_path=os.path.join(DATASETS_DIR, "english/english_30000.txt"),
    #     test_path=os.path.join(DATASETS_DIR, "english/english_test.txt"),
    #     lang="English (Large)", method="nb", fasttext_epochs=10, fasttext_min_count=2, fasttext_vector_size=50)
    # Hindi
    # run_experiment(
    #     train_path=os.path.join(DATASETS_DIR, "hindi/hindi_2500.txt"),
    #     test_path=os.path.join(DATASETS_DIR, "hindi/hindi_test.txt"),
    #     lang="Hindi (Small)", method="nb", fasttext_epochs=10, fasttext_min_count=2, fasttext_vector_size=50)
    # run_experiment(
    #     train_path=os.path.join(DATASETS_DIR, "hindi/hindi_15000.txt"),
    #     test_path=os.path.join(DATASETS_DIR, "hindi/hindi_test.txt"),
    #     lang="Hindi (Medium)", method="nb", fasttext_epochs=10, fasttext_min_count=2, fasttext_vector_size=50)
    # run_experiment(
    #     train_path=os.path.join(DATASETS_DIR, "hindi/hindi_30000.txt"),
    #     test_path=os.path.join(DATASETS_DIR, "hindi/hindi_test.txt"),
    #     lang="Hindi (Large)", method="nb", fasttext_epochs=10, fasttext_min_count=2, fasttext_vector_size=50)