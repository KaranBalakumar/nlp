# ==================== MAIN PIPELINE (inspired by fasttext.py) ====================
import argparse
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def run_pipeline(train_file, test_file, vector_size=100, epochs=5, min_count=1, batch_size=1024, neg_samples=5, lr=0.05, key=0):
    print(f"Loading data: {train_file} / {test_file}")
    train_labels, train_texts = load_dataset(train_file)
    test_labels, test_texts = load_dataset(test_file)
    print(f"Loaded {len(train_texts)} train, {len(test_texts)} test samples.")

    # Build vocab
    vocab, subword_vocab = build_vocab(train_texts, min_count=min_count)
    print(f"Vocab size: {len(vocab)}, Subword vocab: {len(subword_vocab)}")

    # Init model
    model = FastTextJAX(vocab, subword_vocab, vector_size=vector_size, key=key)

    # Train
    print("Training FastText (JAX)...")
    train_fasttext_jax(model, train_texts, epochs=epochs, batch_size=batch_size, neg_samples=neg_samples, lr=lr, key=key)

    # Intrinsic evaluation: pseudo-perplexity (cosine sim proxy)
    print("\n--- Intrinsic Evaluation (pseudo-perplexity) ---")
    def fasttext_perplexity(texts):
        total_logprob = 0.0
        total_words = 0
        for sent in texts:
            tokens = tokenize_text(sent)
            if len(tokens) < 2:
                continue
            prev_emb = np.array(model.sentence_vector(tokens[:-1]))
            for i in range(1, len(tokens)):
                w_emb = np.array(model.get_word_vector(tokens[i]))
                cos_sim = np.dot(prev_emb, w_emb) / (np.linalg.norm(prev_emb) * np.linalg.norm(w_emb) + 1e-8)
                prob = (cos_sim + 1) / 2
                prob = max(prob, 1e-8)
                total_logprob += np.log2(prob)
                total_words += 1
        if total_words == 0:
            return float('inf')
        ppl = 2 ** (-total_logprob / total_words)
        return ppl
    ppl = fasttext_perplexity(test_texts)
    print(f"FastText Embedding Perplexity: {ppl:.2f}")

    # Extrinsic evaluation: classifier on embeddings
    print("\n--- Extrinsic Evaluation (Classifier) ---")
    X_train = get_sentence_embeddings(train_texts, model)
    X_test = get_sentence_embeddings(test_texts, model)
    clf = LinearSVC(class_weight="balanced", max_iter=2000)
    clf.fit(X_train, train_labels)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(test_labels, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, y_pred, average="weighted", zero_division=0)
    print(f"Classifier - Acc: {acc:.3f}, Prec: {precision:.3f}, Rec: {recall:.3f}, F1: {f1:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='datasets/english/english_2500.txt', help='Path to training file (default: datasets/english/english_2500.txt)')
    parser.add_argument('--test', type=str, default='datasets/english/english_test.txt', help='Path to test file (default: datasets/english/english_test.txt)')
    parser.add_argument('--vector_size', type=int, default=100, help='Embedding vector size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--min_count', type=int, default=1, help='Minimum word count threshold')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--neg_samples', type=int, default=5, help='Number of negative samples')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--key', type=int, default=0, help='Random seed for JAX')
    parser.add_argument('--window', type=int, default=5, help='Context window size (not used in current JAX code)')
    parser.add_argument('--min_n', type=int, default=3, help='Min n-gram length for subwords (not used in current JAX code)')
    parser.add_argument('--max_n', type=int, default=6, help='Max n-gram length for subwords (not used in current JAX code)')
    parser.add_argument('--lang', type=str, default='english', choices=['english', 'hindi'], help='Dataset language (default: english)')
    args = parser.parse_args()
    # Set default paths for Hindi if requested
    if args.lang == 'hindi':
        args.train = 'datasets/hindi/hindi_50k.txt'
        args.test = 'datasets/hindi/hindi_test.txt'
    # Note: window, min_n, max_n are parsed for compatibility but not used in this minimal JAX pipeline
    run_pipeline(
        args.train, args.test, args.vector_size, args.epochs, args.min_count,
        args.batch_size, args.neg_samples, args.lr, args.key
    )
# FastText (JAX version, with subwords, GPU support)
import re
import math
import random
import numpy as np
from collections import Counter, defaultdict
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from tqdm import tqdm

# Data loading and tokenization (same as notebook)
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
    return re.findall(r"\b\w+\b", str(text).lower())

# Vocab and subword vocab

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
    vocab = {w: i for i, (w, c) in enumerate(word_counter.most_common(max_vocab)) if c >= min_count}
    subword_vocab = {sw: i for i, (sw, c) in enumerate(subword_counter.most_common(max_subwords)) if c >= 1}
    return vocab, subword_vocab

# FastText JAX model
class FastTextJAX:
    def __init__(self, vocab, subword_vocab, vector_size=100, key=0):
        self.vocab = vocab
        self.subword_vocab = subword_vocab
        self.vector_size = vector_size
        key = jax.random.PRNGKey(key)
        self.word_emb = jax.random.uniform(key, (len(vocab), vector_size), minval=-0.5/vector_size, maxval=0.5/vector_size)
        self.subword_emb = jax.random.uniform(key, (len(subword_vocab), vector_size), minval=-0.5/vector_size, maxval=0.5/vector_size)

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
            vectors.append(self.word_emb[idx])
        if sub_idxs:
            vectors.append(jnp.mean(self.subword_emb[jnp.array(sub_idxs)], axis=0))
        if vectors:
            return jnp.mean(jnp.stack(vectors), axis=0)
        else:
            return jnp.zeros(self.vector_size)

    def sentence_vector(self, tokens):
        vecs = [self.get_word_vector(w) for w in tokens if w in self.vocab]
        if vecs:
            return jnp.mean(jnp.stack(vecs), axis=0)
        else:
            return jnp.zeros(self.vector_size)

# Training (Negative Sampling, batched, JAX)
def train_fasttext_jax(model, data, epochs=5, batch_size=1024, neg_samples=5, lr=0.05, key=0):
    # Create training pairs (center, context) for skip-gram
    train_pairs = []
    for text in data:
        tokens = tokenize_text(text)
        for i in range(len(tokens) - 1):
            if tokens[i] in model.vocab and tokens[i+1] in model.vocab:
                train_pairs.append((model.vocab[tokens[i]], model.vocab[tokens[i+1]]))
                train_pairs.append((model.vocab[tokens[i+1]], model.vocab[tokens[i]]))
    if not train_pairs:
        return
    train_pairs = np.array(train_pairs)
    num_words = len(model.vocab)
    key = jax.random.PRNGKey(key)
    word_emb = model.word_emb
    subword_emb = model.subword_emb
    for epoch in range(epochs):
        perm = np.random.permutation(len(train_pairs))
        total_loss = 0.0
        for start in range(0, len(train_pairs), batch_size):
            idx = perm[start:start+batch_size]
            batch = train_pairs[idx]
            center, context = batch[:,0], batch[:,1]
            # Negative samples
            neg = np.random.randint(0, num_words, size=(len(center), neg_samples))
            # Forward and loss
            def loss_fn(word_emb, subword_emb):
                c_emb = word_emb[center]
                ctx_emb = word_emb[context]
                pos_score = jnp.sum(c_emb * ctx_emb, axis=1)
                pos_loss = -jnp.log(jax.nn.sigmoid(pos_score) + 1e-8)
                neg_emb = word_emb[neg]
                neg_score = jnp.sum(c_emb[:, None, :] * neg_emb, axis=2)
                neg_loss = -jnp.sum(jnp.log(1 - jax.nn.sigmoid(neg_score) + 1e-8), axis=1)
                return jnp.mean(pos_loss + neg_loss)
            grads = grad(loss_fn, argnums=(0,1))(word_emb, subword_emb)
            word_emb = word_emb - lr * grads[0]
            subword_emb = subword_emb - lr * grads[1]
        model.word_emb = word_emb
        model.subword_emb = subword_emb
        print(f"Epoch {epoch+1}/{epochs} done.")

# Sentence embedding extraction
def get_sentence_embeddings(texts, model):
    return np.stack([np.array(model.sentence_vector(tokenize_text(t))) for t in texts])

# The rest of the pipeline (classification, evaluation) can remain as in the PyTorch version, using sklearn.
