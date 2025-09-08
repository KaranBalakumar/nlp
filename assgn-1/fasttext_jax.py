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
