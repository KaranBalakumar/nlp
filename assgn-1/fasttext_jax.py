import jax
import jax.numpy as jnp
import numpy as np
import re
from collections import Counter
import torch
import argparse
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torchmetrics

# Tokenization
def tokenize_text(text):
    return re.findall(r"\b\w+\b", str(text).lower())

# Vocabulary builder with subwords
def build_vocab(texts, max_vocab=10000, min_count=1, max_subwords=50000):
    word_counter = Counter()
    subword_counter = Counter()
    for text in texts:
        tokens = tokenize_text(text)
        for word in tokens:
            word_counter[word] += 1
            padded = f"<{word}>"
            for n in range(3, 7):
                if len(padded) < n: continue
                for i in range(len(padded) - n + 1):
                    subword_counter[padded[i:i+n]] += 1
            subword_counter[word] += 1
    vocab = {w: i for i, (w, c) in enumerate(word_counter.most_common(max_vocab)) if c >= min_count}
    subword_vocab = {sw: i for i, (sw, c) in enumerate(subword_counter.most_common(max_subwords)) if c >= 1}
    return vocab, subword_vocab

def get_subword_indices(word, subword_vocab):
    padded = f"<{word}>"
    subwords = set([word])
    for n in range(3, 7):
        if len(padded) < n: continue
        for i in range(len(padded) - n + 1):
            subwords.add(padded[i:i+n])
    return [subword_vocab[sw] for sw in subwords if sw in subword_vocab]

class FastTextJAX:
    def __init__(self, vocab, subword_vocab, vector_size=100, key=0):
        self.vocab = vocab
        self.subword_vocab = subword_vocab
        self.vector_size = vector_size
        self.rng = jax.random.PRNGKey(key)
        self.word_emb = jax.random.uniform(self.rng, (len(vocab), vector_size), minval=-0.5/vector_size, maxval=0.5/vector_size)
        self.subword_emb = jax.random.uniform(self.rng, (len(subword_vocab), vector_size), minval=-0.5/vector_size, maxval=0.5/vector_size)

    def get_word_vector(self, word):
        idx = self.vocab.get(word, None)
        sub_idxs = get_subword_indices(word, self.subword_vocab)
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

# Simple skip-gram training loop (negative sampling omitted for brevity)
def train_skipgram(model, texts, epochs=5, lr=0.05, batch_size=1024, neg_samples=5, key=0):
    # Simple negative sampling skip-gram training in JAX
    # This is a minimal, non-optimized version for demonstration
    pairs = []
    for text in texts:
        tokens = tokenize_text(text)
        for i in range(len(tokens) - 1):
            if tokens[i] in model.vocab and tokens[i+1] in model.vocab:
                pairs.append((model.vocab[tokens[i]], model.vocab[tokens[i+1]]))
                pairs.append((model.vocab[tokens[i+1]], model.vocab[tokens[i]]))
    if not pairs:
        return
    pairs = np.array(pairs)
    num_words = len(model.vocab)
    word_emb = model.word_emb
    subword_emb = model.subword_emb
    key = jax.random.PRNGKey(key)
    for epoch in range(epochs):
        perm = np.random.permutation(len(pairs))
        for start in range(0, len(pairs), batch_size):
            idx = perm[start:start+batch_size]
            batch = pairs[idx]
            center, context = batch[:,0], batch[:,1]
            neg = np.random.randint(0, num_words, size=(len(center), neg_samples))
            def loss_fn(word_emb, subword_emb):
                c_emb = word_emb[center]
                ctx_emb = word_emb[context]
                pos_score = jnp.sum(c_emb * ctx_emb, axis=1)
                pos_loss = -jnp.log(jax.nn.sigmoid(pos_score) + 1e-8)
                neg_emb = word_emb[neg]
                neg_score = jnp.sum(c_emb[:, None, :] * neg_emb, axis=2)
                neg_loss = -jnp.sum(jnp.log(1 - jax.nn.sigmoid(neg_score) + 1e-8), axis=1)
                return jnp.mean(pos_loss + neg_loss)
            grads = jax.grad(loss_fn, argnums=(0,1))(word_emb, subword_emb)
            word_emb = word_emb - lr * grads[0]
            subword_emb = subword_emb - lr * grads[1]
        model.word_emb = word_emb
        model.subword_emb = subword_emb
        print(f"Epoch {epoch+1}/{epochs} done.")

def pseudo_perplexity(texts, model):
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

def sklearn_classifier_eval(train_texts, train_labels, test_texts, test_labels, model):
    X_train = np.stack([np.array(model.sentence_vector(tokenize_text(t))) for t in train_texts])
    X_test  = np.stack([np.array(model.sentence_vector(tokenize_text(t))) for t in test_texts])
    clf = LinearSVC(class_weight="balanced", max_iter=2000)
    clf.fit(X_train, train_labels)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(test_labels, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, y_pred, average="weighted", zero_division=0)
    return acc, precision, recall, f1

# Metrics with PyTorch
def torch_metrics(y_true, y_pred):
    # Ensure torch tensors
    y_true_t = torch.tensor(y_true)
    y_pred_t = torch.tensor(y_pred)
    acc = (y_pred_t == y_true_t).float().mean().item()
    # Precision/recall/f1 for multi-class classification (macro averaging)
    precision = torchmetrics.functional.precision(y_pred_t, y_true_t, average='macro', num_classes=len(set(y_true)))
    recall = torchmetrics.functional.recall(y_pred_t, y_true_t, average='macro', num_classes=len(set(y_true)))
    f1 = torchmetrics.functional.f1_score(y_pred_t, y_true_t, average='macro', num_classes=len(set(y_true)))
    return acc, precision.item(), recall.item(), f1.item()

# Example evaluation: nearest centroid classifier
def evaluate_fasttext_embeddings(train_texts, train_labels, test_texts, test_labels, model):
    # Sentence embeddings
    X_train = np.stack([np.array(model.sentence_vector(tokenize_text(t))) for t in train_texts])
    X_test  = np.stack([np.array(model.sentence_vector(tokenize_text(t))) for t in test_texts])
    label2centroid = {}
    for lbl in set(train_labels):
        inds = [i for i,l in enumerate(train_labels) if l==lbl]
        label2centroid[lbl] = np.mean(X_train[inds], axis=0)
    # Nearest centroid prediction
    y_pred = []
    for x in X_test:
        sims = {lbl: np.dot(x,cent)/(np.linalg.norm(x)*np.linalg.norm(cent)+1e-8)
                for lbl,cent in label2centroid.items()}
        y_pred.append(max(sims, key=sims.get))
    # Encode string labels to int for metrics
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_true_enc = le.fit_transform(test_labels)
    y_pred_enc = le.transform(y_pred)
    return torch_metrics(y_true_enc, y_pred_enc)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='datasets/english/english_2500.txt')
    parser.add_argument('--test', type=str, default='datasets/english/english_test.txt')
    parser.add_argument('--vector_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--min_count', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--neg_samples', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--key', type=int, default=0)
    parser.add_argument('--lang', type=str, default='english', choices=['english', 'hindi'])
    args = parser.parse_args()
    if args.lang == 'hindi':
        args.train = 'datasets/hindi/hindi_50k.txt'
        args.test = 'datasets/hindi/hindi_test.txt'
    # Load data
    from pathlib import Path
    assert Path(args.train).exists(), f"Train file not found: {args.train}"
    assert Path(args.test).exists(), f"Test file not found: {args.test}"
    with open(args.train, 'r', encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(args.test, 'r', encoding='utf-8') as f:
        test_lines = f.readlines()
    # Parse labels/texts
    def parse_dataset(lines):
        data = []
        current_label = None
        current_text = []
        for line in lines:
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
        return list(texts), list(labels)
    train_texts, train_labels = parse_dataset(train_lines)
    test_texts, test_labels = parse_dataset(test_lines)
    # Build vocab
    vocab, subword_vocab = build_vocab(train_texts, min_count=args.min_count)
    model = FastTextJAX(vocab, subword_vocab, vector_size=args.vector_size, key=args.key)
    # Train
    print("Training FastText (JAX) model...")
    train_skipgram(model, train_texts, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, neg_samples=args.neg_samples, key=args.key)
    # Intrinsic evaluation
    print("\n--- Intrinsic Evaluation (pseudo-perplexity) ---")
    ppl = pseudo_perplexity(test_texts, model)
    print(f"FastText Embedding Perplexity: {ppl:.2f}")
    # Extrinsic evaluation: classifier
    print("\n--- Extrinsic Evaluation (Classifier) ---")
    acc, precision, recall, f1 = sklearn_classifier_eval(train_texts, train_labels, test_texts, test_labels, model)
    print(f"Classifier - Acc: {acc:.3f}, Prec: {precision:.3f}, Rec: {recall:.3f}, F1: {f1:.3f}")
    # Extrinsic evaluation: centroid
    print("\n--- Extrinsic Evaluation (Centroid) ---")
    acc, precision, recall, f1 = evaluate_fasttext_embeddings(train_texts, train_labels, test_texts, test_labels, model)
    print(f"Centroid - Acc: {acc:.3f}, Prec: {precision:.3f}, Rec: {recall:.3f}, F1: {f1:.3f}")

if __name__ == "__main__":
    main()

