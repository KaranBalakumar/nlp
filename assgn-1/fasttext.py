import re
import random
import heapq
import time
from collections import Counter
from typing import List, Dict, Tuple

# JAX and related libraries
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.tree_util import tree_map
from flax.serialization import to_bytes, from_bytes

# Standard libraries
from tqdm import tqdm
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ==================== Huffman Tree (Unchanged) ====================
class HuffmanNode:
    def __init__(self, freq, word=None, left=None, right=None):
        self.freq = freq; self.word = word; self.left = left; self.right = right; self.index = None
    def __lt__(self, other): return self.freq < other.freq

def build_huffman_tree(word_counts: Dict[str, int]) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], int]:
    heap = [HuffmanNode(freq=freq, word=word) for word, freq in word_counts.items()]
    heapq.heapify(heap)
    internal_nodes = []
    while len(heap) > 1:
        left, right = heapq.heappop(heap), heapq.heappop(heap)
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged); internal_nodes.append(merged)
    internal_nodes.sort(key=lambda n: n.freq)
    for i, node in enumerate(internal_nodes): node.index = i
    paths, codes = {}, {}
    def traverse(node, path_nodes, code_bits):
        if node.word is not None:
            paths[node.word], codes[node.word] = list(path_nodes), list(code_bits)
        else:
            if node.left: traverse(node.left, path_nodes + [node.index], code_bits + [0])
            if node.right: traverse(node.right, path_nodes + [node.index], code_bits + [1])
    if heap: traverse(heap[0], [], [])
    return paths, codes, len(internal_nodes)

# ==================== JAX FastText Model with Evaluation ====================
Params = Dict[str, jnp.ndarray]

class FastTextJAX:
    def __init__(self, vector_size=100, window=5, min_count=5, min_n=3, max_n=6,
                 learning_rate=0.025, epochs=5, batch_size=256):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.min_n = min_n
        self.max_n = max_n
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.word_vocab, self.index_to_word, self.subword_vocab = {}, {}, {}
        self.word_counts = Counter()
        self.huffman_paths, self.huffman_codes = {}, {}
        self.num_internal, self.max_path_len = 0, 0
        self.params: Params = {}

    def preprocess_text(self, text: str, lang='english') -> List[str]:
        if lang == 'english':
            return re.findall(r'\b[a-zA-Z]+\b', text.lower())
        elif lang == 'hindi':
            # A simple regex for Hindi words (Devanagari script)
            return re.findall(r'[\u0900-\u097F]+', text)
        return text.split()

    def get_subwords(self, word: str) -> List[int]:
        padded_word = f"<{word}>"
        subwords = {word}
        for n in range(self.min_n, self.max_n + 1):
            if len(padded_word) < n: continue
            for i in range(len(padded_word) - n + 1):
                subwords.add(padded_word[i:i+n])
        return [self.subword_vocab[sw] for sw in subwords if sw in self.subword_vocab]

    def build_vocab(self, sentences: List[List[str]]):
        print("Building vocabulary...")
        for sent in sentences: self.word_counts.update(sent)
        filtered_counts = {w: c for w, c in self.word_counts.items() if c >= self.min_count}
        self.word_vocab = {w: i for i, w in enumerate(filtered_counts.keys())}
        subword_counts = Counter()
        for word, count in filtered_counts.items():
            padded_word = f"<{word}>"
            subwords = {word}
            for n in range(self.min_n, self.max_n + 1):
                if len(padded_word) < n: continue
                for i in range(len(padded_word) - n + 1): subwords.add(padded_word[i:i+n])
            for sw in subwords: subword_counts[sw] += count
        self.subword_vocab = {sw: i for i, sw in enumerate(subword_counts.keys())}
        self.huffman_paths, self.huffman_codes, self.num_internal = build_huffman_tree(filtered_counts)
        self.max_path_len = max(len(p) for p in self.huffman_paths.values()) if self.huffman_paths else 0
        print(f"Vocab size: {len(self.word_vocab)}, Subwords: {len(self.subword_vocab)}, Internal nodes: {self.num_internal}")

    def init_params(self, key: jax.random.PRNGKey) -> Params:
        w_key, sw_key = jax.random.split(key)
        scale = 0.5 / self.vector_size
        return {
            'subword_vectors': (jax.random.uniform(sw_key, (len(self.subword_vocab), self.vector_size)) - 0.5) * scale,
            'hs_vectors': jnp.zeros((self.num_internal, self.vector_size))
        }

    # --- Training Logic (from scratch optimizer) ---
    def train(self, sentences: List[List[str]]):
        self.build_vocab(sentences)
        key = jax.random.PRNGKey(0)
        self.params = self.init_params(key)
        @jit
        def train_step(current_params: Params, batch: Dict[str, jnp.ndarray], lr: float):
            loss_val, grads = jax.value_and_grad(self.skipgram_loss)(current_params, batch)
            updated_params = tree_map(lambda p, g: p - lr * g, current_params, grads)
            return updated_params, loss_val
        print("\nStarting JAX training...")
        for epoch in range(self.epochs):
            total_loss, batch_count = 0.0, 0
            batch_generator = self._create_training_batches(sentences)
            num_batches = int(np.ceil(sum(len(s) * 2 * self.window for s in sentences) / self.batch_size)) # Approx.
            pbar = tqdm(batch_generator, total=num_batches, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False)
            current_lr = self.learning_rate * (1.0 - (epoch / self.epochs))
            for batch in pbar:
                self.params, loss = train_step(self.params, batch, current_lr)
                total_loss += loss; batch_count += 1
                pbar.set_postfix(loss=f"{loss:.4f}")
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f"Epoch {epoch + 1}/{self.epochs}, Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        print("Training done.")

    def _create_training_batches(self, sentences: List[List[str]]):
        # ... (Implementation is identical to the previous version, omitted for brevity)
        examples = []
        for sent in sentences:
            for i, word in enumerate(sent):
                if word not in self.word_vocab: continue
                current_window = random.randint(1, self.window)
                start, end = max(0, i - current_window), min(len(sent), i + current_window + 1)
                context_words = [sent[j] for j in range(start, end) if i != j]
                if not context_words: continue
                center_sub_indices = self.get_subwords(word)
                for ctx_word in context_words:
                    if ctx_word in self.word_vocab:
                        examples.append({'center_subs': center_sub_indices, 'path': self.huffman_paths[ctx_word], 'code': self.huffman_codes[ctx_word]})
        random.shuffle(examples)
        for i in range(0, len(examples), self.batch_size):
            batch_data = examples[i:i + self.batch_size]
            max_subs = max(len(ex['center_subs']) for ex in batch_data) if batch_data else 0
            padded_subs = np.full((len(batch_data), max_subs), -1, dtype=np.int32)
            padded_paths = np.full((len(batch_data), self.max_path_len), -1, dtype=np.int32)
            padded_codes = np.full((len(batch_data), self.max_path_len), -1, dtype=np.int32)
            for j, ex in enumerate(batch_data):
                padded_subs[j, :len(ex['center_subs'])] = ex['center_subs']
                padded_paths[j, :len(ex['path'])] = ex['path']
                padded_codes[j, :len(ex['code'])] = ex['code']
            yield {'center_subs': jnp.array(padded_subs), 'path': jnp.array(padded_paths), 'code': jnp.array(padded_codes)}

    def skipgram_loss(self, params: Params, batch: Dict[str, jnp.ndarray]) -> float:
        losses = vmap(self._loss_for_one_example)(params, batch['center_subs'], batch['path'], batch['code'])
        return jnp.mean(losses)

    def _loss_for_one_example(self, params: Params, center_subs_indices, path_indices, code_bits):
        subword_mask = (center_subs_indices != -1)
        safe_indices = jnp.where(subword_mask, center_subs_indices, 0)
        center_vec = jnp.sum(params['subword_vectors'][safe_indices], axis=0, where=subword_mask[:, None])
        path_mask = (path_indices != -1)
        safe_path_indices = jnp.where(path_mask, path_indices, 0)
        node_vecs = params['hs_vectors'][safe_path_indices]
        dots = jnp.dot(node_vecs, center_vec)
        labels = 1. - 2. * code_bits
        log_probs = jax.nn.log_sigmoid(labels * dots)
        return -jnp.sum(jnp.where(path_mask, log_probs, 0.0))

    @staticmethod
    @jit
    def get_word_vector(params: Params, subword_indices: jnp.ndarray) -> jnp.ndarray:
        return params['subword_vectors'][subword_indices].sum(axis=0)

    # ==================== INTRINSIC EVALUATION ====================
    def perplexity(self, sentences: List[List[str]]) -> float:
        """Calculates perplexity on a test set."""
        total_log_prob = 0.0
        word_count = 0

        # Pre-compile the probability calculation function
        prob_fn = jit(self._get_word_prob)

        print("Calculating perplexity...")
        for sent in tqdm(sentences):
            for i, word in enumerate(sent):
                if word not in self.word_vocab:
                    continue
                
                start = max(0, i - self.window)
                end = min(len(sent), i + self.window + 1)
                context = [sent[j] for j in range(start, end) if i != j and sent[j] in self.word_vocab]
                
                if not context:
                    continue

                # Prepare data for the JIT function
                context_sub_indices = [self.get_subwords(w) for w in context]
                path = jnp.array(self.huffman_paths[word])
                code = jnp.array(self.huffman_codes[word])

                prob = prob_fn(self.params, context_sub_indices, path, code)
                
                total_log_prob += jnp.log2(prob)
                word_count += 1
        
        if word_count == 0:
            return float('inf')
            
        avg_log_likelihood = total_log_prob / word_count
        perplexity = jnp.power(2.0, -avg_log_likelihood)
        return perplexity

    def _get_word_prob(self, params, context_sub_indices, path, code):
        """JIT-able function to calculate P(word|context) using HS."""
        # Calculate context vector `h`
        context_vectors = [self.get_word_vector(params, jnp.array(subs)) for subs in context_sub_indices]
        h = jnp.mean(jnp.array(context_vectors), axis=0)
        
        # Calculate probability along the Huffman path
        node_vecs = params['hs_vectors'][path]
        dots = jnp.dot(node_vecs, h)
        
        # P = sigmoid(dot) if code=1, (1-sigmoid(dot)) if code=0
        probs = jax.nn.sigmoid(dots)
        probs_for_path = jnp.where(code == 1, probs, 1.0 - probs)
        
        return jnp.prod(probs_for_path)

# ==================== EXTRINSIC EVALUATION HELPERS ====================
def text_to_embedding(model: FastTextJAX, text: str, lang='english') -> jnp.ndarray:
    words = model.preprocess_text(text, lang=lang)
    subword_indices_list = [jnp.array(model.get_subwords(w)) for w in words if w in model.word_vocab]
    if not subword_indices_list: return jnp.zeros(model.vector_size)
    word_vectors = jnp.array([model.get_word_vector(model.params, indices) for indices in subword_indices_list])
    return jnp.mean(word_vectors, axis=0)

def train_centroid_classifier(model: FastTextJAX, texts: List[str], labels: List[str], lang='english') -> Dict[str, jnp.ndarray]:
    unique_labels = sorted(list(set(labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    embeddings = jnp.array([text_to_embedding(model, t, lang) for t in texts])
    label_indices = jnp.array([label_map[l] for l in labels])
    num_labels = len(unique_labels)
    sum_vectors = jax.ops.segment_sum(embeddings, label_indices, num_segments=num_labels)
    counts = jax.ops.segment_sum(jnp.ones(len(texts)), label_indices, num_segments=num_labels)
    mean_vectors = sum_vectors / jnp.maximum(counts, 1)[:, None]
    return {label: mean_vectors[i] for i, label in enumerate(unique_labels)}

def predict_centroid(centroids: Dict[str, jnp.ndarray], text_embeddings: jnp.ndarray) -> List[str]:
    labels = list(centroids.keys())
    centroid_matrix = jnp.stack([centroids[lab] for lab in labels])
    text_norm = text_embeddings / (jnp.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)
    centroid_norm = centroid_matrix / (jnp.linalg.norm(centroid_matrix, axis=1, keepdims=True) + 1e-8)
    similarities = jnp.matmul(text_norm, centroid_norm.T)
    pred_indices = jnp.argmax(similarities, axis=1)
    return [labels[idx] for idx in pred_indices]

def get_classification_metrics(y_true: List[str], y_pred: List[str]):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return {"accuracy": acc, "macro_precision": prec, "macro_recall": rec, "macro_f1": f1}

# ==================== MAIN EVALUATION SCRIPT ====================
def load_classification_data(filepath: str):
    texts, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                labels.append(parts[-1])
                texts.append(" ".join(parts[:-1]))
    return texts, labels

def run_evaluation(language: str, train_file: str, test_file: str):
    """Trains a FastText model and runs intrinsic/extrinsic evaluation."""
    print(f"\n{'='*25}\n E V A L U A T I N G: {language.upper()} \n{'='*25}")
    
    # 1. Load and prepare data
    with open(train_file, "r", encoding="utf-8") as f: text = f.read()
    
    # Initialize model
    model = FastTextJAX(vector_size=100, window=5, min_count=5, epochs=5, batch_size=512)
    
    train_sentences = [model.preprocess_text(line, lang=language) for line in text.split("\n") if line.strip()]
    train_sentences = [s for s in train_sentences if len(s) > 1]
    
    # 2. Train the model
    model.train(train_sentences)

    # 3. Load test data
    test_texts, test_labels = load_classification_data(test_file)
    test_sentences = [model.preprocess_text(text, lang=language) for text in test_texts]

    # 4. INTRINSIC EVALUATION
    print("\n--- Intrinsic Evaluation ---")
    perp = model.perplexity(test_sentences)
    print(f"Perplexity on test set: {perp:.4f}")

    # 5. EXTRINSIC EVALUATION
    print("\n--- Extrinsic Evaluation ---")
    print("Training centroid classifier on test data...")
    centroids = train_centroid_classifier(model, test_texts, test_labels, lang=language)
    
    print("Predicting labels...")
    text_embeddings = jnp.array([text_to_embedding(model, t, lang=language) for t in test_texts])
    predictions = predict_centroid(centroids, text_embeddings)
    
    metrics = get_classification_metrics(test_labels, predictions)
    print("Classification Metrics:")
    for key, val in metrics.items():
        print(f"  {key:<18}: {val:.4f}")

def main():
    print(f"JAX is running on: {jax.default_backend()}")
    
    # Run for English
    run_evaluation(
        language='english',
        train_file='datasets/english/english_2500.txt',
        test_file='datasets/english/english_test.txt'
    )
    
    # Run for Hindi
    run_evaluation(
        language='hindi',
        train_file='datasets/hindi/hindi_50k.txt',
        test_file='datasets/hindi/hindi_test.txt'
    )

    print(f"\n{'='*25}\n How to Compare Models \n{'='*25}")
    print("To compare this FastText model with others (like Word2Vec or GloVe),")
    print("you would run this same evaluation script for each model.")
    print("1. Train each embedding model (FastText, Word2Vec, etc.) on the same training data.")
    print("2. For each trained model, calculate its Perplexity (lower is better).")
    print("3. For each trained model, calculate its classification metrics (higher is better).")
    print("4. Compare the resulting scores across the models to determine which one performs best.")

if __name__ == "__main__":
    main()