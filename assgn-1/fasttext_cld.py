import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import re
from collections import defaultdict, Counter
import random
from typing import List, Tuple, Dict, Set
import pickle
import math
import os
import time
from tqdm import tqdm

# Set JAX to use GPU if available
print(f"JAX devices: {jax.devices()}")

class FastTextJAX:
    def __init__(self, vector_size=100, window=5, min_count=5, min_n=3, max_n=6,
                 sg=1, negative=5, learning_rate=0.025, epochs=5):
        """
        FastText implementation with JAX GPU acceleration
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.min_n = min_n
        self.max_n = max_n
        self.sg = sg
        self.negative = negative
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.epochs = epochs

        # Vocabularies (keep on CPU)
        self.word_vocab = {}
        self.subword_vocab = {}
        self.index_to_word = {}
        self.index_to_subword = {}

        # Embedding matrices (will be JAX arrays)
        self.word_vectors = None
        self.subword_vectors = None
        self.output_vectors = None

        # Training data
        self.word_counts = Counter()
        self.subword_counts = Counter()
        self.total_words = 0

        # Negative sampling table
        self.negative_table = []
        
        # Caching
        self._word_vector_cache = {}
        self._subword_cache = {}

        # JIT compiled functions
        self._jit_sigmoid = jit(self._sigmoid_impl)
        self._jit_update_vectors = jit(self._update_vectors_impl)

    def preprocess_text(self, text: str) -> List[str]:
        """Basic text preprocessing"""
        text = text.lower()
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        return words

    def get_subwords(self, word: str) -> List[str]:
        """Extract character n-grams from a word - CACHED VERSION"""
        if word in self._subword_cache:
            return self._subword_cache[word]
            
        padded_word = f"<{word}>"
        subwords = []

        for n in range(self.min_n, self.max_n + 1):
            for i in range(len(padded_word) - n + 1):
                subwords.append(padded_word[i:i + n])

        subwords.append(word)
        self._subword_cache[word] = subwords
        return subwords

    def build_vocab(self, sentences: List[List[str]]):
        """Build vocabulary from sentences"""
        print("Building vocabulary...")

        # Count words
        for sentence in sentences:
            for word in sentence:
                self.word_counts[word] += 1
                self.total_words += 1

        # Filter words by minimum count
        filtered_words = {word: count for word, count in self.word_counts.items()
                         if count >= self.min_count}

        # Build word vocabulary
        self.word_vocab = {word: idx for idx, word in enumerate(filtered_words.keys())}
        self.index_to_word = {idx: word for word, idx in self.word_vocab.items()}

        # Count subwords
        for word in self.word_vocab.keys():
            subwords = self.get_subwords(word)
            for subword in subwords:
                self.subword_counts[subword] += self.word_counts[word]

        # Build subword vocabulary
        self.subword_vocab = {subword: idx for idx, subword in enumerate(self.subword_counts.keys())}
        self.index_to_subword = {idx: subword for subword, idx in self.subword_vocab.items()}

        print(f"Vocabulary size: {len(self.word_vocab)} words, {len(self.subword_vocab)} subwords")

    def init_vectors(self):
        """Initialize embedding matrices as JAX arrays"""
        vocab_size = len(self.word_vocab)
        subword_size = len(self.subword_vocab)

        # Initialize with small random values
        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)
        
        scale = 1.0 / self.vector_size
        
        # Convert to JAX arrays and move to GPU
        self.word_vectors = jax.random.uniform(
            key1, (vocab_size, self.vector_size), 
            minval=-scale, maxval=scale, dtype=jnp.float32
        )
        self.subword_vectors = jax.random.uniform(
            key2, (subword_size, self.vector_size), 
            minval=-scale, maxval=scale, dtype=jnp.float32
        )
        self.output_vectors = jnp.zeros((vocab_size, self.vector_size), dtype=jnp.float32)

    @staticmethod
    def _sigmoid_impl(x):
        """JAX implementation of sigmoid"""
        x = jnp.clip(x, -20, 20)
        return 1.0 / (1.0 + jnp.exp(-x))

    @staticmethod
    def _update_vectors_impl(word_vectors, subword_vectors, output_vectors, 
                           center_idx, subword_indices, context_idx, 
                           learning_rate, center_vec, pos_error, neg_errors, neg_indices):
        """JIT compiled vector updates"""
        # Update output vector for positive sample
        output_vectors = output_vectors.at[context_idx].add(pos_error * center_vec)
        
        # Update word vector
        pos_grad = pos_error * output_vectors[context_idx]
        word_vectors = word_vectors.at[center_idx].add(pos_grad)
        
        # Update subword vectors - vectorized
        subword_vectors = subword_vectors.at[subword_indices].add(pos_grad)
        
        # Update negative samples - fully vectorized approach
        valid_mask = neg_indices >= 0
        valid_neg_indices = jnp.where(valid_mask, neg_indices, 0)
        
        # Apply updates only where mask is valid
        neg_updates = jnp.where(valid_mask[:, None], 
                               neg_errors[:, None] * center_vec[None, :], 
                               0.0)
        
        # Scatter add the updates to output vectors
        output_vectors = output_vectors.at[valid_neg_indices].add(neg_updates)
        
        # Compute negative gradients for center word and subwords
        valid_neg_grads = jnp.where(valid_mask[:, None],
                                   neg_errors[:, None] * output_vectors[valid_neg_indices],
                                   0.0)
        total_neg_grad = jnp.sum(valid_neg_grads, axis=0)
        
        # Update word vector with negative gradients
        word_vectors = word_vectors.at[center_idx].add(total_neg_grad)
        
        # Update subword vectors with negative gradients  
        subword_vectors = subword_vectors.at[subword_indices].add(total_neg_grad)
        
        return word_vectors, subword_vectors, output_vectors

    def get_word_vector(self, word: str) -> jnp.ndarray:
        """Get vector representation of a word - JAX optimized"""
        if word in self._word_vector_cache:
            return self._word_vector_cache[word]
            
        word_vec = jnp.zeros(self.vector_size, dtype=jnp.float32)
        count = 0

        # Get subwords
        subwords = self.get_subwords(word)

        # Collect all relevant indices
        subword_indices = []
        for subword in subwords:
            if subword in self.subword_vocab:
                subword_indices.append(self.subword_vocab[subword])
                count += 1

        # Vectorized addition of subword vectors
        if subword_indices:
            subword_vecs = self.subword_vectors[jnp.array(subword_indices)]
            word_vec = word_vec + jnp.sum(subword_vecs, axis=0)

        # Add word vector if word is in vocabulary
        if word in self.word_vocab:
            word_idx = self.word_vocab[word]
            word_vec = word_vec + self.word_vectors[word_idx]
            count += 1

        # Normalize by count
        if count > 0:
            word_vec = word_vec / count

        # Cache the result (convert to numpy for caching)
        self._word_vector_cache[word] = word_vec
        return word_vec

    def create_negative_table(self):
        """Create negative sampling table"""
        print("Creating negative sampling table...")
        table_size = int(5e4)  # Smaller table for memory efficiency
        power = 0.75

        vocab_words = [(word, count) for word, count in self.word_counts.items()
                       if word in self.word_vocab]

        if not vocab_words:
            print("No words in vocabulary for negative sampling!")
            return

        total_pow = sum([count**power for _, count in vocab_words])

        self.negative_table = []
        for word, count in vocab_words:
            prob = (count**power) / total_pow
            word_idx = self.word_vocab[word]
            freq = int(prob * table_size)
            self.negative_table.extend([word_idx] * max(1, freq))

        if len(self.negative_table) == 0:
            self.negative_table = list(range(len(self.word_vocab)))

        print(f"Negative table size: {len(self.negative_table)}")

    def get_negative_samples(self, target_word: int, num_samples: int) -> List[int]:
        """Get negative samples for a target word"""
        if not self.negative_table:
            candidates = [i for i in range(len(self.word_vocab)) if i != target_word]
            return random.choices(candidates, k=min(num_samples, len(candidates)))

        negative_samples = []
        attempts = 0
        max_attempts = num_samples * 3

        while len(negative_samples) < num_samples and attempts < max_attempts:
            candidate = random.choice(self.negative_table)
            if candidate != target_word and candidate not in negative_samples:
                negative_samples.append(candidate)
            attempts += 1

        # Pad with -1 if not enough samples
        while len(negative_samples) < num_samples:
            negative_samples.append(-1)

        return negative_samples

    def train_skipgram_jax(self, center_word: str, context_words: List[str]):
        """Train skip-gram with JAX acceleration"""
        if center_word not in self.word_vocab:
            return

        center_idx = self.word_vocab[center_word]
        center_vec = self.get_word_vector(center_word)

        # Get subwords for gradient updates
        subwords = self.get_subwords(center_word)
        subword_indices = jnp.array([self.subword_vocab[sw] for sw in subwords 
                                   if sw in self.subword_vocab])

        for context_word in context_words:
            if context_word not in self.word_vocab:
                continue

            context_idx = self.word_vocab[context_word]

            # Positive sample - JAX operations
            score = jnp.dot(center_vec, self.output_vectors[context_idx])
            pred = self._jit_sigmoid(score)
            pos_error = self.learning_rate * (1.0 - pred)

            # Negative samples
            negative_samples = self.get_negative_samples(context_idx, self.negative)
            neg_indices = jnp.array(negative_samples)
            
            # Compute negative errors in batch
            neg_scores = jnp.where(
                neg_indices >= 0,
                jnp.dot(self.output_vectors[jnp.maximum(neg_indices, 0)], center_vec),
                0.0
            )
            neg_preds = self._jit_sigmoid(neg_scores)
            neg_errors = self.learning_rate * (0.0 - neg_preds)

            # Update vectors using JIT compiled function
            self.word_vectors, self.subword_vectors, self.output_vectors = \
                self._jit_update_vectors(
                    self.word_vectors, self.subword_vectors, self.output_vectors,
                    center_idx, subword_indices, context_idx,
                    self.learning_rate, center_vec, pos_error, neg_errors, neg_indices
                )

    def train(self, sentences: List[List[str]]):
        """Train the FastText model with JAX acceleration"""
        print("Starting JAX-accelerated training...")

        # Build vocabulary
        print("Building vocabulary...")
        self.build_vocab(sentences)

        if len(self.word_vocab) == 0:
            print("No words in vocabulary after filtering! Try reducing min_count.")
            return

        # Initialize vectors
        self.init_vectors()

        # Create negative sampling table
        self.create_negative_table()

        # Pre-filter sentences
        print("Pre-filtering sentences...")
        filtered_sentences = []
        for sentence in tqdm(sentences, desc="Filtering sentences"):
            filtered_words = [word for word in sentence if word in self.word_vocab]
            if len(filtered_words) > 1:
                filtered_sentences.append(filtered_words)
        
        print(f"Filtered to {len(filtered_sentences)} useful sentences")

        # Training loop
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            random.shuffle(filtered_sentences)

            words_in_epoch = 0
            
            # Progress bar for sentences in this epoch
            sentence_pbar = tqdm(filtered_sentences, desc=f"Epoch {epoch+1}", unit="sent")
            
            for sentence in sentence_pbar:
                for i, word in enumerate(sentence):
                    # Get context words
                    context_start = max(0, i - self.window)
                    context_end = min(len(sentence), i + self.window + 1)
                    context_words = sentence[context_start:i] + sentence[i+1:context_end]

                    if context_words:
                        if self.sg:  # Skip-gram
                            self.train_skipgram_jax(word, context_words)

                    words_in_epoch += 1
                
                # Update progress bar description
                if words_in_epoch % 1000 == 0:
                    sentence_pbar.set_postfix({
                        'words': f"{words_in_epoch:,}",
                        'lr': f"{self.learning_rate:.6f}"
                    })

            print(f"  Processed {words_in_epoch:,} words in epoch {epoch + 1}")

            # Clear cache periodically
            if epoch % 2 == 0:
                self._word_vector_cache.clear()

            # Learning rate decay
            self.learning_rate = self.initial_learning_rate * (1.0 - (epoch + 1) / self.epochs)
            self.learning_rate = max(self.learning_rate, self.initial_learning_rate * 0.0001)
            print(f"  Learning rate: {self.learning_rate:.6f}")

        print("JAX-accelerated training completed!")

    def most_similar(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most similar words using JAX"""
        word_vec = self.get_word_vector(word)

        if jnp.allclose(word_vec, 0):
            print(f"Word '{word}' has zero vector")
            return []

        # Vectorized similarity computation
        all_words = list(self.word_vocab.keys())
        all_vecs = jnp.stack([self.get_word_vector(w) for w in all_words])
        
        # Compute cosine similarities in batch
        norms = jnp.linalg.norm(all_vecs, axis=1)
        word_norm = jnp.linalg.norm(word_vec)
        
        valid_mask = (norms > 0) & (word_norm > 0)
        similarities = jnp.where(
            valid_mask,
            jnp.dot(all_vecs, word_vec) / (norms * word_norm),
            -1.0
        )

        # Get top-k (excluding the word itself)
        top_indices = jnp.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            if len(results) >= top_k:
                break
            word_candidate = all_words[int(idx)]
            if word_candidate != word and similarities[idx] > -1.0:
                results.append((word_candidate, float(similarities[idx])))

        return results

    def save(self, filepath: str):
        """Save the trained model (convert JAX arrays to numpy)"""
        model_data = {
            'word_vocab': self.word_vocab,
            'subword_vocab': self.subword_vocab,
            'index_to_word': self.index_to_word,
            'index_to_subword': self.index_to_subword,
            'word_vectors': np.array(self.word_vectors),  # Convert to numpy
            'subword_vectors': np.array(self.subword_vectors),
            'output_vectors': np.array(self.output_vectors),
            'vector_size': self.vector_size,
            'window': self.window,
            'min_n': self.min_n,
            'max_n': self.max_n,
            'word_counts': self.word_counts,
            'subword_counts': self.subword_counts
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load a trained model (convert numpy to JAX arrays)"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.word_vocab = model_data['word_vocab']
        self.subword_vocab = model_data['subword_vocab']
        self.index_to_word = model_data['index_to_word']
        self.index_to_subword = model_data['index_to_subword']
        
        # Convert back to JAX arrays
        self.word_vectors = jnp.array(model_data['word_vectors'])
        self.subword_vectors = jnp.array(model_data['subword_vectors'])
        self.output_vectors = jnp.array(model_data['output_vectors'])
        
        self.vector_size = model_data['vector_size']
        self.window = model_data.get('window', 5)
        self.min_n = model_data['min_n']
        self.max_n = model_data['max_n']
        self.word_counts = model_data['word_counts']
        self.subword_counts = model_data['subword_counts']
        print(f"Model loaded from {filepath}")


class FastTextJAXEvaluator:
    def __init__(self, model):
        self.model = model
        self._embedding_cache = {}

    def preprocess_text(self, text: str) -> List[str]:
        text = text.lower()
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        return words

    def calculate_word_probability(self, target_word: str, context_words: List[str]) -> float:
        """
        FIXED: Proper perplexity calculation using sampling-based softmax approximation
        """
        if target_word not in self.model.word_vocab:
            return 1e-10
        
        target_idx = self.model.word_vocab[target_word]
        
        if not context_words:
            return 1e-10
        
        # Get context representation
        context_vecs = []
        for word in context_words:
            if word in self.model.word_vocab:
                vec = self.model.get_word_vector(word)
                context_vecs.append(vec)
        
        if not context_vecs:
            return 1e-10
        
        context_vec = jnp.mean(jnp.stack(context_vecs), axis=0)
        
        # PROPER METHOD: Sample a subset of vocabulary for softmax approximation
        vocab_size = len(self.model.word_vocab)
        
        if vocab_size <= 1000:
            # Small vocab: use full softmax
            all_scores = jnp.dot(self.model.output_vectors, context_vec)
            exp_scores = jnp.exp(all_scores - jnp.max(all_scores))
            probabilities = exp_scores / jnp.sum(exp_scores)
            return float(probabilities[target_idx])
        else:
            # Large vocab: use sampled softmax approximation
            sample_size = min(1000, vocab_size // 2)
            
            # Always include target word
            sample_indices = [target_idx]
            
            # Add random sample from vocabulary
            other_indices = [i for i in range(vocab_size) if i != target_idx]
            sampled_others = random.sample(other_indices, min(sample_size - 1, len(other_indices)))
            sample_indices.extend(sampled_others)
            
            sample_indices = jnp.array(sample_indices)
            
            # Compute scores for sample
            sample_vectors = self.model.output_vectors[sample_indices]
            sample_scores = jnp.dot(sample_vectors, context_vec)
            
            # Apply softmax to sample
            exp_scores = jnp.exp(sample_scores - jnp.max(sample_scores))
            sample_probs = exp_scores / jnp.sum(exp_scores)
            
            # Target probability is the first one (since target_idx is first)
            target_prob = float(sample_probs[0])
            
            # Scale probability to account for sampling
            scaling_factor = sample_size / vocab_size
            adjusted_prob = target_prob * scaling_factor
            
            return max(adjusted_prob, 1e-10)

    def load_classification_data(self, data_file_path: str) -> Tuple[List[str], List[str]]:
        """Load data from 'label','text' format"""
        texts = []
        labels = []

        if not os.path.exists(data_file_path):
            print(f"Data file {data_file_path} not found!")
            return texts, labels

        with open(data_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    if line.startswith("'") and "'," in line:
                        quote_comma_idx = line.find("','")
                        if quote_comma_idx > 0:
                            label_part = line[1:quote_comma_idx]
                            text_part = line[quote_comma_idx + 3:-1]
                            
                            if label_part and text_part:
                                texts.append(text_part)
                                labels.append(label_part)
                    else:
                        if ',' in line:
                            parts = line.split(',', 1)
                            if len(parts) == 2:
                                label = parts[0].strip().strip("'\"")
                                text = parts[1].strip().strip("'\"")
                                
                                if text and label:
                                    texts.append(text)
                                    labels.append(label)
                except Exception as e:
                    print(f"Warning: Could not parse line {line_num}: {str(e)}")
                    continue

        print(f"Loaded {len(texts)} samples with {len(set(labels))} unique classes")
        return texts, labels

    def calculate_perplexity(self, test_file_path: str, max_sentences: int = 500) -> float:
        """
        FIXED: Calculate perplexity with proper train/test separation
        """
        print("Calculating perplexity with proper methodology...")

        if not os.path.exists(test_file_path):
            print(f"Test file {test_file_path} not found!")
            return float('inf')

        # Load test data and extract ONLY text (no labels for perplexity)
        texts, labels = self.load_classification_data(test_file_path)
        
        if not texts:
            print("No valid texts found in test file!")
            return float('inf')

        # FIXED: Use SEPARATE portion of test data for perplexity (first half)
        # This ensures no overlap with classification evaluation (second half)
        perplexity_texts = texts[:len(texts)//2]
        
        if not perplexity_texts:
            print("No texts available for perplexity calculation!")
            return float('inf')

        # Process into sentences
        sentences = []
        for text in tqdm(perplexity_texts[:max_sentences], desc="Processing texts"):
            words = self.preprocess_text(text)
            if len(words) > 1:
                sentences.append(words)

        if not sentences:
            print("No valid sentences for perplexity!")
            return float('inf')

        print(f"Evaluating perplexity on {len(sentences)} sentences (separate from classification data)")

        total_log_prob = 0.0
        total_words = 0
        oov_words = 0

        # Progress bar for perplexity calculation
        for sentence in tqdm(sentences, desc="Computing perplexity"):
            for i, word in enumerate(sentence):
                # Skip OOV words for fair comparison
                if word not in self.model.word_vocab:
                    oov_words += 1
                    continue
                    
                context_start = max(0, i - self.model.window)
                context_end = min(len(sentence), i + self.model.window + 1)
                context_words = sentence[context_start:i] + sentence[i+1:context_end]

                # Only use context words that are in vocabulary
                context_words = [w for w in context_words if w in self.model.word_vocab]
                
                if context_words:  # Only calculate if we have valid context
                    prob = self.calculate_word_probability(word, context_words)
                    total_log_prob += math.log2(prob)
                    total_words += 1

        if total_words == 0:
            print("No words could be evaluated!")
            return float('inf')

        avg_log_prob = total_log_prob / total_words
        perplexity = 2 ** (-avg_log_prob)

        print(f"Perplexity: {perplexity:.2f} (lower is better)")
        print(f"Total words evaluated: {total_words}")
        print(f"OOV words skipped: {oov_words}")
        print(f"Coverage: {total_words/(total_words + oov_words)*100:.1f}%")

        return perplexity

    def text_to_embedding(self, text: str, method='average') -> jnp.ndarray:
        """Convert text to embedding using JAX"""
        cache_key = f"{hash(text)}_{method}"
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
            
        words = self.preprocess_text(text)
        if not words:
            result = jnp.zeros(self.model.vector_size, dtype=jnp.float32)
        else:
            embeddings = []
            for word in words:
                vec = self.model.get_word_vector(word)
                embeddings.append(vec)

            if not embeddings:
                result = jnp.zeros(self.model.vector_size, dtype=jnp.float32)
            else:
                embeddings = jnp.stack(embeddings)

                if method == 'average':
                    result = jnp.mean(embeddings, axis=0)
                elif method == 'max':
                    result = jnp.max(embeddings, axis=0)
                elif method == 'sum':
                    result = jnp.sum(embeddings, axis=0)
                else:
                    result = jnp.mean(embeddings, axis=0)
        
        if len(self._embedding_cache) < 5000:
            self._embedding_cache[cache_key] = result
        
        return result

    def train_test_split(self, texts: List[str], labels: List[str], test_size: float = 0.2, random_state: int = 42):
        random.seed(random_state)
        combined = list(zip(texts, labels))
        random.shuffle(combined)

        split_idx = int(len(combined) * (1 - test_size))
        train_data = combined[:split_idx]
        test_data = combined[split_idx:]

        train_texts, train_labels = zip(*train_data) if train_data else ([], [])
        test_texts, test_labels = zip(*test_data) if test_data else ([], [])

        return list(train_texts), list(train_labels), list(test_texts), list(test_labels)

    def simple_classifier_train(self, X_train: jnp.ndarray, y_train: List[str]):
        """Train centroid classifier using JAX"""
        if len(X_train) == 0 or len(y_train) == 0:
            return {'centroids': jnp.array([]), 'labels': [], 'label_to_idx': {}}
        
        unique_labels = sorted(list(set(y_train)))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        y_train_idx = jnp.array([label_to_idx[label] for label in y_train])

        n_classes = len(unique_labels)
        n_features = X_train.shape[1]

        centroids = jnp.zeros((n_classes, n_features), dtype=jnp.float32)
        
        for i, label in enumerate(unique_labels):
            class_mask = y_train_idx == i
            class_samples = X_train[class_mask]
            
            if jnp.sum(class_mask) > 0:
                centroids = centroids.at[i].set(jnp.mean(class_samples, axis=0))

        return {'centroids': centroids, 'labels': unique_labels, 'label_to_idx': label_to_idx}

    def simple_classifier_predict(self, model_params: dict, X_test: jnp.ndarray) -> List[str]:
        """Predict using JAX-accelerated classifier"""
        centroids = model_params['centroids']
        labels = model_params['labels']

        if len(centroids) == 0 or len(labels) == 0:
            return ['unknown'] * len(X_test)

        # Vectorized similarity computation
        X_test_norm = jnp.linalg.norm(X_test, axis=1, keepdims=True)
        centroids_norm = jnp.linalg.norm(centroids, axis=1, keepdims=True)
        
        # Compute all similarities at once
        similarities = jnp.dot(X_test, centroids.T) / (X_test_norm * centroids_norm.T)
        
        # Get predictions
        predicted_indices = jnp.argmax(similarities, axis=1)
        
        predictions = [labels[int(idx)] for idx in predicted_indices]
        return predictions

    def calculate_metrics(self, y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        """Calculate classification metrics"""
        if len(y_true) == 0 or len(y_pred) == 0:
            return {'accuracy': 0.0, 'macro_precision': 0.0, 'macro_recall': 0.0, 'macro_f1': 0.0}

        labels = list(set(y_true + y_pred))
        confusion = defaultdict(lambda: defaultdict(int))
        for true, pred in zip(y_true, y_pred):
            confusion[true][pred] += 1

        total_correct = 0
        total_samples = len(y_true)
        precisions, recalls, f1s = [], [], []

        for label in labels:
            tp = confusion[label][label]
            fp = sum(confusion[other][label] for other in labels if other != label)
            fn = sum(confusion[label][other] for other in labels if other != label)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            total_correct += tp

        return {
            'accuracy': total_correct / total_samples if total_samples > 0 else 0.0,
            'macro_precision': np.mean(precisions) if precisions else 0.0,
            'macro_recall': np.mean(recalls) if recalls else 0.0,
            'macro_f1': np.mean(f1s) if f1s else 0.0
        }

    def evaluate_classification(self, data_file_path: str, dataset_name: str) -> Dict[str, float]:
        """
        FIXED: Classification evaluation using SECOND HALF of test data
        (First half is reserved for perplexity)
        """
        print(f"\n=== Evaluating Classification on {dataset_name} ===")

        texts, labels = self.load_classification_data(data_file_path)

        if len(texts) == 0:
            return {'accuracy': 0.0, 'macro_precision': 0.0, 'macro_recall': 0.0, 'macro_f1': 0.0}

        # CRITICAL: Use SECOND HALF of test data for classification
        # First half is reserved for perplexity calculation
        split_point = len(texts) // 2
        classification_texts = texts[split_point:]
        classification_labels = labels[split_point:]
        
        print(f"Using {len(classification_texts)} samples for classification (second half of test data)")
        print(f"Sample: '{classification_labels[0]}' -> {classification_texts[0][:100]}...")

        if len(classification_texts) < 10:
            print("Warning: Very small classification dataset!")
            
        # Further split classification data into train/test
        train_texts, train_labels, test_texts, test_labels = self.train_test_split(
            classification_texts, classification_labels, test_size=0.4, random_state=42
        )
        
        print(f"Classification split - Train: {len(train_texts)}, Test: {len(test_texts)}")

        if len(test_texts) == 0:
            return {'accuracy': 0.0, 'macro_precision': 0.0, 'macro_recall': 0.0, 'macro_f1': 0.0}

        print("Converting texts to embeddings...")
        X_train = jnp.stack([self.text_to_embedding(text) for text in tqdm(train_texts, desc="Training embeddings")])
        X_test = jnp.stack([self.text_to_embedding(text) for text in tqdm(test_texts, desc="Test embeddings")])

        classifier = self.simple_classifier_train(X_train, train_labels)
        predictions = self.simple_classifier_predict(classifier, X_test)

        metrics = self.calculate_metrics(test_labels, predictions)

        print(f"\nResults for {dataset_name}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")

        return metrics

    def full_evaluation(self, test_file_path: str = None) -> Dict:
        """
        FIXED: Run complete evaluation with proper data separation
        """
        results = {}

        print("="*60)
        print("STARTING EVALUATION WITH PROPER DATA SEPARATION")
        print("="*60)

        if test_file_path and os.path.exists(test_file_path):
            # Load test data once
            texts, labels = self.load_classification_data(test_file_path)
            
            if len(texts) < 20:
                print("Warning: Test dataset is very small. Results may not be reliable.")
                
            split_point = len(texts) // 2
            print(f"Data separation strategy:")
            print(f"  - First half ({split_point} samples): Perplexity evaluation")
            print(f"  - Second half ({len(texts) - split_point} samples): Classification evaluation")
            
            print("\n1. INTRINSIC EVALUATION (Perplexity)")
            print("-" * 40)
            perplexity = self.calculate_perplexity(test_file_path)
            results['perplexity'] = perplexity

            print("\n2. EXTRINSIC EVALUATION (Classification)")
            print("-" * 40)
            classification_metrics = self.evaluate_classification(test_file_path, "Text Classification")
            results['classification'] = classification_metrics
        else:
            print("Test file not found!")
            results['perplexity'] = None
            results['classification'] = {}

        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        if results['perplexity']:
            print(f"Perplexity: {results['perplexity']:.2f}")
        if results['classification']:
            print(f"Classification F1: {results['classification']['macro_f1']:.4f}")
        
        print(f"Data separation: ✓ No overlap between perplexity and classification evaluation")

        return results


# Fast JAX configuration with Hindi support
def get_jax_configs():
    """Optimized JAX configurations for both English and Hindi"""
    return [
        # English datasets
        {
            'name': 'English 2500 Model (JAX)',
            'data_path': 'datasets/english/english_2500.txt',
            'test_path': 'datasets/english/english_test.txt',
            'language': 'english',
            'config': {
                'vector_size': 64,
                'window': 3,
                'min_count': 3,
                'min_n': 3,
                'max_n': 4,
                'epochs': 3,
                'learning_rate': 0.1,
                'negative': 5
            }
        },
        {
            'name': 'English 15000 Model (JAX)',
            'data_path': 'datasets/english/english_15000.txt',
            'test_path': 'datasets/english/english_test.txt',
            'language': 'english',
            'config': {
                'vector_size': 64,
                'window': 3,
                'min_count': 5,
                'min_n': 3,
                'max_n': 4,
                'epochs': 3,
                'learning_rate': 0.1,
                'negative': 5
            }
        },
        {
            'name': 'English 30000 Model (JAX)',
            'data_path': 'datasets/english/english_30000.txt',
            'test_path': 'datasets/english/english_test.txt',
            'language': 'english',
            'config': {
                'vector_size': 64,
                'window': 3,
                'min_count': 8,
                'min_n': 3,
                'max_n': 4,
                'epochs': 3,
                'learning_rate': 0.1,
                'negative': 5
            }
        },
        # Hindi datasets
        {
            'name': 'Hindi 2500 Model (JAX)',
            'data_path': 'datasets/hindi/hindi_2500.txt',
            'test_path': 'datasets/hindi/hindi_test.txt',
            'language': 'hindi',
            'config': {
                'vector_size': 64,
                'window': 3,
                'min_count': 3,
                'min_n': 3,
                'max_n': 4,
                'epochs': 3,
                'learning_rate': 0.1,
                'negative': 5
            }
        },
        {
            'name': 'Hindi 15000 Model (JAX)',
            'data_path': 'datasets/hindi/hindi_15000.txt',
            'test_path': 'datasets/hindi/hindi_test.txt',
            'language': 'hindi',
            'config': {
                'vector_size': 64,
                'window': 3,
                'min_count': 5,
                'min_n': 3,
                'max_n': 4,
                'epochs': 3,
                'learning_rate': 0.1,
                'negative': 5
            }
        },
        {
            'name': 'Hindi 30000 Model (JAX)',
            'data_path': 'datasets/hindi/hindi_30000.txt',
            'test_path': 'datasets/hindi/hindi_test.txt',
            'language': 'hindi',
            'config': {
                'vector_size': 64,
                'window': 3,
                'min_count': 8,
                'min_n': 3,
                'max_n': 4,
                'epochs': 3,
                'learning_rate': 0.1,
                'negative': 5
            }
        }
    ]


def train_and_evaluate_jax_model(data_file_path: str, model_config: dict, model_name: str, test_file_path: str = None):
    """Train and evaluate JAX FastText model with detailed timing"""
    print(f"\n{'='*80}")
    print(f"TRAINING AND EVALUATING {model_name}")
    print(f"{'='*80}")

    if not os.path.exists(data_file_path):
        print(f"Training data file {data_file_path} not found!")
        return None

    # Record system info
    import platform
    import psutil
    
    system_info = {
        'python_version': platform.python_version(),
        'system': platform.system(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'jax_devices': str(jax.devices())
    }
    
    print(f"System Info: {system_info['system']}, CPU cores: {system_info['cpu_count']}, "
          f"Memory: {system_info['memory_gb']}GB, JAX devices: {system_info['jax_devices']}")

    # Start overall timing
    total_start_time = time.time()

    # Data loading timing
    data_load_start = time.time()
    with open(data_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    model = FastTextJAX(**model_config)

    # Preprocess text into sentences
    sentences = []
    for line in text.split('\n'):
        if line.strip():
            words = model.preprocess_text(line.strip())
            if len(words) > 1:
                sentences.append(words)

    data_load_time = time.time() - data_load_start
    print(f"Data loading and preprocessing: {data_load_time:.2f} seconds")
    print(f"Loaded {len(sentences)} sentences for training")

    if len(sentences) == 0:
        print("No valid sentences found!")
        return None

    # Training timing
    training_start_time = time.time()
    model.train(sentences)
    training_time = time.time() - training_start_time

    if len(model.word_vocab) == 0:
        print("Model training failed!")
        return None

    print(f"Training completed in: {training_time:.2f} seconds")
    print(f"Training speed: {len(sentences) / training_time:.1f} sentences/second")

    # Model saving timing
    save_start = time.time()
    model_filepath = f"{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_jax_model.pkl"
    model.save(model_filepath)
    save_time = time.time() - save_start
    print(f"Model saving: {save_time:.2f} seconds")

    print(f"\nModel Statistics:")
    print(f"  Vocabulary size: {len(model.word_vocab):,}")
    print(f"  Subword vocabulary size: {len(model.subword_vocab):,}")
    print(f"  Vector dimension: {model.vector_size}")
    print(f"  Total parameters: {(len(model.word_vocab) + len(model.subword_vocab)) * model.vector_size:,}")

    # Evaluation timing
    if test_file_path and os.path.exists(test_file_path):
        eval_start_time = time.time()
        evaluator = FastTextJAXEvaluator(model)
        results = evaluator.full_evaluation(test_file_path)
        eval_time = time.time() - eval_start_time
        print(f"Evaluation completed in: {eval_time:.2f} seconds")
    else:
        print(f"Test file {test_file_path} not found, skipping evaluation")
        results = {'perplexity': None, 'classification': {}}
        eval_time = 0

    total_time = time.time() - total_start_time

    # Compile timing summary
    timing_summary = {
        'data_loading_time': data_load_time,
        'training_time': training_time,
        'evaluation_time': eval_time,
        'model_saving_time': save_time,
        'total_time': total_time,
        'training_speed_sentences_per_sec': len(sentences) / training_time if training_time > 0 else 0,
        'system_info': system_info
    }

    print(f"\n{'='*50}")
    print("TIMING SUMMARY")
    print(f"{'='*50}")
    print(f"Data Loading:     {data_load_time:.2f}s")
    print(f"Training:         {training_time:.2f}s ({len(sentences) / training_time:.1f} sentences/sec)")
    print(f"Evaluation:       {eval_time:.2f}s")
    print(f"Model Saving:     {save_time:.2f}s")
    print(f"Total Time:       {total_time:.2f}s")

    return {
        'model': model, 
        'results': results, 
        'model_filepath': model_filepath,
        'timing': timing_summary
    }


def main_jax():
    """Main function for JAX FastText with comprehensive timing"""
    print("="*100)
    print("FASTTEXT JAX: COMPREHENSIVE TRAINING AND EVALUATION")
    print("="*100)
    
    pipeline_start_time = time.time()
    
    configs = get_jax_configs()

    all_results = {}
    trained_models = {}
    all_timings = {}

    # Use tqdm for overall progress
    config_pbar = tqdm(configs, desc="Training models")
    
    for config in config_pbar:
        model_name = config['name']
        data_path = config['data_path']
        test_path = config['test_path']
        language = config['language']
        model_config = config['config']
        
        config_pbar.set_description(f"Processing {model_name}")

        if not os.path.exists(data_path):
            print(f"\nDataset {data_path} not found! Skipping {model_name}")
            continue

        print(f"\n{'*'*60}")
        print(f"Processing: {model_name} ({language.upper()})")
        print(f"{'*'*60}")

        result = train_and_evaluate_jax_model(data_path, model_config, model_name, test_path)

        if result:
            all_results[model_name] = result['results']
            trained_models[model_name] = result['model']
            all_timings[model_name] = result['timing']

    pipeline_total_time = time.time() - pipeline_start_time

    # Comprehensive results display
    if all_results:
        print(f"\n{'='*120}")
        print("COMPREHENSIVE MODEL COMPARISON")
        print(f"{'='*120}")

        # Header
        header = f"{'Model':<25} {'Language':<8} {'Dataset':<8} {'Vocab':<8} {'Perplexity':<12} {'Accuracy':<10} {'F1':<8} {'Train(s)':<10} {'Eval(s)':<8}"
        print(header)
        print("-" * 120)

        # Results for each model
        for model_name, results in all_results.items():
            # Extract info from model name
            if 'English' in model_name:
                language = 'English'
            elif 'Hindi' in model_name:
                language = 'Hindi'
            else:
                language = 'Unknown'
                
            if '2500' in model_name:
                dataset_size = '2.5K'
            elif '15000' in model_name:
                dataset_size = '15K'
            elif '30000' in model_name:
                dataset_size = '30K'
            else:
                dataset_size = 'Unknown'

            # Get metrics
            model = trained_models[model_name]
            vocab_size = f"{len(model.word_vocab):,}"
            
            perp = results.get('perplexity', 'N/A')
            accuracy = results.get('classification', {}).get('accuracy', 'N/A')
            f1 = results.get('classification', {}).get('macro_f1', 'N/A')
            
            # Get timing info
            timing = all_timings[model_name]
            train_time = timing['training_time']
            eval_time = timing['evaluation_time']
            
            # Format strings
            perp_str = f"{perp:.2f}" if isinstance(perp, (int, float)) and perp != float('inf') else str(perp)
            acc_str = f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) else str(accuracy)
            f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
            
            # Print row
            row = f"{model_name:<25} {language:<8} {dataset_size:<8} {vocab_size:<8} {perp_str:<12} {acc_str:<10} {f1_str:<8} {train_time:<10.1f} {eval_time:<8.1f}"
            print(row)

        # Performance analysis
        print(f"\n{'='*80}")
        print("PERFORMANCE ANALYSIS")
        print(f"{'='*80}")

        # Best models by metric
        english_models = {name: res for name, res in all_results.items() if 'English' in name}
        hindi_models = {name: res for name, res in all_results.items() if 'Hindi' in name}

        for lang_name, models in [('English', english_models), ('Hindi', hindi_models)]:
            if not models:
                continue
                
            print(f"\n{lang_name} Models:")
            
            # Best by perplexity (lower is better)
            valid_perp = {name: res['perplexity'] for name, res in models.items() 
                         if res.get('perplexity') and res['perplexity'] != float('inf')}
            if valid_perp:
                best_perp = min(valid_perp.items(), key=lambda x: x[1])
                print(f"  Best Perplexity: {best_perp[0]} ({best_perp[1]:.2f})")

            # Best by F1
            valid_f1 = {name: res['classification']['macro_f1'] for name, res in models.items() 
                       if res.get('classification') and 'macro_f1' in res['classification']}
            if valid_f1:
                best_f1 = max(valid_f1.items(), key=lambda x: x[1])
                print(f"  Best F1-Score: {best_f1[0]} ({best_f1[1]:.4f})")

            # Fastest training
            lang_timings = {name: timing for name, timing in all_timings.items() if lang_name in name}
            if lang_timings:
                fastest = min(lang_timings.items(), key=lambda x: x[1]['training_time'])
                print(f"  Fastest Training: {fastest[0]} ({fastest[1]['training_time']:.1f}s)")

        # Overall timing summary
        print(f"\n{'='*60}")
        print("TIMING SUMMARY")
        print(f"{'='*60}")
        total_training_time = sum(timing['training_time'] for timing in all_timings.values())
        total_eval_time = sum(timing['evaluation_time'] for timing in all_timings.values())
        
        print(f"Total Pipeline Time:  {pipeline_total_time:.1f}s")
        print(f"Total Training Time:  {total_training_time:.1f}s")
        print(f"Total Evaluation Time: {total_eval_time:.1f}s")
        print(f"Models Trained:       {len(trained_models)}")
        
        if trained_models:
            avg_training_time = total_training_time / len(trained_models)
            print(f"Average Training Time: {avg_training_time:.1f}s per model")

    else:
        print("\nNo models were trained successfully.")
        print("Please check that your dataset files exist:")
        for config in configs:
            print(f"  - {config['data_path']}")
            print(f"  - {config['test_path']}")

    return all_results, trained_models, all_timings


# Example usage
if __name__ == "__main__":
    print("FastText with JAX GPU acceleration - Comprehensive Evaluation")
    print(f"Available devices: {jax.devices()}")
    
    results, models, timings = main_jax()
    
    print("\n" + "="*100)
    print("FASTTEXT JAX PIPELINE COMPLETED!")
    print("="*100)
    
    if models:
        print(f"Successfully trained {len(models)} models:")
        for name in models:
            timing = timings[name]
            vocab_size = len(models[name].word_vocab)
            train_time = timing['training_time']
            print(f"  - {name}: {vocab_size:,} words, {train_time:.1f}s training")
        
        print("\nModel files saved:")
        for name in models:
            filename = f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_jax_model.pkl"
            print(f"  - {filename}")
            
        print(f"\nSystem Configuration:")
        sample_timing = list(timings.values())[0]['system_info']
        print(f"  - {sample_timing['system']}, {sample_timing['cpu_count']} CPU cores")
        print(f"  - {sample_timing['memory_gb']}GB RAM")
        print(f"  - JAX devices: {sample_timing['jax_devices']}")
        print(f"  - Python {sample_timing['python_version']}")
    else:
        print("No models trained. Check your data files:")
        print("Required files:")
        print("  English: english_2500.txt, english_15000.txt, english_30000.txt, english_test.txt")
        print("  Hindi: hindi_2500.txt, hindi_15000.txt, hindi_30000.txt, hindi_test.txt")