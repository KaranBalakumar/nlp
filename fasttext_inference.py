import jax.numpy as jnp
from fasttext_jax import (
    FastTextJAX,
    text_to_embedding,
    train_centroid_classifier,
    predict_centroid,
    load_classification_data
)

# ================================
# 1. Load trained FastText model
# ================================
model = FastTextJAX.load("fasttext_model.pkl")  # change filename if needed

# ================================
# 2. Load labeled dataset & train centroids
# ================================
dataset_path = "datasets/hindi/hindi_test.txt" 
texts, labels = load_classification_data(dataset_path)
centroids = train_centroid_classifier(model, texts, labels)
# ================================
# 3. Classify ONE new sentence
# ================================
sentence = input("Enter a sentence to classify: ")

embedding = text_to_embedding(model, sentence)
prediction = predict_centroid(centroids, jnp.array([embedding]))[0]

print(f"\nSentence: {sentence}")
print(f"Predicted Label: {prediction}")