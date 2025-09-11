
import argparse
import jax.numpy as jnp
from fasttext_jax import (
    FastTextJAX,
    text_to_embedding,
    train_centroid_classifier,
    predict_centroid,
    load_classification_data
)

def main():
    parser = argparse.ArgumentParser(description="FastText Inference: Classify a sentence using a trained model and labeled dataset.")
    parser.add_argument('--model', type=str, required=True, help='Path to trained FastText model (.pkl)')
    parser.add_argument('--dataset', type=str, required=True, help='Path to labeled dataset (txt)')
    args = parser.parse_args()

    # 1. Load trained FastText model
    model = FastTextJAX.load(args.model)

    # 2. Load labeled dataset & train centroids
    texts, labels = load_classification_data(args.dataset)
    centroids = train_centroid_classifier(model, texts, labels)

    # 3. Classify ONE new sentence
    sentence = input("Enter a sentence to classify: ")
    embedding = text_to_embedding(model, sentence)
    prediction = predict_centroid(centroids, jnp.array([embedding]))[0]

    print(f"\nSentence: {sentence}")
    print(f"Predicted Label: {prediction}")

if __name__ == "__main__":
    main()