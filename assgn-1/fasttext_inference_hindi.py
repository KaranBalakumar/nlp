import glob
import jax.numpy as jnp

def test_classification():
    """Simple function to test classification with a sentence"""
    
    # 1. Find available models
    models = glob.glob("fasttext_*.pkl")
    if not models:
        print("No FastText models found!")
        return
    
    print("Available models:")
    for i, model in enumerate(models):
        print(f"{i+1}. {model}")
    
    # 2. Select model
    choice = int(input("Choose model number: ")) - 1
    model_path = models[choice]
    
    # 3. Load model
    print(f"Loading {model_path}...")
    model = FastTextJAX.load(model_path)
    
    # 4. Load test data to create classes
    test_texts, test_labels = load_classification_data("datasets/hindi/hindi_test.txt")
    centroids = train_centroid_classifier(model, test_texts, test_labels, lang='hindi')
    
    print(f"Available classes: {list(centroids.keys())}")
    
    # 5. Test with sentences
    while True:
        sentence = input("\nEnter sentence (or 'quit' to exit): ")
        if sentence.lower() == 'quit':
            break
            
        # Get sentence embedding
        embedding = text_to_embedding(model, sentence, lang='hindi')
        
        # Find best match
        best_class = None
        best_score = -1
        
        for class_name, centroid in centroids.items():
            # Calculate similarity
            similarity = float(jnp.dot(embedding, centroid) / 
                             (jnp.linalg.norm(embedding) * jnp.linalg.norm(centroid) + 1e-8))
            
            if similarity > best_score:
                best_score = similarity
                best_class = class_name
        
        print(f"Predicted class: {best_class} (score: {best_score:.3f})")

# Quick test function
def quick_test():
    """Test with predefined sentences"""
    
    # Use first available model
    models = glob.glob("fasttext_*.pkl")
    if not models:
        print("No models found!")
        return
    
    model = FastTextJAX.load(models[0])
    test_texts, test_labels = load_classification_data("datasets/hindi/hindi_test.txt")
    centroids = train_centroid_classifier(model, test_texts, test_labels, lang='hindi')
    
    # Test sentences
    sentences = [
        "यह बहुत अच्छा है",
        "मुझे यह पसंद नहीं",
        "यह ठीक है"
    ]
    
    for sentence in sentences:
        embedding = text_to_embedding(model, sentence, lang='hindi')
        
        best_class = max(centroids.keys(), 
                        key=lambda c: jnp.dot(embedding, centroids[c]))
        
        print(f"'{sentence}' -> {best_class}")

if _name_ == "_main_":
    # Choose what to run
    print("1. Interactive test")
    print("2. Quick test")
    
    choice = input("Choose (1 or 2): ")
    
    if choice == "1":
        test_classification()
    else:
        quick_test()