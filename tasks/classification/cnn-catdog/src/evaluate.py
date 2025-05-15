import os
from tensorflow.keras.models import load_model
from src.data_loader import get_data_generators

def main():
    # Load the saved model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'cnn_catdog_model.h5')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return

    model = load_model(model_path)

    # Load the test data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    _, test_gen = get_data_generators(data_dir)

    # Evaluate the model
    loss, accuracy = model.evaluate(test_gen, verbose=1)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
