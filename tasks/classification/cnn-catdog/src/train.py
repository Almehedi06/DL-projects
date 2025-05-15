import os
import matplotlib.pyplot as plt
from src.data_loader import get_data_generators
from src.model import build_cnn_model

def plot_training_history(history, output_dir='outputs'):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'training_plot.png'))
    plt.show()

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_gen, test_gen = get_data_generators(data_dir)

    model = build_cnn_model()
    history = model.fit(train_gen, epochs=20, validation_data=test_gen, verbose=1)

    plot_training_history(history)

    # ✅ Save the model after training
    os.makedirs('outputs', exist_ok=True)
    model.save('outputs/cnn_catdog_model.h5')
    print("✅ Model saved at outputs/cnn_catdog_model.h5")

if __name__ == "__main__":
    main()
