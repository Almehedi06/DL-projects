# 🐶🐱 CNN-Based Cat vs Dog Image Classification

This project implements a Convolutional Neural Network (CNN) to classify images of cats and dogs using TensorFlow and Keras.

---

## 🚀 Project Structure

```
cnn-catdog/
├── data/                  # Place your 'train' and 'test' folders here
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── test/
│       ├── cats/
│       └── dogs/
├── outputs/               # Saved models and plots
│   ├── cnn_catdog_model.h5
│   └── training_plot.png
├── notebooks/             # (Optional) Jupyter notebooks
├── src/                   # Source code: data loading, model, training, evaluation
│   ├── __init__.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── model.py
│   └── train.py
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 🚂 Training the Model

```bash
python -m src.train
```

- Model saved to: `outputs/cnn_catdog_model.h5`
- Training plot: `outputs/training_plot.png`

---

## 🧪 Evaluating the Model

```bash
python -m src.evaluate
```

---

## 📌 Notes

- This project uses a basic CNN with Keras.
- Designed for binary classification: cat vs. dog.
- Data should be structured under `data/train/` and `data/test/` folders.

---

## 🙋‍♂️ Contact

Maintained by [@Almehedi06](https://github.com/Almehedi06)
