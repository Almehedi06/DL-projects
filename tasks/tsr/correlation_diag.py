# tsr/correlation_diag.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_diagnostic(csv_path, output_path=None):
    """
    Load CSV, print correlation matrix, and plot heatmap.

    Args:
        csv_path (str): Path to the CSV file.
        output_path (str): Optional. Path to save heatmap image.
    """
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df.drop(columns=['Date'], errors='ignore')

    # Compute and print correlation matrix
    corr = df.corr()
    print("\n=== Correlation Matrix ===")
    print(corr)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Feature Correlation Heatmap")

    # Save or show
    if output_path:
        plt.savefig(output_path)
        print(f"✅ Correlation heatmap saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    csv_path = "data/Microsoft_Stock.csv"  # Adjust if needed
    output_image = "outputs/correlation_heatmap.png"
    correlation_diagnostic(csv_path, output_path=output_image)
