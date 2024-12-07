import matplotlib.pyplot as plt
import pandas as pd
import os


def load_data():
    """
    Dynamically load data from the relative path.
    """
    try:
        # Set file path dynamically
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "..", "classification", "output", "final_without_outlier_walmart_data.csv")

        # Load the data
        data = pd.read_csv(file_path)
        print(f"✅ File successfully loaded: {file_path}")

        # Define target variable
        y = data['Gender']
        return y
    except Exception as e:
        print(f"Error while loading data: {e}")


def plot_class_distribution(y):
    """
    Visualize the class distribution.
    """
    try:
        # Plot the distribution
        plt.figure(figsize=(8, 6))
        plt.hist(y, bins=2, edgecolor='black')
        plt.title("Class Distribution (y)")
        plt.xlabel("Class")
        plt.ylabel("Frequency")
        plt.xticks([0, 1], ['Class 0', 'Class 1'])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
        print("✅ Class distribution successfully visualized.")
    except Exception as e:
        print(f"Error during visualization: {e}")


if __name__ == "__main__":
    # Load the data
    y = load_data()

    # Plot class distribution
    plot_class_distribution(y)
