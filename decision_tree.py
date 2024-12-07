import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score


def load_data():
    """
    Dynamically load data from the relative path.
    """
    try:
        # Aktuális szkript elérési útjának megkeresése
        script_dir = os.path.dirname(os.path.abspath(__file__))  # A decision_tree.py helye
        file_path = os.path.join(script_dir, "output", "final_without_outlier_walmart_data.csv")

        print(f"Trying to load from: {file_path}")  # Debug print

        # Ellenőrizzük, hogy az útvonal valóban létezik-e
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")

        # Betöltés
        data = pd.read_csv(file_path)
        print("✅ File successfully loaded.")
        print("Loaded data:")
        print(data.head())  # Debug info: nézd meg az adatokat

        # Ellenőrizzük az oszlopokat is
        if 'Gender' not in data.columns:
            raise ValueError("The 'Gender' column is missing in the data.")
        if any(col not in data.columns for col in ['Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Product_Category', 'Purchase']):
            raise ValueError("One or more expected feature columns are missing.")

        # Beállítjuk a target és feature változókat
        y = data['Gender']
        X = data[['Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Product_Category', 'Purchase']].dropna()

        # Debug info
        print("X shape:", X.shape)
        print("y shape:", y.shape)

        return X, y
    except Exception as e:
        print(f"Error loading the data: {e}")
        return None, None


def train_decision_tree(X_train, y_train):
    """
    Train a DecisionTreeClassifier on the training data.
    """
    try:
        # Initialize and train the Decision Tree Classifier
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
        print("✅ Decision tree trained successfully.")
        return clf
    except Exception as e:
        print(f"Error training the Decision Tree: {e}")


def evaluate_model(clf, X_test, y_test):
    """
    Evaluate the model performance using AUC and accuracy.
    """
    try:
        # Predict probabilities
        y_pred_prob = clf.predict_proba(X_test)[:, 1]

        # Predict classes
        y_pred = clf.predict(X_test)

        # Compute accuracy
        acc = accuracy_score(y_test, y_pred)

        # Compute AUC
        auc = roc_auc_score(y_test, y_pred_prob)

        # Print results
        print(f"✅ Model Accuracy: {acc:.4f}")
        print(f"✅ Model AUC Score: {auc:.4f}")

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve for Decision Tree")
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Error during model evaluation: {e}")


def visualize_decision_tree(clf, feature_names):
    """
    Visualize the Decision Tree structure.
    """
    try:
        plt.figure(figsize=(20, 10))
        plot_tree(clf, feature_names=feature_names, filled=True, class_names=["Class 0", "Class 1"], rounded=True)
        plt.show()
        print("✅ Decision tree visualization completed.")
    except Exception as e:
        print(f"Error visualizing Decision Tree: {e}")


if __name__ == "__main__":
    # Load the data
    X, y = load_data()

    # Split data into train/validation/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% train
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6667,
                                                    random_state=42)  # 10% validation, 20% test

    print(f"✅ Training set size: {X_train.shape}, {y_train.shape}")
    print(f"✅ Validation set size: {X_val.shape}, {y_val.shape}")
    print(f"✅ Test set size: {X_test.shape}, {y_test.shape}")

    # Train Decision Tree
    clf = train_decision_tree(X_train, y_train)

    # Evaluate model performance
    evaluate_model(clf, X_test, y_test)

    # Visualize the Decision Tree
    visualize_decision_tree(clf, X.columns)
