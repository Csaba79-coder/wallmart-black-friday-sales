import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve


# Adatok betöltése és előkészítése
def load_and_prepare_data():
    """
    Dinamikusan betölti az adatokat a relatív útvonal alapján.
    """
    try:
        # Dinamikus elérési út beállítása
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "..", "classification", "output", "final_without_outlier_walmart_data.csv")

        # Adatok betöltése
        data = pd.read_csv(file_path)
        print(f"✅ Fájl sikeresen beolvasva: {file_path}")

        # Célváltozó létrehozása (Gender -> Target)
        data['Target'] = data['Gender']

        # Független változók
        features = ['Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Product_Category', 'Purchase']
        X = data[features]
        y = data['Target']

        # Kezeljük a NaN értékeket
        X = X.dropna(subset=features)
        y = y.loc[X.index]

        print("✅ Adatok előkészítve.")
        return X, y
    except Exception as e:
        print(f"Hiba adatok betöltése során: {e}")


# Modell szétbontása
def split_data(X, y):
    """
    Az adatok szétbontása 70% training, 10% validáció, 20% teszt szettekre.
    """
    try:
        # Szétbontás: 70%-30%
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        validation_fraction = 0.1 / (0.7 + 0.1)  # Normalizáljuk a validációs arányt
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=validation_fraction,
                                                          random_state=42)

        print("✅ Adatok szétbontva: Train, Validation, Teszt szettek.")
        return X_train, X_val, X_test, y_train, y_val, y_test
    except Exception as e:
        print(f"Hiba szétbontás során: {e}")


# Logistic Regression modell tanítása
def train_model(X_train, y_train):
    """
    Tanítjuk a Logistic Regression modellt.
    """
    try:
        # Skálázás
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Modell létrehozása és tanítása
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        print("✅ Modell sikeresen tanítva.")
        return model, scaler
    except Exception as e:
        print(f"Hiba a modell tanítása során: {e}")


# ROC görbe ábrázolása
def plot_roc_curve(model, scaler, X_test, y_test):
    """
    ROC görbe ábrázolása a teszt szetten.
    """
    try:
        # Skálázás
        X_test_scaled = scaler.transform(X_test)

        # Predikciók kiszámítása
        y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

        # ROC görbe értékei
        fpr, tpr, _ = roc_curve(y_test, y_test_prob)

        # Ábrázolás
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC Curve')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)  # Diagonális referencia
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Görbe')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

        print("✅ ROC görbe megjelenítve.")
    except Exception as e:
        print(f"Hiba az ROC görbe ábrázolásában: {e}")


# Modell értékelése
def evaluate_model(model, scaler, X_val, y_val, X_test, y_test):
    """
    A validációs és teszt szetteken értékeljük a modellt.
    """
    try:
        # Skálázás validációra és teszt adatokra
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Értékelés validációs szettel
        y_val_pred = model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_auc = roc_auc_score(y_val, model.predict_proba(X_val_scaled)[:, 1])

        # Értékelés teszt szettel
        y_test_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

        # Kiértékelési riport
        print("\n✅ **Validációs szett eredményei:**")
        print(f"Pontosság: {val_accuracy:.4f}")
        print(f"AUC: {val_auc:.4f}")
        print("\n✅ **Teszt szett eredményei:**")
        print(f"Pontosság: {test_accuracy:.4f}")
        print(f"AUC: {test_auc:.4f}")

        print("\n📊 **Classification Report:**")
        print("\n", classification_report(y_test, y_test_pred))

        # Ábrázoljuk az ROC görbét
        plot_roc_curve(model, scaler, X_test, y_test)

    except Exception as e:
        print(f"Hiba az értékelés során: {e}")


if __name__ == "__main__":
    # Adatok betöltése
    X, y = load_and_prepare_data()
    # Adatok szétbontása
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    # Modell tanítása
    model, scaler = train_model(X_train, y_train)
    # Modell értékelése
    evaluate_model(model, scaler, X_val, y_val, X_test, y_test)
