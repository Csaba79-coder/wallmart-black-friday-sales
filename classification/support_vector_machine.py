from classification.final_data_load import load_data, split_data
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score


def support_vector_machine():
    """
    Egyetlen metódus, ami betölti az adatokat, három részre osztja őket,
    majd Support Vector Machine (SVM) modellt hajt végre a Gender előrejelzésére.
    """
    # Adatok betöltése
    data_sample = load_data()

    if data_sample is not None:
        # Osztás három részre
        training_data, validation_data, test_data = split_data(data_sample)

        # Features és target változók definiálása
        features = ['Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Product_Category', 'Purchase']
        target = 'Gender'

        # Készítsük el az X (features) és y (target) változókat
        X_train = training_data[features]
        y_train = training_data[target]

        X_val = validation_data[features]
        y_val = validation_data[target]

        X_test = test_data[features]
        y_test = test_data[target]

        # Adatok skálázása
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Modell létrehozása és betanítása (lineáris kernel-el)
        model = SVC(kernel='linear', probability=True)  # Support Vector Machine létrehozása
        model.fit(X_train_scaled, y_train)  # Modell betanítása

        # Validáció
        y_val_pred = model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_report = classification_report(y_val, y_val_pred)
        y_val_pred_prob = model.predict_proba(X_val_scaled)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_pred_prob)

        # Teszt értékelése
        y_test_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_report = classification_report(y_test, y_test_pred)
        y_test_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_pred_prob)

        # Pontosságok és riportok kiírása
        print("\nValidációs pontosság: ")
        print(val_accuracy)
        print("\nValidációs riport: ")
        print(val_report)
        print("\nValidáció AUC értéke:", val_auc)

        print("\nTeszt szett pontossága: ")
        print(test_accuracy)
        print("\nTeszt szett riport: ")
        print(test_report)
        print("\nTeszt szett AUC értéke:", test_auc)


if __name__ == "__main__":
    support_vector_machine()
