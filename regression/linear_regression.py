from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from regression.load_data import load_data, split_data


def regression_model():
    """
    Lineáris regressziós modell létrehozása és kiértékelése.
    """
    # Adatok beolvasása
    data_sample = load_data()

    if data_sample is not None:
        # Osztás három részre
        training_data, validation_data, test_data = split_data(data_sample)

        # Features és target változók definiálása
        features = [
            'Gender',
            'Age',
            'Occupation',
            'City_Category',
            'Stay_In_Current_City_Years',
            'Marital_Status',
            'Product_Category'
        ]
        target = 'Purchase'

        # Adatok előkészítése
        X_train = training_data[features]
        y_train = training_data[target]
        X_val = validation_data[features]
        y_val = validation_data[target]
        X_test = test_data[features]
        y_test = test_data[target]

        # Skálázás 0 és 1 közötti értékekre
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=['float64', 'int64']))
        X_val_scaled = scaler.transform(X_val.select_dtypes(include=['float64', 'int64']))
        X_test_scaled = scaler.transform(X_test.select_dtypes(include=['float64', 'int64']))

        print("Eredeti adatok:")
        print(X_train.head())  # Az adatok az osztás után

        print("\nSkálázott adatok:")
        print(X_train_scaled[:10])  # Az első 10 skálázott adatpont

        # Lineáris regresszió modell létrehozása
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        # Validáció
        y_val_pred = model.predict(X_val_scaled)
        val_mse = mean_squared_error(y_val, y_val_pred)
        print(f"Validation MSE: {val_mse}")

        # Teszt eredmények
        y_test_pred = model.predict(X_test_scaled)
        test_mse = mean_squared_error(y_test, y_test_pred)
        print(f"Test MSE: {test_mse}")

        # Modell összegzése
        print("\nModel Coefficients:")
        print(model.coef_)
        print("\nModel Intercept:")
        print(model.intercept_)

        # Eredmény visszatérítése
        return {
            "model": model,
            "val_mse": val_mse,
            "test_mse": test_mse,
            "coefficients": model.coef_,
            "intercept": model.intercept_
        }


# Important answer is the Model Coefficients! [ 0.02292213  0.03196575  0.00634823  0.03443631  0.00104801 -0.00224058 -0.35396522]
# Explanation: 0.0229 (Gender): A modell szerint a nem (Gender) kis mértékben pozitívan befolyásolja a vásárlás valószínűségét.
# 0.0344 (City_Category): Az adott városi kategória magas értéket ad az előrejelzésnek.
# -0.3540 (Product_Category): Az érték negatív, ami azt jelenti, hogy egy adott termékkategória esetén a vásárlás valószínűsége alacsonyabb.
# Intercept: 0.466 -> Ez az alapvető valószínűségi érték, amit minden egyes előrejelzéshez a bemenetek mellett hozzáad a modell.
if __name__ == "__main__":
    print("🔧 Regressziós modell futtatása...")
    result = regression_model()
    if result:
        print("\nModell eredményei:")
        print(result)