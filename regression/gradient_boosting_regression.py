from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

from regression.load_data import load_data, split_data


def regression_model():
    """
    Gradient Boosting Regressziós modell létrehozása és kiértékelése.
    """
    # Adatok beolvasása
    data_sample = load_data()

    # Korrelációs mátrix ellenőrzése
    correlation_matrix = data_sample.corr()
    sns.heatmap(correlation_matrix, annot=True)
    plt.show()

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

        # Gradient Boosting Regressziós modell létrehozása
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
        model.fit(X_train_scaled, y_train)

        # Validáció
        y_val_pred = model.predict(X_val_scaled)
        val_mse = mean_squared_error(y_val, y_val_pred)
        print(f"Validation MSE: {val_mse}")

        # Teszt eredmények
        y_test_pred = model.predict(X_test_scaled)
        test_mse = mean_squared_error(y_test, y_test_pred)
        print(f"Test MSE: {test_mse}")

        # Feature importances ellenőrzése
        print("\nModel Feature Importances:")
        print(model.feature_importances_)

        # Visszatérítés az eredményekkel
        return {
            "model": model,
            "val_mse": val_mse,
            "test_mse": test_mse,
            "feature_importances": model.feature_importances_
        }


if __name__ == "__main__":
    print("🔧 Gradient Boosting Regressziós modell futtatása...")
    result = regression_model()
    if result:
        print("\nModell eredményei:")
        print(result)
