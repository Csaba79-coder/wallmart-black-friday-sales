from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from classification.data_check import create_output_folder


def remove_outliers(data, column):
    """
    Outlierek eltávolítása az IQR módszer alapján.
    Csak azok az értékek maradnak, amelyek nem outlierek.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    print(f"Outlierek eltávolítva az oszlopból: {column}")
    return filtered_data


def clean_and_scale_data(file_path):
    """Adattisztítás és skálázás elvégzése az adatokra."""

    # 1. Adatok beolvasása
    try:
        data = pd.read_csv(file_path)
        print(f"Adatok sikeresen beolvassa: {file_path}")
    except FileNotFoundError:
        print(f"Hiba! Az alábbi fájl nem található: {file_path}")
        exit(1)

    # 2. Eltávolítjuk a NaN értékkel rendelkező sorokat
    data = remove_rows_with_missing_values(data)

    # 3. Elrelevant oszlopok eltávolítása
    columns_to_drop = ['User_ID', 'Product_ID']
    data.drop(columns=columns_to_drop, axis=1, inplace=True)
    print("Irreleváns oszlopok eltávolítva.")

    # 4. Gender átalakítása: 'F'=0, 'M'=1
    data['Gender'] = data['Gender'].map({'F': 0, 'M': 1})
    print("Gender értékei numerikus formára átalakítva.")

    # 5. City_Category szkeálzása numerikus értékekre
    data['City_Category'] = data['City_Category'].map({'A': 0.0, 'B': 0.5, 'C': 1.0})
    print("City_Category értékei numerikus formára átalakítva.")

    # 6. Az új 'Stay_In_Current_City_Years' értékek számszerűsítése
    data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].replace({
        '4+': 4, '3': 3, '2': 2, '1': 1, '0': 0
    })
    print("Stay_In_Current_City_Years értékei számszerűsítve.")

    # 7. Age oszlop szkeálzása tartományba
    age_mapping = {
        '0-17': 0,
        '18-25': 18,
        '26-35': 26,
        '36-45': 36,
        '46-50': 46,
        '51-55': 51,
        '55+': 55
    }
    data['Age'] = data['Age'].map(age_mapping)
    print("Age oszlop értékei átalakítva.")

    # 8. Outlierek eltávolítása a 'Purchase' oszlopból
    data = remove_outliers(data, "Purchase")

    # 9. MinMaxScaler skálázás
    scaler = MinMaxScaler()
    numerical_features = [
        'Age', 'Occupation', 'Stay_In_Current_City_Years', 'Product_Category', 'Purchase'
    ]
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    print("Numerikus adatok skálázása megtörtént.")

    # 10. Mentési mappa létrehozása
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)  # Létrehozza a mappát, ha nem létezik
    cleaned_file_path = os.path.join(output_dir, "cleaned_final_regression_walmart_data.csv")

    # Mentés
    data.to_csv(cleaned_file_path, index=False)
    print(f"Adatok tisztítva, skálázva, outlierek eltávolítva, és mentve az alábbi helyre: {cleaned_file_path}")


def remove_rows_with_missing_values(data):
    """
    Függvény, amely eltávolítja azokat a sorokat, ahol bármelyik oszlop értéke NaN (üres).
    """
    # Ellenőrizzük és eltávolítjuk a NaN értékeket tartalmazó sorokat
    initial_rows = data.shape[0]
    data = data.dropna()
    final_rows = data.shape[0]
    print(f"Eltávolított sorok száma: {initial_rows - final_rows}")
    return data


if __name__ == "__main__":
    # Input fájl elérési út
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../input/walmart_data.csv")

    # Tisztítás, skálázás, outlierek eltávolítása és mentés
    clean_and_scale_data(file_path)
