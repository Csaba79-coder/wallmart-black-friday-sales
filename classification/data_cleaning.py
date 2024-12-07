from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

from classification.data_check import create_output_folder


def clean_and_scale_data(file_path):
    """Adattisztítás és skálázás elvégzése az adatokra."""

    # 1. Adatok beolvasása
    try:
        data = pd.read_csv(file_path)
        print(f"Adatok sikeresen beolvassa: {file_path}")
    except FileNotFoundError:
        print(f"Hiba! Az alábbi fájl nem található: {file_path}")
        exit(1)

    # 2. Elrelevant oszlopok eltávolítása
    columns_to_drop = ['User_ID', 'Product_ID', 'Marital_Status']
    data.drop(columns=columns_to_drop, axis=1, inplace=True)
    print("Irreleváns oszlopok eltávolítva.")

    # 3. Gender átalakítása: 'F'=0, 'M'=1
    data['Gender'] = data['Gender'].map({'F': 0, 'M': 1})
    print("Gender értékei numerikus formára átalakítva.")

    # 4. City_Category szkeálzása numerikus értékekre
    data['City_Category'] = data['City_Category'].map({'A': 0.0, 'B': 0.5, 'C': 1.0})
    print("City_Category értékei numerikus formára átalakítva.")

    # 5. Az új 'Stay_In_Current_City_Years' értékek számszerűsítése
    data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].replace({
        '4+': 4, '3': 3, '2': 2, '1': 1, '0': 0
    })
    print("Stay_In_Current_City_Years értékei számszerűsítve.")

    # 6. Age oszlop szkeálzása tartományba
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

    # 7. MinMaxScaler skálázás
    scaler = MinMaxScaler()
    numerical_features = [
        'Age', 'Occupation', 'Stay_In_Current_City_Years', 'Product_Category', 'Purchase'
    ]
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    print("Numerikus adatok skálázása megtörtént.")

    # 8. Mentési mappa létrehozása
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)  # Létrehozza a mappát, ha nem létezik
    cleaned_file_path = os.path.join(output_dir, "cleaned_final_walmart_data.csv")

    # Mentés
    data.to_csv(cleaned_file_path, index=False)
    print(f"Adatok tisztítva, skálázva és mentve az alábbi helyre: {cleaned_file_path}")


if __name__ == "__main__":
    # Input fájl elérési út
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../input/walmart_data.csv")

    # Tisztítás, skálázás, mentés
    clean_and_scale_data(file_path)
