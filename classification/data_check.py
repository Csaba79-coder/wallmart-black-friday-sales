import pandas as pd
import os


def create_output_folder():
    """Létrehoz egy üres output mappát, ha még nem létezik, a szülőkönyvtárban."""
    # Szülőkönyvtárból létrehozni az output mappát
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../output")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Üres 'output' mappa létrehozva az alábbi helyen: {output_dir}")
    else:
        print("'output' mappa már létezik.")


def analyze_data(file_path):
    """Adatok ellenőrzése az általad megadott lépések alapján."""
    # Adatok betöltése
    data = pd.read_csv(file_path)

    # Oszlopok szűrése az ellenőrzéshez
    columns_to_check = [col for col in data.columns if col not in ['User_ID', 'Product_ID']]

    # Egyedi értékek lekérése
    for col in columns_to_check:
        print(f"{col}:")
        print(data[col].unique())
        print("-" * 40)

    # Ellenőrizzük, hogy a 'Purchase' tartalmaz-e nem egész számokat vagy lebegőpontos értékeket
    non_integer_purchase = data[~data['Purchase'].astype(str).str.isnumeric()]['Purchase'].unique()

    print("\nNem egész számok vagy lebegőpontos számok a 'Purchase' oszlopban:")
    print(non_integer_purchase)

    # Ellenőrizzük a hiányzó adatokat
    missing_data_rows = data[data.isnull().any(axis=1)]

    # Ellenőrizzük a sorazonos duplikációkat
    exact_duplicates = data[data.duplicated(keep=False)]

    # Hiányzó adatok kiíratása
    print("\nHiányzó adatok ellenőrzése:")
    if missing_data_rows.empty:
        print("Nincs hiányzó adat az adatokban.")
    else:
        print("Hiányzó adatokkal rendelkező sorok:")
        print(missing_data_rows)

    # Duplikációk ellenőrzése
    print("\nDuplikációk ellenőrzése:")
    if exact_duplicates.empty:
        print("Nincs sor duplikáció az adatok között.")
    else:
        print("Sor duplikációk:")
        print(exact_duplicates.to_string(index=False))


if __name__ == "__main__":
    # Fájl elérési út
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../input/walmart_data.csv")

    # Output mappa létrehozása
    create_output_folder()

    # Adatok ellenőrzése
    analyze_data(file_path)
