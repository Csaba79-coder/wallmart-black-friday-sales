import pandas as pd
import os


def create_output_folder():
    """Létrehoz egy üres output mappát, ha még nem létezik, a szülőkönyvtárban."""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../output")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Üres 'output' mappa létrehozva az alábbi helyen: {output_dir}")
    else:
        print("'output' mappa már létezik.")

    return output_dir


def analyze_correlation(data):
    """Korreláció elemzése a Gender és Marital_Status oszlopok között."""

    # 1. Nem numerikus adatok átalakítása
    # Átalakítjuk a nemeket numerikus értékekre
    data['Gender'] = data['Gender'].map({'F': 0, 'M': 1})

    # 2. Kiválasztjuk a korrelációhoz szükséges oszlopokat
    correlation_data = data[['Gender', 'Marital_Status']]

    # 3. Korrelációs mátrix számítása
    correlation_matrix = correlation_data.corr()

    # 4. Mentési útvonalak beállítása
    output_dir = create_output_folder()
    correlation_file = os.path.join(output_dir, "result_gender_marital_status_correlation.csv")
    processed_data_file = os.path.join(output_dir, "cleaned_walmart_data_gender_marital_status_correlation.csv")

    # 5. Korrelációs mátrix mentése CSV-be
    correlation_matrix.to_csv(correlation_file, index=True)
    print(f"Korrelációs mátrix mentve: {correlation_file}")

    # 6. Feldolgozott adatok mentése CSV-be
    data.to_csv(processed_data_file, index=False)
    print(f"Feldolgozott adatok mentve: {processed_data_file}")

    # 7. Korrelációs mátrix kiírása
    print("Korrelációs mátrix:")
    print(correlation_matrix)

    # 8. Következtetések elemzése
    correlation_value = correlation_matrix.loc['Gender', 'Marital_Status']
    if correlation_value == 0:
        print("\nNincs korreláció a Gender és a Marital_Status között.")
    else:
        print("\nVan korreláció a Gender és a Marital_Status között, elemzés szükséges!")


if __name__ == "__main__":
    # Fájl elérési út
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../input/walmart_data.csv")

    # Adatok betöltése
    try:
        data = pd.read_csv(file_path)
        print(f"Adatok sikeresen beolvassa: {file_path}")
    except FileNotFoundError:
        print(f"Hiba! Az alábbi fájl nem található: {file_path}")
        exit(1)

    # Korreláció elemzése
    analyze_correlation(data)
