import os
import pandas as pd


def load_data():
    """
    Betölti az adatokat és visszaadja az első 10 sort.
    """
    try:
        # Aktuális szkript elérési útjának megkeresése
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "output", "final_without_outlier_walmart_data.csv")

        print(f"Próbálkozás az alábbi elérési útvonalról: {file_path}")

        # Ellenőrizzük, hogy a file valóban létezik
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fájl nem található az alábbi elérési úton: {file_path}")

        # Betöltés
        data = pd.read_csv(file_path)

        # Visszaadjuk az első 10 sort
        print("✅ Fájl sikeresen beolvasva.")
        print("Első 10 sor az adatokból:")
        print(data.head(10))

        return data

    except Exception as e:
        print(f"⚠️ Hiba a fájl beolvasása során: {e}")
        return None


def split_data(data):
    """
    Adatokat 70-10-20 arányban osztja szét.
    """
    # Számítsuk ki a részarányokat az adatok hosszának megfelelően
    total_len = len(data)

    # Indexek kiszámítása az arányok alapján
    split1_end = int(total_len * 0.7)  # 70% - az első rész végének indexe
    split2_end = int(total_len * 0.8)  # 10% + 20% arány alapján

    # Osztás az indexek segítségével
    split1 = data.iloc[:split1_end]
    split2 = data.iloc[split1_end:split2_end]
    split3 = data.iloc[split2_end:]

    return split1, split2, split3


if __name__ == "__main__":
    # Betöltjük az adatokat
    data_sample = load_data()

    if data_sample is not None:
        # Három részre osztás 70-10-20 arány szerint
        part1, part2, part3 = split_data(data_sample)

        # Adatok számának kiíratása
        print("\nAdatok számának kiíratása:")
        print(f"Összes adat: {len(data_sample)}")
        print(f"Első rész hossza (70%): {len(part1)}")
        print(f"Második rész hossza (10%): {len(part2)}")
        print(f"Harmadik rész hossza (20%): {len(part3)}")
