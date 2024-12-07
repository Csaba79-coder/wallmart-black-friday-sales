import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class EDAChecker:
    def __init__(self, file_path):
        """
        Inicializálja az objektumot és betöltési útvonalat ad meg.
        """
        self.file_path = file_path
        self.data = None  # Adat betöltéséhez üres attribútum

    def load_data(self):
        """
        Adatok betöltése a megadott útvonalról.
        """
        try:
            self.data = pd.read_csv(self.file_path)
            print("✅ Adatok betöltve.")
        except Exception as e:
            print(f"❌ Hiba az adatok betöltésekor: {e}")

    def basic_info(self):
        """
        Alapinformáció az adathalmazról.
        """
        print("\n🗂️ Alapinformációk:")
        print(self.data.head(10).to_string(index=False))
        print("\n📊 Adathalmaz információ:")
        print(self.data.info())
        print("\n🔎 Hiányzó értékek száma:")
        print(self.data.isnull().sum())

    def plot_missing_values(self):
        """
        Hiányzó értékek ábrázolása heatmappal.
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.data.isnull(), cbar=False, cmap="viridis")
        plt.title("❌ Hiányzó értékek ábrázolása")
        plt.show()

    def plot_distribution(self, column):
        """
        Általános eloszlások ábrázolása az adott oszlopra.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data[column].dropna(), kde=True)
        plt.title(f"📊 Oszlop eloszlása: {column}")
        plt.xlabel(column)
        plt.ylabel('Gyakoriság')
        plt.show()

    def plot_correlation(self):
        """
        Korreláció ábrázolása oszlopok között.
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.data.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title("🔗 Oszlopok közötti korreláció")
        plt.show()

    def feature_summary(self):
        """
        Statisztikai összegzés numerikus oszlopokra.
        """
        print("\n📊 Numerikus oszlopok statisztikai összegzése:")
        print(self.data.describe())

    def run_all(self):
        """
        Az összes elemzési funkció egyszerre futtatása.
        """
        self.load_data()
        self.basic_info()
        self.plot_missing_values()
        self.feature_summary()
        self.plot_correlation()

        # Általános eloszlások ábrázolása numerikus oszlopokra
        for col in self.data.select_dtypes(include=["float64", "int64"]).columns:
            self.plot_distribution(col)


# Használati példa
if __name__ == "__main__":
    # Dynamikus útvonal számítása a szkript helyéről
    script_dir = os.path.dirname(os.path.abspath(__file__))  # A szkript helyének elérési útja
    file_path = os.path.join(script_dir, "..", "classification", "output", "cleaned_final_walmart_data.csv")

    print("Ellenőrzendő útvonal:", file_path)

    eda_checker = EDAChecker(file_path)  # Objektum létrehozása
    eda_checker.run_all()  # Teljes elemzés futtatása