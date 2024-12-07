import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class OutlierChecker:
    def __init__(self):
        """
        Alap szkripthez szükséges útvonal előkészítése.
        """
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(self.script_dir, "..", "classification", "output",
                                      "cleaned_final_walmart_data.csv")
        self.data = None
        self.outliers_detected = {}

    def load_data(self):
        """
        Adatok betöltése.
        """
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"✅ Adatok betöltve: {self.file_path}")
        except Exception as e:
            print(f"❌ Hiba adatok betöltésekor: {e}")

    def check_outliers(self):
        """
        Outlierek vizsgálata megadott oszlopokra.
        Kiszámolja, hogy hány outlier van és eltávolítja őket, ha van.
        """
        # Oszlopok az outlier vizsgálathoz
        columns_to_check = [
            'Age',
            'Occupation',
            'City_Category',
            'Stay_In_Current_City_Years',
            'Product_Category',
            'Purchase'
        ]

        # For each column, identify outliers using IQR method
        for column in columns_to_check:
            Q1 = self.data[column].quantile(0.25)  # 25. percentil
            Q3 = self.data[column].quantile(0.75)  # 75. percentil
            IQR = Q3 - Q1  # Interquartile range
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Detect outliers
            outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
            self.outliers_detected[column] = len(outliers)

            # Remove outliers
            self.data = self.data[~((self.data[column] < lower_bound) | (self.data[column] > upper_bound))]

        print("\n✅ Outlier-ek ellenőrzése és eltávolítása befejezve.")
        print("Outlier-ek száma oszloponként:")
        for col, num in self.outliers_detected.items():
            print(f" - {col}: {num} outlier eltávolítva.")

        # Menteni az outlier-ektől mentes adatokat
        outlier_free_file_path = os.path.join(self.script_dir, "..", "classification", "output", "final_without_outlier_walmart_data.csv")
        self.data.to_csv(outlier_free_file_path, index=False)
        print(f"\n✅ Outlier-ektől mentes adatok mentve: {outlier_free_file_path}")

    def plot_outliers(self):
        """
        Vizuális ábrázolás outlierek vizsgálatához.
        """
        plt.figure(figsize=(15, 10))
        for i, column in enumerate(
                ['Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Product_Category', 'Purchase']):
            ax = plt.subplot(3, 2, i + 1)
            sns.boxplot(x=self.data[column], ax=ax)  # Vizualizáció outlier-ekre
            plt.title(f'Outlier vizsgálat: {column}')
        plt.tight_layout()
        plt.show()
        print("\n✅ Outlier ábrázolva boxplotokkal.")

    def run(self):
        """
        Teljes folyamat: adatok betöltése, outlier-ellenőrzés, eltávolítás, ábrázolás és mentés.
        """
        self.load_data()
        self.check_outliers()
        self.plot_outliers()


# Szkripthez futtatási szempont
if __name__ == "__main__":
    checker = OutlierChecker()
    checker.run()
