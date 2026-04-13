import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


class DataAnalytics:
    def __init__(self, path):
        self.path = path
        self.df = None

    # 1. Load Data
    def load_data(self):
        self.df = pd.read_csv(self.path)
        print("Data Loaded Successfully")
        print(self.df.head(0))

    # 2. Basic Info
    def explore_data(self):
        print("\n--- DATA INFO ---")
        print(self.df.info())
        print("\n--- DESCRIBE ---")
        print(self.df.describe())

    # 3. Data Cleaning
    def clean_data(self):
        print("\n--- CLEANING DATA ---")

        # Remove duplicates
        self.df = self.df.drop_duplicates()

        # Handle missing values
        print(self.df.isnull().sum())
        self.df = self.df.dropna()

        # Parse mixed date strings (for example 15/04/2018) safely.
        self.df["Order Date"] = pd.to_datetime(
            self.df["Order Date"],
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )
        self.df["Ship Date"] = pd.to_datetime(
            self.df["Ship Date"],
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )

        print("Cleaning Done")

    # 4. Feature Engineering
    def feature_engineering(self):
        print("\n--- FEATURE ENGINEERING ---")

        if self.df is None:
            raise ValueError("Data is not loaded. Call load_data() first.")

        if "Order Date" in self.df.columns:
            self.df["Order Date"] = pd.to_datetime(
                self.df["Order Date"],
                format="mixed",
                dayfirst=True,
                errors="coerce",
            )

        if "Ship Date" in self.df.columns:
            self.df["Ship Date"] = pd.to_datetime(
                self.df["Ship Date"],
                format="mixed",
                dayfirst=True,
                errors="coerce",
            )

        if "Order Date" in self.df.columns:
            self.df["Month"] = self.df["Order Date"].dt.month

        if {"Profit", "Sales"}.issubset(self.df.columns):
            self.df["Profit Ratio"] = self.df["Profit"] / self.df["Sales"]
        else:
            print("Skipping Profit Ratio feature: required columns are missing.")

        if {"Ship Date", "Order Date"}.issubset(self.df.columns):
            self.df["Shipping Days"] = (self.df["Ship Date"] - self.df["Order Date"]).dt.days
        else:
            print("Skipping Shipping Days feature: required columns are missing.")

        print("New Features Created")

    # 5. Missing Postal Code Pattern Check
    def check_missing_postal_code_pattern(self):
        print("\n--- MISSING POSTAL CODE PATTERN CHECK ---")

        if self.df is None:
            raise ValueError("Data is not loaded. Call load_data() first.")

        working_df = self.df.copy()

        if "Order Date" in working_df.columns:
            working_df["Order Date"] = pd.to_datetime(
                working_df["Order Date"],
                format="mixed",
                dayfirst=True,
                errors="coerce",
            )

        missing_mask = working_df["Postal Code"].isna() | (working_df["Postal Code"].astype(str).str.strip() == "")
        missing_rows = working_df.loc[missing_mask].copy()

        if missing_rows.empty:
            print("No missing Postal Code values found.")
            return {
                "missing_count": 0,
                "by_date": pd.DataFrame(),
                "by_location": pd.DataFrame(),
                "by_product": pd.DataFrame(),
            }

        print(f"Missing Postal Code rows: {len(missing_rows)}")

        if "Order Date" in missing_rows.columns:
            missing_rows["Order Day"] = missing_rows["Order Date"].dt.date
            by_date = (
                missing_rows.groupby("Order Day")
                .size()
                .reset_index(name="Missing Count")
                .sort_values("Missing Count", ascending=False)
            )
            print("\nMissing Postal Code by Date:\n", by_date)
        else:
            by_date = pd.DataFrame()

        location_cols = [col for col in ["Country", "Region", "State", "City"] if col in missing_rows.columns]
        if location_cols:
            by_location = (
                missing_rows.groupby(location_cols)
                .size()
                .reset_index(name="Missing Count")
                .sort_values("Missing Count", ascending=False)
            )
            print("\nMissing Postal Code by Location:\n", by_location)
        else:
            by_location = pd.DataFrame()

        if "Product Name" in missing_rows.columns:
            by_product = (
                missing_rows.groupby("Product Name")
                .size()
                .reset_index(name="Missing Count")
                .sort_values("Missing Count", ascending=False)
            )
            print("\nMissing Postal Code by Product:\n", by_product)
        else:
            by_product = pd.DataFrame()

        return {
            "missing_count": len(missing_rows),
            "by_date": by_date,
            "by_location": by_location,
            "by_product": by_product,
        }

    # 5. Advanced Analysis
    def analysis(self):
        print("\n--- ANALYSIS ---")

        # Multi-dimensional analysis
        result = self.df.groupby(["Category", "Region"])["Profit"].sum()
        print("\nProfit by Category & Region:\n", result)

        # Correlation
        print("\nCorrelation:\n", self.df[["Sales", "Profit", "Discount"]].corr())

        # Percentiles
        print("\nSales Percentiles:\n", self.df["Sales"].quantile([0.25, 0.5, 0.75, 0.9]))

        # Outliers
        outliers = self.df[self.df["Sales"] > self.df["Sales"].quantile(0.95)]
        print("\nOutliers Count:", len(outliers))

    # 6. Visualization
    def visualize(self):
        print("\n--- CREATING VISUALIZATIONS ---")

        os.makedirs("visualization_images", exist_ok=True)

        # Sales by Category
        self.df.groupby("Category")["Sales"].sum().plot(kind="bar")
        plt.title("Sales by Category")
        plt.xlabel("Category")
        plt.ylabel("Sales")
        plt.tight_layout()
        plt.savefig("visualization_images/sales_by_category.png", dpi=300)
        plt.clf()

        # Profit vs Discount
        if {"Discount", "Profit"}.issubset(self.df.columns):
            sns.scatterplot(x="Discount", y="Profit", data=self.df)
            plt.title("Profit vs Discount")
            plt.tight_layout()
            plt.savefig("visualization_images/profit_vs_discount.png", dpi=300)
            plt.clf()
        else:
            print("Skipping Profit vs Discount plot: required columns are missing.")

        # Correlation Heatmap
        corr_cols = [col for col in ["Sales", "Profit", "Discount"] if col in self.df.columns]
        if len(corr_cols) >= 2:
            sns.heatmap(self.df[corr_cols].corr(), annot=True)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.savefig("visualization_images/correlation_heatmap.png", dpi=300)
            plt.clf()
        else:
            print("Skipping Correlation Heatmap: not enough numeric columns available.")

        # Monthly Sales Trend
        if "Month" not in self.df.columns and "Order Date" in self.df.columns:
            self.df["Month"] = pd.to_datetime(
                self.df["Order Date"],
                format="mixed",
                dayfirst=True,
                errors="coerce",
            ).dt.month

        if {"Month", "Sales"}.issubset(self.df.columns):
            self.df.groupby("Month")["Sales"].sum().plot()
            plt.title("Monthly Sales Trend")
            plt.xlabel("Month")
            plt.ylabel("Sales")
            plt.tight_layout()
            plt.savefig("visualization_images/monthly_sales.png", dpi=300)
            plt.clf()
        else:
            print("Skipping Monthly Sales Trend: required columns are missing.")

        print("Visualizations saved in /visualization_images folder")



# MAIN EXECUTION
# if __name__ == "__main__":
    # path = "train.csv"  # change path if needed

    # project = DataAnalytics(path)

    # project.load_data()
    # project.explore_data()
    # project.clean_data()
    # project.feature_engineering()
    # project.analysis()
    # project.visualize()

path = "asset/train_with_profit_discount.csv"
k = DataAnalytics(path)
k.load_data()
k.explore_data()
# k.check_missing_postal_code_pattern()
# k.clean_data()

k.feature_engineering()
k.analysis()
k.visualize()
# k.feature_engineering()
# k.check_missing_postal_code_pattern()

