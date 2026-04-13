# CSEC Final Data Analytics Project

## Overview
This project runs a simple end-to-end sales analytics workflow using pandas, seaborn, and matplotlib.

The main script in main.py does the following:
- Loads CSV data from asset/train_with_profit_discount.csv
- Prints dataset info and descriptive statistics
- Cleans data (drops duplicates, drops rows with missing values, parses mixed date formats)
- Creates features such as Month, Profit Ratio, and Shipping Days
- Runs summary analysis (profit by category and region, correlations, sales percentiles, outlier count)
- Generates visualization images in visualization_images/

## Project Structure
- main.py: pipeline entry point and DataAnalytics class
- asset/: input CSV files
- visualization_images/: saved charts
- requirements.txt: currently empty (dependencies listed below)

## Prerequisites
- Python 3.10+ (3.14 also works)
- Git

## Clone the Repository
1. Open a terminal.
2. Run:

```bash
git clone <repository-url>
cd CSEC_final
```

Replace <repository-url> with your repository URL.

## Set Up Environment
1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate it:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project
From the project root:

```bash
python main.py
```

## Output
After a successful run:
- Console shows loading, cleaning, feature engineering, and analysis summaries.
- Charts are written to visualization_images/, such as:
  - sales_by_category.png
  - profit_vs_discount.png (if Discount and Profit columns exist)
  - correlation_heatmap.png (if enough numeric columns exist)
  - monthly_sales.png

## Notes
- The script currently points to asset/train_with_profit_discount.csv in the main block.
- If your dataset path is different, update the path variable near the bottom of main.py.
- If some columns are missing (for example Discount or Profit), the script skips dependent charts/features and continues.
