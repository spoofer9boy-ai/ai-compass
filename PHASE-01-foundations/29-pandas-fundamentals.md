# Pandas Fundamentals

**Phase:** PHASE-01-foundations  
**Prerequisites:** [28] NumPy Fundamentals  
**Estimated Time:** 60 minutes

## Why am I learning this?

You can do everything in NumPy. You can load CSVs, filter rows, compute group statistics, and join datasets using nothing but `np.loadtxt`, boolean indexing, and a lot of patience. But you will not. In production, in Kaggle notebooks, in research papers, and in every data pipeline you ever touch, the default tool for tabular data manipulation is pandas. It is not because pandas is faster—it often is not. It is because pandas gives you labeled axes, mixed data types, missing-value handling, and time-series indexing out of the box. These are not luxuries; they are necessities when your dataset has 40 columns with names like `customer_id`, `signup_date`, and `lifetime_value`, and you need to know which rows have null `signup_date` while grouping by `region`.

This file exists so that when you open a `.csv` for the first time, you know how to inspect it, clean it, reshape it, and extract the exact slice you need for your model. You will not become a pandas expert in one hour, but you will know the 20% of the API that handles 80% of real-world tasks.

## Where will I be using it?

- **Data Loading:** Reading CSV, Parquet, JSON, and SQL into structured DataFrames.
- **Exploratory Data Analysis (EDA):** Inspecting distributions, missing values, and correlations before modeling.
- **Feature Engineering:** Creating new columns, encoding categoricals, and normalizing values for downstream ML pipelines.
- **Time-Series Analysis:** Resampling, shifting, and rolling-window calculations on ordered data.
- **Data Validation:** Asserting schema constraints and catching data drift in production pipelines.

## Resources

- [pandas Official Documentation — 10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html) — The canonical quick-start from the core team.
- [Python Data Science Handbook — Data Manipulation with Pandas](https://jakevdp.github.io/PythonDataScienceHandbook/03.00-introduction-to-pandas.html) — Free, in-depth coverage of Series, DataFrames, and operations.
- [Real Python — Using pandas and Python to Explore Your Dataset](https://realpython.com/pandas-python-explore-dataset/) — Practical walkthrough with a real dataset.
- [Python for Data Analysis, 3E — Getting Started with pandas](https://wesmckinney.com/book/pandas-basics.html) — Written by the creator of pandas; the definitive reference.
- [pandas on GitHub](https://github.com/pandas-dev/pandas) — Source code, issue tracker, and release notes.

## Appendix

### Core Data Structures

- **Series:** A one-dimensional labeled array. Think of it as a NumPy array with an index.
  ```python
  import pandas as pd
  s = pd.Series([1, 3, 5, 7], index=['a', 'b', 'c', 'd'])
  ```
- **DataFrame:** A two-dimensional labeled data structure with columns of potentially different types. The workhorse of pandas.
  ```python
  df = pd.DataFrame({
      'name': ['Alice', 'Bob', 'Charlie'],
      'age': [25, 30, 35],
      'score': [85.5, 90.2, 88.0]
  })
  ```
- **Index:** Immutable sequence used for axis labels. Enables fast alignment, joins, and lookups.

### Common Operations Cheat Sheet

| Task | Code |
|------|------|
| Load CSV | `pd.read_csv('file.csv')` |
| First rows | `df.head(n)` |
| Summary stats | `df.describe()` |
| Select column | `df['col']` or `df.col` |
| Filter rows | `df[df['age'] > 25]` |
| Missing values | `df.isna().sum()` |
| Drop missing | `df.dropna()` |
| Fill missing | `df.fillna(0)` |
| Group and aggregate | `df.groupby('category')['value'].mean()` |
| Sort | `df.sort_values('age', ascending=False)` |
| Merge | `pd.merge(df1, df2, on='key')` |
| Pivot | `df.pivot_table(values='sales', index='month', columns='region', aggfunc='sum')` |

### Common Pitfalls

- **SettingWithCopyWarning:** Chained indexing (`df[df.A > 0]['B'] = 1`) can fail silently. Use `.loc` for assignment: `df.loc[df.A > 0, 'B'] = 1`.
- **NaN in integer columns:** pandas casts integer columns with missing values to float64. Use nullable `Int64` (capital I) dtype if you need true integer nulls.
- **Memory bloat:** Object dtype columns (strings) consume far more RAM than necessary. Consider `category` dtype for low-cardinality strings.
- **Index alignment:** Arithmetic operations align on index labels, not position. This is powerful but surprising if you expect positional behavior.

### Further Reading

- [pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html) — Comprehensive guides on every subsystem.
- [Effective Pandas](https://github.com/TomAugspurger/effective-pandas) — Best practices from a pandas core developer.