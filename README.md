# Data Science Workflow
#### By, Alena McLucas

Are you tired of writing the same code over and over? Me too. This tool will serve as a place to add common tasks, to group together statistics and visualizations that come from the same tasks, and to iterate faster with tested code.

Nothing is usable yet. The following will be implemented in this order:

1. Dataset
2. Statistic
3. Algorithm

## Test Datasets

Below are links to the original source of the test datasets:
- [Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci) -> `data/heart-disease.csv`
- [Titanic](https://www.kaggle.com/c/titanic/overview) -> `data/titanic.csv`
  - Only the original training set is included
- [Wine Reviews](https://www.kaggle.com/zynicide/wine-reviews) -> `data/wine-reviews.csv`
  - Only the second version, titled "winemag-data-130k-v2.csv" is include
  - When downloaded, the csv doesn't have a header name for the "#" field, so I gave it the name "id"
- [Household Electric Power Consumption](https://www.kaggle.com/uciml/electric-power-consumption-data-set) -> `data/power-consumption.csv`
  - I've included a short script to parse it to a csv -> `data/txt_to_csv.py`
- [Daily Climate time series data](https://www.kaggle.com/sumanthvrao/daily-climate-time-series-data) -> `data/daily-climate.csv`
