import collections

import numpy as np

import util

import pandas as pd
# import svm

def load_dataset(csv_path, label_col='Bio', add_intercept=False):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 'l').
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    def add_intercept_fn(x):
        global add_intercept
        return add_intercept(x)

    # Validate label_col argument
    allowed_label_cols = ('Bio')
    if label_col not in allowed_label_cols:
        raise ValueError('Invalid label_col: {} (expected {})'
                         .format(label_col, allowed_label_cols))

    # Load headers
    with open(csv_path, 'r', encoding = "ISO-8859-1") as csv_fh:
        headers_line = csv_fh.readline()
        headers = headers_line.strip().split(',')
    x_cols = [i for i in range(len(headers))]
    inputs = pd.read_csv(csv_path, encoding = "ISO-8859-1")

    aggregation_functions = {'Tweet content': 'sum', 'Hour': 'first'}
    inputs = inputs.groupby(inputs['Date']).aggregate(aggregation_functions)

    return inputs

def main():
    out = load_dataset('aapl_2016_06_15_14_30_09/export_dashboard_aapl_2016_06_15_14_30_09.csv')
    data = out.values
    # print(data)
    print(data[0])

if __name__ == "__main__":
    main()
