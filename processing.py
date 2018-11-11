import collections

import numpy as np

import util
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
    inputs = np.genfromtxt(csv_path, dtype=np.dtype(str), delimiter=',', encoding = "ISO-8859-1") #, missing_values=True, skip_header=1, usecols=x_cols, encoding = "ISO-8859-1", filling_values="N/A"


    # Load features and labels
    # x_cols = [i for i in range(len(headers))]
    # x_cols = [i for i in range(len(headers)) if headers[i] != label_col]
    # print(x_cols)
    # l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    # print(l_cols)
    # inputs = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=x_cols, encoding = "ISO-8859-1", filling_values="N/A")
    # labels = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=l_cols, encoding = "ISO-8859-1", filling_values="N/A")
    # inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, encoding = "ISO-8859-1")
    # labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols, encoding = "ISO-8859-1")

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs
    # return inputs, labels

def main():
    out = load_dataset('aapl_2016_06_15_14_30_09/export_dashboard_aapl_2016_06_15_14_30_09.csv')
    print(out)

if __name__ == "__main__":
    main()
