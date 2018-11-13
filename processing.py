import collections
import numpy as np
import util
import pandas as pd


def load_dataset(tweets_path, prices_path, add_intercept=False):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 'l').
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    symbol = tweets_path[:tweets_path.find('_')].upper()

    # Load Tweets
    tweets = pd.read_csv(tweets_path, encoding = "ISO-8859-1", parse_dates = ['Date'])
    aggregation_functions = {'Tweet content': 'sum', 'Hour': 'first', 'Date': 'first'}
    tweets = tweets.groupby(tweets['Date']).aggregate(aggregation_functions)
    tweets['date'] = tweets['Date']
    tweets = tweets.drop(columns=['Date', 'Hour'])

    # Load Prices
    prices = pd.read_csv(prices_path, encoding = "ISO-8859-1", parse_dates = ['date'])
    prices = prices.loc[prices['symbol'] == symbol]
    prices = prices.drop(columns=['open', 'low', 'high', 'volume'])
    next_day_prices = prices.copy()
    next_day_prices = next_day_prices.drop(columns=['symbol'])
    next_day_prices['date'] -= pd.DateOffset(days=1)

    new_prices = prices.merge(next_day_prices, how = 'inner', on = ['date'])
    new_prices['increase'] = np.where((new_prices['close_y'] - new_prices['close_x']) > 0, 1, 0) # For a certain date, increase represents whether or not the stock price increased between the current date's close and the next day's close
    prices = new_prices.drop(columns=['close_y', 'close_x'])

    # Merge tweets and prices
    final = tweets.merge(prices, how = 'inner', on = ['date'])

    final.to_csv("final_data/" + symbol + ".csv")

    return final

def main():
    out = load_dataset('aapl_2016_06_15_14_30_09/export_dashboard_aapl_2016_06_15_14_30_09.csv', 'prices.csv')
    data = out.values # convert to numpy.ndarray

if __name__ == "__main__":
    main()
