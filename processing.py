import collections
import numpy as np
import util
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import re

def normalize(data):
    stop_words = set(stopwords.words('english'))
    stop_words.add(",")
    stop_words.add(".")
    tokenizer = TweetTokenizer()

    num_tweets, _ = data.shape
    for i in range(num_tweets):
        line = data[i, 1]
        line = re.sub(r"\$[A-Za-z]+", "", line)
        line = re.sub(r"http[\S]*", "", line)
        line = re.sub(r"\s", " ", line)
        line = re.sub(r"[\S]*(Ã|â|Â|¦|œ)[\S]*", "", line)
        line = re.sub(r"[\S]*t\.co[\S]*", "", line)
        word_tokens = tokenizer.tokenize(line)
        filtered_sentence = ""
        for w in word_tokens:
            if w not in stop_words and "Ã" not in w and "â" not in w:
                filtered_sentence += " " + w

        filtered_sentence = filtered_sentence.lower()
        data[i, 1] = filtered_sentence
        print(i)
    normalized = pd.DataFrame(data=data)
    normalized.to_csv("final_data/normalized_data.csv")

def load_dataset(tweets_paths, prices_path, add_intercept=False):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 'l').
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """
    prices = pd.read_csv(prices_path, encoding = "ISO-8859-1", parse_dates = ['date'])
    final = pd.DataFrame()

    for tweets_path in tweets_paths:
        symbol = tweets_path[:tweets_path.find('/')].upper()
        tweets_path = "initial_data/" + tweets_path

        # Load Tweets
        tweets = pd.read_excel(tweets_path, sheet_name='Stream', encoding = "ISO-8859-1", parse_dates = ['Date'])
        tweets = tweets.drop(columns=['Hour', 'Tweet Id', 'User Name', 'Nickname', 'Bio'])
        aggregation_functions = {'Tweet content': 'sum', 'Date': 'first'}
        tweets = tweets.groupby(tweets['Date']).aggregate(aggregation_functions)
        tweets['date'] = tweets['Date']
        tweets = tweets.drop(columns=['Date'])

        # Load Prices
        specific_prices = prices.loc[prices['symbol'] == symbol]
        specific_prices = specific_prices.drop(columns=['open', 'low', 'high', 'volume'])
        next_day_prices = specific_prices.copy()
        next_day_prices = next_day_prices.drop(columns=['symbol'])
        next_day_prices['date'] -= pd.DateOffset(days=1)

        new_prices = specific_prices.merge(next_day_prices, how = 'inner', on = ['date'])
        new_prices['increase'] = np.where((new_prices['close_y'] - new_prices['close_x']) > 0, 1, 0) # For a certain date, increase represents whether or not the stock price increased between the current date's close and the next day's close
        specific_prices = new_prices.drop(columns=['close_y', 'close_x'])

        # Merge tweets and prices
        merged = tweets.merge(specific_prices, how = 'inner', on = ['date'])

        merged.to_csv("final_data/" + symbol + ".csv")

        final = final.append(merged)


    final.to_csv("final_data/compiled_data.csv")

    return final

def main():
    # files = ['aal/export_dashboard_aal.xlsx',
    # 'aapl/export_dashboard_aapl.xlsx',
    # 'adbe/export_dashboard_adbe.xlsx',
    # 'adp/export_dashboard_adp.xlsx',
    # 'adsk/export_dashboard_adsk.xlsx',
    # 'akam/export_dashboard_akam.xlsx',
    # 'alxn/export_dashboard_alxn.xlsx',
    # 'amat/export_dashboard_amat.xlsx',
    # 'amgn/export_dashboard_amgn.xlsx',
    # 'amzn/export_dashboard_amzn.xlsx',
    # 'atvi/export_dashboard_atvi.xlsx',
    # 'avgo/export_dashboard_avgo.xlsx',
    # 'bbby/export_dashboard_bbby.xlsx',
    # 'bidu/export_dashboard_bidu.xlsx',
    # 'bmrn/export_dashboard_bmrn.xlsx',
    # 'ca/export_dashboard_ca.xlsx',
    # 'celg/export_dashboard_celg.xlsx',
    # 'cern/export_dashboard_cern.xlsx',
    # 'chkp/export_dashboard_chkp.xlsx',
    # 'chtr/export_dashboard_chtr.xlsx',
    # 'cmcsa/export_dashboard_cmcsa.xlsx',
    # 'cost/export_dashboard_cost.xlsx',
    # 'csco/export_dashboard_csco.xlsx',
    # 'csx/export_dashboard_csx.xlsx',
    # 'ctrp/export_dashboard_ctrp.xlsx',
    # 'ctsh/export_dashboard_ctsh.xlsx',
    # 'disca/export_dashboard_disca.xlsx',
    # 'disck/export_dashboard_disck.xlsx',
    # 'dish/export_dashboard_dish.xlsx',
    # 'dltr/export_dashboard_dltr.xlsx',
    # 'ea/export_dashboard_ea.xlsx',
    # 'ebay/export_dashboard_ebay.xlsx',
    # 'endp/export_dashboard_endp.xlsx',
    # 'esrx/export_dashboard_esrx.xlsx',
    # 'expe/export_dashboard_expe.xlsx',
    # 'fast/export_dashboard_fast.xlsx',
    # 'fb/export_dashboard_fb.xlsx',
    # 'fisv/export_dashboard_fisv.xlsx',
    # 'fox/export_dashboard_fox.xlsx',
    # 'foxa/export_dashboard_foxa.xlsx',
    # 'gild/export_dashboard_gild.xlsx',
    # 'goog/export_dashboard_goog.xlsx',
    # 'googl/export_dashboard_googl.xlsx',
    # 'hsic/export_dashboard_hsic.xlsx',
    # 'ilmn/export_dashboard_ilmn.xlsx',
    # 'inct/export_dashboard_inct.xlsx',
    # 'incy/export_dashboard_incy.xlsx',
    # 'intu/export_dashboard_intu.xlsx',
    # 'isrg/export_dashboard_isrg.xlsx',
    # 'jd/export_dashboard_jd.xlsx',
    # 'khc/export_dashboard_khc.xlsx',
    # 'lbtya/export_dashboard_lbtya.xlsx',
    # 'lbtyk/export_dashboard_lbtyk.xlsx',
    # 'lltc/export_dashboard_lltc.xlsx',
    # 'lmca/export_dashboard_lmca.xlsx',
    # 'lmck/export_dashboard_lmck.xlsx',
    # 'lrcx/export_dashboard_lrcx.xlsx',
    # 'lvnta/export_dashboard_lvnta.xlsx',
    # 'mar/export_dashboard_mar.xlsx',
    # 'mat/export_dashboard_mat.xlsx',
    # 'mdlz/export_dashboard_mdlz.xlsx',
    # 'mnst/export_dashboard_mnst.xlsx',
    # 'msft/export_dashboard_msft.xlsx',
    # 'mu/export_dashboard_mu.xlsx',
    # 'mxim/export_dashboard_mxim.xlsx',
    # 'myl/export_dashboard_myl.xlsx',
    # 'nclh/export_dashboard_nclh.xlsx',
    # 'nflx/export_dashboard_nflx.xlsx',
    # 'ntap/export_dashboard_ntap.xlsx',
    # 'ntes/export_dashboard_ntes.xlsx',
    # 'nvda/export_dashboard_nvda.xlsx',
    # 'nxpi/export_dashboard_nxpi.xlsx',
    # 'orly/export_dashboard_orly.xlsx',
    # 'payx/export_dashboard_payx.xlsx',
    # 'pcar/export_dashboard_pcar.xlsx',
    # 'pcln/export_dashboard_pcln.xlsx',
    # 'pypl/export_dashboard_pypl.xlsx',
    # 'qcom/export_dashboard_qcom.xlsx',
    # 'qvca/export_dashboard_qvca.xlsx',
    # 'regn/export_dashboard_regn.xlsx',
    # 'rost/export_dashboard_rost.xlsx',
    # 'sbac/export_dashboard_sbac.xlsx',
    # 'sbux/export_dashboard_sbux.xlsx',
    # 'sndk/export_dashboard_sndk.xlsx',
    # 'srcl/export_dashboard_srcl.xlsx',
    # 'stx/export_dashboard_stx.xlsx',
    # 'swks/export_dashboard_swks.xlsx',
    # 'symc/export_dashboard_symc.xlsx',
    # 'tmus/export_dashboard_tmus.xlsx',
    # 'trip/export_dashboard_trip.xlsx',
    # 'tsco/export_dashboard_tsco.xlsx',
    # 'tsla/export_dashboard_tsla.xlsx',
    # 'txn/export_dashboard_txn.xlsx',
    # 'ulta/export_dashboard_ulta.xlsx',
    # 'viab/export_dashboard_viab.xlsx',
    # 'vod/export_dashboard_vod.xlsx',
    # 'vrsk/export_dashboard_vrsk.xlsx',
    # 'vrtx/export_dashboard_vrtx.xlsx',
    # 'wba/export_dashboard_wba.xlsx',
    # 'wdc/export_dashboard_wdc.xlsx',
    # 'wfm/export_dashboard_wfm.xlsx',
    # 'xlnx/export_dashboard_xlnx.xlsx',
    # 'yhoo/export_dashboard_yhoo.xlsx']

    # out = load_dataset(files, 'initial_data/prices.csv')
    # data = out.values # convert to numpy.ndarray

    # Read compiled data using:
    out = pd.read_csv("final_data/compiled_data.csv", encoding = "ISO-8859-1")
    data = out.values
    normalize(data)

if __name__ == "__main__":
    main()
