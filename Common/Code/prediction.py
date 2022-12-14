import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sys
import multiprocessing as mp

ROOT = Path('/home/ubuntu/2022_VAIV_Cho/VAIV')
sys.path.extend(str(ROOT))
sys.path.extend(str(ROOT / 'Common' / 'Code'))

from manager import VAIV  # noqa: E402
sys.path = list(set(sys.path))


# date 별 prediction 행 return
def make_prediction(vaiv: VAIV, stock):
    candle = vaiv.kwargs.get('candle')
    trade_date = vaiv.kwargs.get('trade_date')
    dates = stock.index.tolist()
    index = dates.index(trade_date)
    end = dates[index-1]
    start = dates[index-candle]
    new_predict = pd.DataFrame({
                        'Date': [trade_date],
                        'Start': [start],
                        'End': [end],
                    })
    new_predict.set_index('Date', inplace=True)
    return new_predict


# ticker를 지정하고 date 별 make_prediction
def make_ticker_predictions(vaiv: VAIV, ticker=None):
    vaiv.set_kwargs(ticker=ticker)
    candle = vaiv.kwargs.get('candle')
    vaiv.load_df('stock')
    vaiv.load_df('predict')
    stock = vaiv.modedf.get('stock')
    if len(stock) > candle:
        dates = stock.iloc[candle:].index.tolist()
        vaiv.load_df('predict')
        predict = vaiv.modedf.get('predict')
        pred_list = [predict]
        for trade_date in dates:
            vaiv.set_kwargs(trade_date=trade_date)
            new_predict = make_prediction(vaiv, stock)
            pred_list.append(new_predict)
        predict = pd.concat(pred_list)
        vaiv.set_df('predict', predict)
        vaiv.save_df('predict')


# market과 candle을 지정하고 ticker별 make_ticker_predictions
def make_all_predictions(vaiv: VAIV):
    market = vaiv.kwargs.get('market')
    vaiv.load_df(market)
    df = vaiv.modedf.get(market).reset_index()

    pool = mp.Pool(5)
    tickers = df.Ticker.tolist()
    vaiv_list = [vaiv for i in range(len(tickers))]

    pool.starmap(make_ticker_predictions, zip(vaiv_list, tickers))

    # pbar = tqdm(total=len(df.Ticker))
    # for ticker in df.Ticker:
    #     vaiv.set_kwargs(ticker=ticker)
    #     make_ticker_predictions(vaiv)
    #     pbar.update()
    # pbar.close()


def update_prediction(vaiv: VAIV):
    vaiv.load_df('stock')
    vaiv.load_df('predict')
    stock = vaiv.modedf.get('stock')
    predict = vaiv.modedf.get('predict')
    today = vaiv.kwargs.get('today')
    trade_date = vaiv.kwargs.get('trade_date')
    candle = vaiv.kwargs.get('candle')
    dates = stock.index.tolist()
    if (today in dates) & (len(stock) > candle):
        index = dates.index(today)
        start = dates[index-candle+1]
        new_predict = pd.DataFrame({
                        'Date': [trade_date],
                        'Start': [start],
                        'End': [today],
                    })
        new_predict.set_index('Date', inplace=True)
        predict = pd.concat([predict, new_predict])
        vaiv.set_df('predict', predict)
        vaiv.save_df('predict')


# market과 candle을 set 하고 prediction 파일 update
def update_all_predictions(vaiv: VAIV, today, trade_date):
    vaiv.set_kwargs(today=today)
    vaiv.set_kwargs(trade_date=trade_date)
    market = vaiv.kwargs.get('market')
    vaiv.load_df(market)
    df = vaiv.modedf.get(market).reset_index()

    pbar = tqdm(total=len(df.Ticker))
    for ticker in df.Ticker:
        vaiv.set_kwargs(ticker=ticker)
        update_prediction(vaiv)
        pbar.update()
    pbar.close()


if __name__ == '__main__':
    vaiv = VAIV(ROOT)
    kwargs = {
        'candle': 245,
        'market': 'Kosdaq',
    }
    vaiv.set_kwargs(**kwargs)
    vaiv.set_stock()
    vaiv.set_prediction()
    vaiv.make_dir(common=True, prediction=True)
    today = '2022-09-29'
    # vaiv.set_kwargs(ticker='03481K')
    make_ticker_predictions(vaiv, ticker='03481K')
    # make_all_predictions(vaiv)
    # update_all_predictions(vaiv, today)
