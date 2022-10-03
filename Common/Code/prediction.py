import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sys

ROOT = Path('/home/ubuntu/2022_VAIV_Cho/VAIV')
sys.path.append(ROOT)
sys.path.append(ROOT / 'Common' / 'Code')

from manager import VAIV  # noqa: E402


# date 별 prediction 행 return
def make_prediction(vaiv: VAIV, stock):
    candle = vaiv.kwargs.get('candle')
    date = vaiv.kwargs.get('date')
    dates = stock.index.tolist()
    index = dates.index(date)
    end = dates[index-1]
    start = dates[index-candle]
    close = stock.loc[date, 'Close']
    new_predict = pd.DataFrame({
                        'Date': [date],
                        'Start': [start],
                        'End': [end],
                        'Close': [close],
                    })
    new_predict.set_index('Date', inplace=True)
    return new_predict


# ticker를 지정하고 date 별 make_prediction
def make_ticker_predictions(vaiv: VAIV):
    candle = vaiv.kwargs.get('candle')
    vaiv.load_df('stock')
    vaiv.load_df('predict')
    stock = vaiv.modedf.get('stock')
    if len(stock) > candle:
        dates = stock.iloc[candle:].index.tolist()
        vaiv.load_df('predict')
        predict = vaiv.modedf.get('predict')
        pred_list = [predict]
        for date in dates:
            vaiv.set_kwargs(date=date)
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

    pbar = tqdm(total=len(df.Ticker))
    for ticker in df.Ticker:
        vaiv.set_kwargs(ticker=ticker)
        make_ticker_predictions(vaiv)
        pbar.update()
    pbar.close()


def update_prediction(vaiv: VAIV):
    vaiv.load_df('stock')
    vaiv.load_df('predict')
    stock = vaiv.modedf.get('stock')
    predict = vaiv.modedf.get('predict')
    date = vaiv.kwargs.get('date')
    candle = vaiv.kwargs.get('candle')
    dates = stock.index.tolist()
    if (date in dates) & (len(stock) > candle):
        new_predict = make_prediction(vaiv, stock)
        predict = pd.concat([predict, new_predict])
        vaiv.set_df('predict', predict)
        vaiv.save_df('predict')
        

# market과 candle을 set 하고 prediction 파일 update
def update_all_predictions(vaiv: VAIV, today):
    vaiv.set_kwargs(date=today)
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
        'market': 'Kospi',
    }
    vaiv.set_kwargs(**kwargs)
    vaiv.set_stock()
    vaiv.set_prediction()
    vaiv.make_dir(common=True, prediction=True)
    today = '2022-09-29'
    update_all_predictions(vaiv, today)
