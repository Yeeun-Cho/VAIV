from pathlib import Path
import sys
import warnings
from tqdm import tqdm
warnings.simplefilter(action='ignore', category=FutureWarning)

ROOT = Path('/home/ubuntu/2022_VAIV_Cho/VAIV')
sys.path.append(ROOT)
sys.path.append(ROOT / 'Common' / 'Code')

from manager import VAIV  # noqa: E402


def delete_stock(vaiv: VAIV):
    vaiv.load_df('stock')


def delete_prediction(vaiv: VAIV):
    vaiv.load_df('predict')
    predict = vaiv.modedf.get('predict')
    predict = predict.reset_index().drop_duplicates('Date').set_index('Date')
    vaiv.set_df('predict', predict)
    vaiv.save_df('predict')


def delete_predictions(vaiv: VAIV):
    market = vaiv.kwargs.get('market')
    vaiv.load_df(market)
    df = vaiv.modedf.get(market).reset_index()

    pbar = tqdm(total=len(df))
    for ticker in df.Ticker:
        vaiv.set_kwargs(ticker=ticker)
        delete_prediction(vaiv)
        pbar.update()
    pbar.close()


if __name__ == '__main__':
    vaiv = VAIV(ROOT)
    kwargs = {
        'market': 'Kospi',
        'feature': {'Volume': False, 'MA': [-1], 'MACD': False},
        'offset': 1,
        'size': [224, 224],
        'candle': 20,
        'linespace': 1,
        'candlewidth': 0.8,
        'style': 'dark_background'  # default는 'classic'
    }
    vaiv.set_kwargs(**kwargs)
    vaiv.set_stock()
    vaiv.set_prediction()
    delete_predictions(vaiv)
