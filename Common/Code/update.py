from pathlib import Path
import sys
import exchange_calendars as ecals
from datetime import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

ROOT = Path('/home/ubuntu/2022_VAIV_Cho/VAIV')
sys.path.append(ROOT)
sys.path.append(ROOT / 'Common' / 'Code')

from manager import VAIV  # noqa: E402
from stock import update_all_stocks  # noqa: E402
from prediction import update_all_predictions  # noqa: E402
from candlestick import update_all_candlesticks  # noqa: E402


def DateCheck(today):
    XKRX = ecals.get_calendar('XKRX')
    return XKRX.is_session(today)


def update(vaiv: VAIV):
    today = datetime.now()
    today = today.strftime('%Y-%m-%d')
    if DateCheck(today):
        update_all_stocks(vaiv, today)
        update_all_predictions(vaiv, today)
        update_all_candlesticks(vaiv, today)


if __name__ == '__main__':
    vaiv = VAIV(ROOT)
    kwargs = {
        'market': 'Kospi',
        'feature': {'Volume': False, 'MA': [-1], 'MACD': False},
        'offset': 1,
        'size': [1800, 650],
        'candle': 245,
        'linespace': 1,
        'candlewidth': 0.8,
        'style': 'dark_background'  # default는 'classic'
    }
    vaiv.set_kwargs(**kwargs)
    vaiv.set_stock()
    vaiv.set_prediction()
    vaiv.set_image()
    update(vaiv)
