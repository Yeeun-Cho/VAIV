import sys
from pathlib import Path

ROOT = Path('/home/ubuntu/2022_VAIV_Cho/VAIV')
sys.path.append(ROOT)
sys.path.append(ROOT / 'Common' / 'Code')

from manager import VAIV  # noqa: E402
from stock import make_all_stocks  # noqa: E402
from prediction import make_all_predictions  # noqa: E402
from candlestick import make_all_candlesticks  # noqa: E402


def make_all(
        vaiv: VAIV, stock=True, prediction=True, candlestick=True,
        start_date='2006', end_date='a'
        ):
    if stock:
        make_all_stocks(vaiv)
        print('Create Stocks Finished')
    if prediction:
        make_all_predictions(vaiv)
        print('Create Predictions Finished')
    if candlestick:
        make_all_candlesticks(vaiv, start_date, end_date)
        print('Create Candlesticks Finished')


def make_all_vaivs(vaivs, stock=True, prediction=True, candlestick=True):
    for vaiv in vaivs:
        make_all(vaiv, stock, prediction, candlestick)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--year', '-y', default=-1, type=int, help='update yolo files'
    )
    parser.add_argument(
        '--yolo', action='store_true', help='update yolo files'
    )
    opt = parser.parse_args()
    vaiv = VAIV(ROOT)
    kwargs = {
        'market': 'Kosdaq',
        'feature': {'Volume': False, 'MA': [-1], 'MACD': False},
        'offset': 1,
        'size': [1800, 650],
        'candle': 245,
        'linespace': 1,
        'candlewidth': 0.8,
        'style': 'default'  # 밝은 배경은 'default', 어두운 배경은 'dark_background'
    }
    year = opt.year
    vaiv.set_kwargs(**kwargs)
    vaiv.set_stock()
    vaiv.set_prediction()
    vaiv.set_image()
    vaiv.make_dir(common=True, stock=True, prediction=True, image=True)

    print(opt)
    if year > 0:
        make_all(vaiv, stock=True, prediction=True, candlestick=True, start_date=str(year), end_date=str(year+1))
    else:
        make_all(vaiv, stock=True, prediction=True, candlestick=True)
