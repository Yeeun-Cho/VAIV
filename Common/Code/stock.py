import sys
# fdr은 ETF, ETN 포함이다. 따라서 다음과 같이 제거
# etf_list = fdr.StockListing("ETF/KR").Symbol
# df = df[~df['Ticker'].isin(etf_list)]
# df = df[~df['Name'].str.contains('ETN')]
import FinanceDataReader as fdr
from pykrx import stock
from pathlib import Path
from tqdm import tqdm
import pandas as pd

ROOT = Path('/home/ubuntu/2022_VAIV_Cho/VAIV')
sys.path.append(ROOT)
sys.path.append(ROOT / 'Common' / 'Code')

from manager import VAIV  # noqa: E402
from utils.stock_format import correction  # noqa: E402


# market을 set 하고, 각 market마다 추가된 종목 있으면 업데이트
def update_market(vaiv: VAIV):
    market = vaiv.kwargs.get('market')
    vaiv.load_df(market)
    df = vaiv.modedf.get(market).reset_index()
    krx = stock.get_market_ticker_list(market=market.upper())
    new_list = [df]
    for ticker in krx:
        if ticker not in df.Ticker.tolist():
            name = stock.get_market_ticker_name(ticker)
            new = pd.DataFrame({'Ticker': [ticker], 'Name': [name]})
            new_list.append(new)
    df = pd.concat(new_list, ignore_index=True)
    df.sort_values('Ticker', inplace=True)
    df.set_index('Ticker', inplace=True)
    vaiv.set_df(market, df)
    vaiv.save_df(market)
    return


# market과 ticker를 set 하고, 해당 ticker의 주가 데이터를 업데이트
def update_stock(vaiv: VAIV, today):
    ticker = vaiv.kwargs.get('ticker')
    vaiv.load_df('stock')
    df = fdr.DataReader(ticker, today, today).reset_index(level=0)
    df = correction(df, [-1], '%Y-%m-%d %H:%M:%S')
    if df.empty:
        return
    df = pd.concat([vaiv.modedf.get('stock'), df])
    vaiv.set_df('stock', df)
    vaiv.save_df('stock')


# market을 set 하고, 시장의 모든 ticker마다 update_stock
def update_all_stocks(vaiv: VAIV, today):
    market = vaiv.kwargs.get('market')
    vaiv.load_df(market)
    df = vaiv.modedf.get(market).reset_index()

    pbar = tqdm(total=len(df.Ticker))
    for ticker in df.Ticker:
        vaiv.set_kwargs(ticker=ticker)
        update_stock(vaiv, today)
        pbar.update()
    pbar.close()


def make_stock(vaiv: VAIV):
    ticker = vaiv.kwargs.get('ticker')
    vaiv.load_df('stock')
    df = fdr.DataReader(ticker).reset_index(level=0)
    df = correction(df, [-1], '%Y-%m-%d %H:%M:%S')
    vaiv.set_df('stock', df)
    vaiv.save_df('stock')


# market을 set 하고, 해당 market의 모든 주가 데이터 받기
def make_all_stocks(vaiv: VAIV):
    market = vaiv.kwargs.get('market')
    vaiv.load_df(market)
    df = vaiv.modedf.get(market).reset_index()

    pbar = tqdm(total=len(df))
    for ticker in df.Ticker:
        vaiv.set_kwargs(ticker=ticker)
        make_stock(vaiv)
        pbar.update()
    pbar.close()


if __name__ == '__main__':
    vaiv = VAIV(ROOT)
    market = 'Kospi'
    ticker = '172580'
    today = '2022-09-07'
    vaiv.set_kwargs(market=market)  # market 지정
    vaiv.set_stock()  # stock 폴더 지정
    vaiv.make_dir(common=True, stock=True)
    # update_all_stocks(vaiv, today)
    vaiv.set_kwargs(ticker=ticker)
    # update_market(vaiv)
    # make_all_stocks(vaiv)
    # make_stock(vaiv)
    update_all_stocks(vaiv, today)
