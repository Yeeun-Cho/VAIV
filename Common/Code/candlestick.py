# candlestick 차트를 만든다.
# market → ticker → pred
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
from PIL import Image
from pathlib import Path
import sys

ROOT = Path('/home/ubuntu/2022_VAIV_Cho/VAIV')
sys.path.append(ROOT)
sys.path.append(ROOT / 'Common' / 'Code')

from manager import VAIV  # noqa: E402
from utils.mpf import candlestick_ohlc, volume_overlay  # noqa: E402


def make_pixel(lines, patches, fig, stock, vaiv: VAIV):
    height = vaiv.kwargs.get('size')[1]
    xmin, xmax, ymin, ymax = [[] for i in range(4)]

    for i in range(len(stock)):
        bbox_x = patches[i].get_window_extent(fig.canvas.get_renderer())
        bbox_y = lines[i].get_window_extent(fig.canvas.get_renderer())
        xmin.append(bbox_x.x0)
        ymin.append(height-bbox_y.y1)
        xmax.append(bbox_x.x1)
        ymax.append(height-bbox_y.y0)

    dates = stock.index.tolist()

    df = pd.DataFrame({
        'Date': dates,
        'Xmin': xmin, 'Ymin': ymin,
        'Xmax': xmax, 'Ymax': ymax
    })
    df.set_index('Date', inplace=True)
    vaiv.set_df('pixel', df)
    vaiv.save_df('pixel')


def subplots(volume, MACD):
    tf = [volume, MACD]
    ax = []
    count = tf.count(True)
    if count == 0:
        ax = [111]
    elif count == 1:
        ax = [211, 212]
    else:
        ax = [211, 223, 224]
    return count, ax


def make_candlestick(vaiv: VAIV, stock, pred):
    ticker = vaiv.kwargs.get('ticker')
    date = vaiv.kwargs.get('trade_date')
    feature = vaiv.kwargs.get('feature')
    Volume = feature.get('Volume')
    MACD = feature.get('MACD')
    MA = feature.get('MA')
    style = vaiv.kwargs.get('style')
    size = vaiv.kwargs.get('size')
    candle = vaiv.kwargs.get('candle')
    linespace = vaiv.kwargs.get('linespace')
    candlewidth = vaiv.kwargs.get('candlewidth')
    
    vaiv.set_fname('png', ticker=ticker, date=date)
    vaiv.set_path(vaiv.common.image.get('images'))

    if vaiv.path.exists():
        return

    try:
        c = stock.loc[pred.Start:pred.End]
    except KeyError:
        print(ticker, pred)
        return
    plt.style.use(style)
    color = ['#0061cb', '#efbb00', '#ff4aad', '#882dff', '#2bbcff']
    num, ax = subplots(Volume, MACD)
    fig = plt.figure(figsize=(size[0]/100, size[1]/100))
    ax1 = fig.add_subplot(ax[0])

    t = np.arange(0, candle*linespace, linespace)
    quote = c.copy()
    quote.insert(0, 't', t)
    quote.reset_index(drop=True, inplace=True)
    quote = quote.astype(int)
    
    lines, patches = candlestick_ohlc(
        ax1, quote.values, width=candlewidth,
        colorup='#77d879', colordown='#db3f3f', alpha=None
    )

    if Volume:
        ax2 = fig.add_subplot(ax[1])
        bc = volume_overlay(
            ax2, c['Open'], c['Close'], c['Volume'], width=1,
            colorup='#77d879', colordown='#db3f3f', alpha=None,
        )
        ax2.add_collection(bc)
        ax2.grid(False)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        ax2.axis('off')

    if MACD:
        ax3 = fig.add_subplot(ax[num])
        ax3.plot(c['MACD'], linewidth=1, color='red', alpha=None)
        ax3.plot(c['MACD_Signal'], linewidth=1, color='white', alpha=None)
        ax3.grid(False)
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.xaxis.set_visible(False)
        ax3.yaxis.set_visible(False)
        ax3.axis('off')

    if MA != [-1]:
        for m, i in zip(MA, range(len(MA))):
            ax1.plot(
                c[f'{MA}MA'], linewidth=size[1]/224,
                color=color[i], alpha=None
            )

    ax1.grid(False)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.axis('off')

    plt.tight_layout(pad=0)
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0)

    fig.savefig(vaiv.path)
    
    pil_image = Image.open(vaiv.path)
    rgb_image = pil_image.convert('RGB')
    rgb_image.save(vaiv.path)

    vaiv.load_df('pixel')
    make_pixel(lines, patches, fig, c, vaiv)
    plt.close(fig)


def make_ticker_candlesticks(vaiv: VAIV, start_date, end_date):
    vaiv.load_df('stock')
    vaiv.load_df('predict')
    stock = vaiv.modedf.get('stock')
    predict = vaiv.modedf.get('predict')
    condition = (predict.index >= start_date) & (predict.index <= end_date)
    predict = predict.loc[condition]

    if not predict.empty:
        for date in predict.index.tolist():
            pred = predict.loc[date]
            vaiv.set_kwargs(trade_date=date)
            make_candlestick(vaiv, stock, pred)


def make_all_candlesticks(
            vaiv: VAIV,
            start_date='2006',
            end_date='a',
        ):
    market = vaiv.kwargs.get('market')
    vaiv.load_df(market)
    df = vaiv.modedf.get(market).reset_index()

    pbar = tqdm(total=len(df.Ticker))
    for ticker in df.Ticker:
        vaiv.set_kwargs(ticker=ticker)
        make_ticker_candlesticks(vaiv, start_date, end_date)
        pbar.update()
    pbar.close()


def update_candlestick(vaiv: VAIV):
    date = vaiv.kwargs.get('trade_date')
    vaiv.load_df('stock')
    vaiv.load_df('predict')
    stock = vaiv.modedf.get('stock')
    predict = vaiv.modedf.get('predict')

    if date in predict.index.tolist():
        pred = predict.loc[date]
        make_candlestick(vaiv, stock, pred)


def update_all_candlesticks(vaiv: VAIV, today):
    vaiv.set_kwargs(trade_date=today)
    market = vaiv.kwargs.get('market')
    vaiv.load_df(market)
    df = vaiv.modedf.get(market).reset_index()

    pbar = tqdm(total=len(df.Ticker))
    for ticker in df.Ticker:
        vaiv.set_kwargs(ticker=ticker)
        update_candlestick(vaiv)
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
    start_date = '2006-01-01'
    end_date = '2021-12-31'
    vaiv.set_kwargs(**kwargs)
    vaiv.set_stock()
    vaiv.set_prediction()
    vaiv.set_image()
    vaiv.make_dir(common=True, image=True)
    make_all_candlesticks(vaiv, start_date=start_date, end_date=end_date)
