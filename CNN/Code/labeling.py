import pandas as pd
import sys
from pathlib import Path
import warnings
from tqdm import tqdm, trange
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path('/home/ubuntu/2022_VAIV_Cho/VAIV')
sys.path.append(ROOT)
sys.path.append(ROOT / 'Common' / 'Code')

from manager import VAIV  # noqa: E402
from utils.stock_format import correction  # noqa: E402


def make_ticker_labeling(vaiv: VAIV):
    market = vaiv.kwargs.get('market')
    candle = vaiv.kwargs.get('candle')
    forecast = vaiv.kwargs.get('forecast')
    df = vaiv.modedf.get('stock')
    date = df.Date
    

def make_all_labelings(vaiv: VAIV):
    market = vaiv.kwargs.get('market')
    vaiv.load_df(market)
    df = vaiv.modedf.get(market)
    vaiv.load_df('label')
    labeling = [vaiv.modedf.get('label')]

    pbar = tqdm(total=len(df.Ticker))
    for ticker in df.Ticker:
        vaiv.set_kwargs(ticker=ticker)
        vaiv.load_df('stock')
        label = make_ticker_labeling(vaiv)
        labeling.append(label)
        pbar.update()
    pbar.close()
    df = pd.concat(labeling)
    vaiv.set_df('label', df)
    vaiv.save_df('label')


if __name__ == '__main__':
    vaiv = VAIV(ROOT)
    kwargs = {
        'offset': 1,
        'market': 'Kospi',
        'labeling': '4%_01_ex',
        'candle': 20,
        'forecast': 5,
    }
    vaiv.set_kwargs(**kwargs)
    make_all_labelings(vaiv)


def process_labeling(market, dp, fore, df, ticker, labeling, offset) :
    path = f'/home/ubuntu/2022_VAIV_Dataset/Labeling/{offset}/{market}/{labeling}_{dp}_{fore}' +'.csv'
    date = df['Date']
    print(ticker)
    print('\nLabeling')
    end_dates = []
    tickers = []
    labels = []
    num = 0
    for i in trange(0, len(df), offset):
        c = df.iloc[i:i + int(dp), :]
        if len(c) == dp:
            end_date = date[i+int(dp)-1]
            # print(ticker, end_date)
            try:
                f = df.iloc[i+int(dp)+fore-1, :]
            except:
                continue
            starting = 0
            endvalue = 0
            label = ""
            if len(c) == int(dp):
                num += 1
                if labeling.split('%')[1] == '_012':
                    starting = c["Close"].iloc[-1]
                    endvalue = f["Close"]

                    if starting * (1 - (int(labeling[0]) / 100)) >= endvalue:
                        label = 1
                    elif starting * (1 + (int(labeling[0]) / 100)) <= endvalue:
                        label = 2
                    else:
                        label = 0

                elif labeling.split('%')[1] == '_01':
                    starting = c["Close"].iloc[-1]
                    endvalue = f["Close"]

                    if endvalue >= (1 + (int(labeling[0]) / 100)) * starting:
                        label = 1
                    else:
                        label = 0

                elif labeling.split('%')[1] == '_01_2' :
                    starting = c["Close"].iloc[-1]
                    endvalue = f["Close"]

                    if endvalue >= (1 + (int(labeling[0])/100)) * starting:
                        label = 1
                    elif endvalue < starting:
                        label = 0
                    else:
                        continue

                elif labeling.split('%')[1] == '_01_in':
                    try:
                        f_in = df.iloc[i+int(dp):i + int(dp) + fore, :]
                    except IndexError:
                        continue
                    starting = c["Close"].iloc[-1]
                    endvalues = f_in["Close"]
                    n = int(labeling.split('%')[0]) / 100
                    if any(endvalue >= (1 + n) * starting for endvalue in endvalues.values.tolist()):
                        label = 1
                    else:
                        label = 0

                elif labeling == 'High_Low_01':
                    starting = c["High"].iloc[-1]
                    endvalue = f["Low"]

                    if endvalue > starting:
                        label = 1
                    else:
                        label = 0

                elif labeling == '0123':
                    label_row = f
                    candle = label_row["Close"] - label_row["Open"]
                    line = label_row["High"] - label_row["Low"]

                    if candle <= 0.0:
                        label = 0
                        if abs(candle) / line >= 0.7:
                            label = 1
                    else:
                        label = 2
                        if abs(candle) / line >= 0.7:
                            label = 3
                else:
                    print('Please Select Correct Labeling!\n')
                    quit()

                end_dates.append(end_date)
                tickers.append(ticker)
                labels.append(label)

    labeling_row = pd.DataFrame({'Date': end_dates, 'Ticker': tickers, 'Label': labels})
    return path, labeling_row

def main() :
    data_kospi = '/home/ubuntu/2022_VAIV_Dataset/Stock_Data/Kospi_Data'
    data_kosdaq = '/home/ubuntu/2022_VAIV_Dataset/Stock_Data/Kosdaq_Data'

    kospi_files = os.listdir(data_kospi)
    kosdaq_files = os.listdir(data_kosdaq)

    # df = pd.read_csv('/home/ubuntu/2022_VAIV_Dataset/Labeling/5%_01.csv')
    # count = 1
    # kospi_path = ''
    # kospi_list = []

    # for file in kospi_files:
    #     ticker = file.split('.')[0]
    #     print('\nTicker {}/{}\t{}'.format(count, len(kospi_files), ticker))
    #     data_path = os.path.join(data_kospi, file)
    #     data = pd.read_csv(data_path)
    #     data = correction(data, ['5', '10', '30']) # correct {ticker}.csv
        # process_labeling(label_kospi, 20, 20, data, ticker, '10%_01_in')
        # kospi_path, new_df = process_labeling(label_kospi, 20, 5, data, ticker, '4%_01_2', 5)
        # process_labeling(label_kospi, 20, 1, data, ticker, '01')
        # process_labeling(label_kospi, 20, 1, data, ticker, '4%_01')
        # process_labeling('/home/ubuntu/2022_VAIV_Dataset/Labeling/Fore1', 20, 1, data, ticker, '4%_012')
        # process_labeling('/home/ubuntu/2022_VAIV_Dataset/Labeling/Fore1', 20, 1, data, ticker, 'High_Low_01')
        # process_labeling('/home/ubuntu/2022_VAIV_Dataset/Labeling/Fore1', 20, 1, data, ticker, '0123')
    #     kospi_list.append(new_df)
    #     count += 1

    # kospi_df = pd.concat(kospi_list, ignore_index = True)
    # kospi_df = kospi_df.astype({'Label':'int'})
    # kospi_df.to_csv(kospi_path)

    count = 1
    kosdaq_path = ''
    kosdaq_list = []

    for file in kosdaq_files:
        ticker = file.split('.')[0]
        print('\nTicker {}/{}\t{}'.format(count, len(kosdaq_files), ticker))
        data_path = os.path.join(data_kosdaq, file)
        data = pd.read_csv(data_path)
        data = correction(data, ['5', '10', '30'])  # correct {ticker}.csv
        kosdaq_path, new_df = process_labeling('Kosdaq', 20, 5, data, ticker, '4%_01_2', 5)
        kosdaq_list.append(new_df)
        count += 1

    kosdaq_df = pd.concat(kosdaq_list, ignore_index = True)
    kosdaq_df = kosdaq_df.astype({'Label':'int'})
    kosdaq_df.to_csv(kosdaq_path)

main()
