import pandas as pd
from pathlib import Path
import sys

# ROOT = Path('/home/ubuntu/2022_VAIV_Cho/VAIV')
# sys.path.append(ROOT)
# sys.path.append(ROOT / 'Common' / 'Code')
#
# from manager import VAIV  # noqa: E402
#
# vaiv = VAIV(ROOT)
# vaiv.load_df('Kospi')
# df = vaiv.modedf.get('Kospi')
# df.set_index('Ticker', inplace=True)
# new_df = pd.DataFrame({'Ticker': ['000000'], 'Name': ['Test']})
# new_df.set_index('Ticker', inplace=True)
# test = pd.concat([df, new_df])
# print(test.loc[(test.index > '000000') & True])
# test = test.iloc[:5]
# t = [1, 2, 3, 4, 5]
# quote = test.copy()
# quote.insert(0, 'Ticker', t)
# print(quote.values)

p = Path('/home/ubuntu/2022_VAIV_Dataset/Yolo1/Image/Kospi/1800x650_245_1_0.8/Image')
files = sorted(list(p.iterdir()))
# print(files.index(Path('/home/ubuntu/2022_VAIV_Dataset/Yolo1/Image/Kospi/1800x650_245_1_0.8/Image/A005880_2021-12-30.png')))
print(files[190353])
