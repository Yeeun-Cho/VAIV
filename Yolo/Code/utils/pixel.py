from pathlib import Path
import sys

sys.path.append('/home/ubuntu/2022_VAIV_Cho/VAIV/Yolo/Code/yolov7')

ROOT = Path('/home/ubuntu/2022_VAIV_Cho/VAIV')
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / 'Common' / 'Code'))

from manager import VAIV  # noqa: E402


class StockImage:
    def __init__(self, path, vaiv: VAIV):
        p = Path(path)
        size = vaiv.kwargs.get('size')
        file_name = p.stem
        ticker, date = file_name.split('_')
        vaiv.set_kwargs(ticker=ticker, trade_date=date)
        vaiv.load_df('pixel')
        vaiv.load_df('stock')
        pixel = vaiv.modedf.get('pixel')
        self.stock = vaiv.modedf.get('stock')
        self.pixel = pixel
        self.ticker = ticker
        self.trade_date = date
            
    def get_box_date(self, xmin, xmax):
        pixels = self.pixel.to_dict('index')
        dates = []
        for date, pixel in pixels.items():
            pix_min = pixel['Xmin']
            pix_max = pixel['Xmax']
            i = min(pix_max, xmax) - max(pix_min, xmin)
            w = pix_max - pix_min
            if i / w > 0.2:
                dates.append(date)
        return dates

    def get_trade_date(self, dates):
        try:
            end = dates[-1]
        except IndexError:
            print(self.ticker, self.trade_date, dates)
            return None
        after = self.pixel.index[self.pixel.index > end].tolist()
        if len(after) == 0:
            return self.trade_date
        else:
            return after[0]
    
    def get_trade_close(self, date):
        try:
            close = self.stock.loc[date, 'Close']
        except KeyError:
            close = 0
        return close

    def get_pixel(self, i):
        pix_min = self.pixel.Xmin.iloc[i]
        pix_max = self.pixel.Xmax.iloc[i]
        return pix_min, pix_max

    def last_signal(self, xmin, xmax, date_thres):
        index = [-i for i in range(1, date_thres+1)]
        for i in index:
            pix_min, pix_max = self.get_pixel(i)
            i = min(pix_max, xmax) - max(pix_min, xmin)
            w = pix_max - pix_min
            if i / w > 0.2:
                return True
        return False


        # i = min(pix_max, xmax) - max(pix_min, xmin)
        # w = pix_max - pix_min

        # if i / w > 0.4:
        #     return True
        # else:
        #     return False
