'''
VAIV 폴더의 전반적인 Path를 관리하는 코드다
'''

import pandas as pd
from pathlib import Path


class FileManager:
    def set_fname(self, extension, **kwargs):
        values = '_'.join(kwargs.values())
        self.fname = f'{values}.{extension}'

    def set_path(self, dir):
        self.path = dir / self.fname

    def load_df(self, empty: dict, index=None):
        try:
            self.df = pd.read_csv(self.path)
        except FileNotFoundError:
            self.df = pd.DataFrame(empty)
        if index:
            self.df.set_index(index, inplace=True)

    def set_df(self, df):
        self.df = df

    def save_df(self):
        self.df.to_csv(self.path)


class Classification:  # Classification 폴더 관리
    def __init__(self, vaiv):
        '''
        Classification
          ㄴLabeling
          ㄴDataset
          ㄴCode
        '''
        self.root = vaiv / 'Classification'
        self.top = {
            'labeling': self.root / 'Labeling',
            'dataset': self.root / 'Dataset',
            'code': self.root / 'Code'
        }

    def set_labeling(self, offset, market):  # 레이블링 폴더 세팅
        '''
        labeling = str # labeling 이름 ex) '4%_01_2'
        forecast = int # 며칠 후를 예측할지 (= forecast interval)

        Labeling
          ㄴ{offset}
            ㄴ{market}
        '''
        self.offset = offset
        self.market = market

        self.labeling = self.top['labeling'] / str(offset) / market

    def set_dataset(self, name):  # 데이터셋 폴더 세팅
        '''
        Dataset
          ㄴ{name}
            ㄴ images
              ㄴtrain / valid / test
            ㄴ labels
        '''
        self.output = self.top['dataset'] / name
        images = self.output / 'images'
        labels = self.output / 'labels'

        self.dataset = {
            'images': {
                'train': images / 'train',
                'valid': images / 'valid',
                'test': images / 'test'
            },
            'labels': labels
        }

    # Classification폴더 내부에 폴더들 생성
    def make_classification(self, labeling=None, dataset=None):
        for classification in self.top.values():
            classification.mkdir(parents=True, exist_ok=True)

        if labeling:
            self.labeling.mkdir(parents=True, exist_ok=True)
        if dataset:
            for image in self.dataset['images'].values():
                image.mkdir(parents=True, exist_ok=True)
            self.dataset['labels'].mkdir(parents=True, exist_ok=True)


class Yolo:  # Yolo 폴더 관리
    def __init__(self, vaiv):
        '''
        Yolo
          ㄴDataset
          ㄴCode
        '''
        self.root = vaiv / 'Yolo'
        self.top = {
            'dataset': self.root / 'Dataset',
            'code': self.root / 'Code'
        }

    # 데이터셋 폴더 세팅
    def set_dataset(self, name, model=None):
        '''
        Dataset
          ㄴ{name}
            ㄴimages
              ㄴtrain / valid / test
            ㄴlabels
              ㄴtrain / valid / test
            ㄴsignals
        '''
        self.output = self.top['dataset'] / name
        images = self.output / 'images'
        labels = self.output / 'labels'
        signals = self.output / 'signals'

        self.dataset = {
            'images': {
                'train': images / 'train',
                'valid': images / 'valid',
                'test': images / 'test'
            },
            'labels': {
                'train': labels / 'train',
                'valid': labels / 'valid',
                'test': labels / 'test'
            },
            'signals': signals
        }

        if model:
            self.model = model
            self.signals = self.dataset['signals'] / model

    # Yolo폴더 내부에 폴더들 생성
    def make_yolo(
        self,
        dataset=None,
        signal=True
    ):
        for yolo in self.top.values():
            yolo.mkdir(parents=True, exist_ok=True)

        if dataset:
            for image in self.dataset['images'].values():
                image.mkdir(parents=True, exist_ok=True)
            for label in self.dataset['labels'].values():
                label.mkdir(parents=True, exist_ok=True)
            if signal:
                self.signals.mkdir(parents=True, exist_ok=True)


class Common:  # Common 폴더 관리
    def __init__(self, vaiv):
        '''
        Common
          ㄴStock
          ㄴPrediction
          ㄴImage
          ㄴServer
          ㄴModel
          ㄴCode
        '''
        self.root = vaiv / 'Common'
        self.top = {
            'stock': self.root / 'Stock',
            'predict': self.root / 'Prediction',
            'image': self.root / 'Image',
            'server': self.root / 'Server',
            'model': self.root / 'Model',
            'code': self.root / 'Code'
        }

        self.candidate = ['Candle', 'Vol', 'MA', 'Vol_MA',
                          'MACD', 'Vol_MACD', 'MA_MACD', 'Vol_MA_MACD']

    def set_stock(self, market):
        '''
        Stock
          ㄴ{market}
        '''
        self.market = market
        self.stock = self.top['stock'] / self.market

    # Prediction 폴더 세팅 -> 예측일 기준으로 필요한 정보 담김
    def set_prediction(self, candle, market):
        '''
        Prediction
          ㄴ{candle}
            ㄴ{market}
        '''
        self.pred = self.top['predict'] / candle / market

    # 차트 이미지 폴더 세팅
    def set_image(
        self,
        feature: dict,  # {'Volume': bool, 'MA': [int], 'MACD': bool}
        offset: int,  # 거래일 간격
        market: str,  # 주식 시장
        style: str,  # background 색깔
        size: list,  # [width, height]
        candle: int,  # 캔들 개수
        linespace: float,  # 캔들 간격
        candlewidth: float  # 캔들 두께
    ):
        '''
        Image
          ㄴ {feature}
            ㄴ {style}
              ㄴ {width}x{height}_{candle}_{linespace}_{candlewidth}
                ㄴ {offset}
                  ㄴ{market}
                    ㄴimages
                    ㄴpixels
        '''
        f = 0
        if feature['Volume']:  # volume 있을 때
            f += 1
        if feature['MA'] != [-1]:  # ma 있을 때
            f += 2
        if feature['MACD']:  # macd 있을 때
            f += 4
        f = self.candidate[f]

        root = self.top['image']
        set = f'{size[0]}x{size[1]}_{candle}_{linespace}_{candlewidth}'
        folder = root / f / style / str(offset) / set / market

        self.image = {
            'images': folder / 'images',
            'pixels': folder / 'pixels'
        }

    def set_server(self, server):
        '''
        Server
          ㄴ{app} # MakeDataset, Simulator
            ㄴtemplates
            ㄴstatic
        '''
        self.seroot = self.top['server'] / server
        self.server = {
            'templates': self.seroot / 'templates',
            'static': self.seroot / 'static'
        }

    # Common폴더 내부에 폴더들 생성
    def make_common(
        self,
        stock=None,
        prediction=None,
        image=None,
        server=None
    ):
        for common in self.top.values():
            common.mkdir(parents=True, exist_ok=True)

        for ft in self.candidate:
            (self.top['image'] / ft).mkdir(parents=True, exist_ok=True)

        if prediction:
            self.pred.mkdir(parents=True, exist_ok=True)

        if image:
            self.image['images'].mkdir(parents=True, exist_ok=True)
            self.image['pixels'].mkdir(parents=True, exist_ok=True)

        if server:
            self.server['templates'].mkdir(parents=True, exist_ok=True)
            self.server['static'].mkdir(parents=True, exist_ok=True)

        if stock:
            self.stock.mkdir(parents=True, exist_ok=True)


class VAIV(FileManager):
    def __init__(self, vaiv):
        self.vaiv = Path(vaiv)

        self.classification = Classification(self.vaiv)
        self.yolo = Yolo(self.vaiv)
        self.common = Common(self.vaiv)
        self.kwargs = {}

        # DataFrame들
        self.modedf = {
            'label': None,
            'train': None, 'valid': None, 'test': None,
            'pixel': None,
            'predict': None,
            'signal': None, 'total': None,
            'info': None,
            'stock': None,
            'Kospi': None, 'Kosdaq': None,
        }

        # Load한 DataFrame Path들
        self.load = {
            'label': None,
            'train': None, 'valid': None, 'test': None,
            'pixel': None,
            'predict': None,
            'signal': None, 'total': None,
            'info': None,
            'stock': None,
            'Kospi': None, 'Kosdaq': None,
        }

    def set_kwargs(self, **kwargs):
        for k, v in kwargs.items():
            self.kwargs[k] = v

    def load_df(self, mode):
        '''
        [kwargs]
        label: 'ticker', 'trade_date'
        train / valid / test: None
        info: 'folder'
        pixel: 'ticker', 'trade_date'
        signal: 'ticker'
        total: None
        predict: 'ticker'
        stock: 'ticker'
        Kospi / Kosdaq: None
        '''
        index = None
        if mode == 'label':
            empty = {'Date': [], 'Ticker': [], 'Label': []}
            ticker = self.kwargs['ticker']
            trade_date = self.kwargs['trade_date']
            self.set_fname('csv', ticker=ticker, trade_date=trade_date)
            self.set_path(self.classification.labeling)

        elif (mode == 'train') or (mode == 'valid') or (mode == 'test'):
            empty = {'Date': [], 'Ticker': [], 'Label': []}
            self.set_fname('csv', mode=mode)
            self.set_path(self.classification.dataset['labels'])

        elif mode == 'info':
            folder = self.kwargs['folder']
            self.set_fname('csv', mode=mode)
            if folder == 'classification':
                empty = {'Name': [], 'Image': [], 'Labeling': [],
                         'Market': [], 'Train': [], 'Valid': [], 'Test': []}
                self.set_path(self.classification.top['dataset'])
            elif folder == 'yolo':
                empty = {'Name': [], 'Image': [], 'Market': [],
                         'Train': [], 'Valid': [], 'Test': []}
                self.set_path(self.yolo.top['dataset'])

        elif mode == 'pixel':
            empty = {
                'Date': [],
                'Xmin': [], 'Xmax': [],
                'Ymin': [], 'Ymax': [],
            }
            ticker = self.kwargs['ticker']
            trade_date = self.kwargs['trade_date']
            self.set_fname('csv', ticker=ticker, trade_date=trade_date)
            self.set_path(self.yolo.image['pixels'])
            index = 'Date'

        elif mode == 'signal':
            empty = {'Date': [], 'Ticker': [], 'Label': [],
                     'Probability': [], 'Range': [], 'Detect': []}
            ticker = self.kwargs['ticker']
            self.set_fname('csv', ticker=ticker)
            self.set_path(self.yolo.signals)

        elif mode == 'total':
            empty = {'Date': [], 'Ticker': [], 'Label': [],
                     'Probability': [], 'Range': [], 'Detect': []}
            self.set_fname('csv', mode=mode)
            self.set_path(self.yolo.signals)

        elif mode == 'predict':
            empty = {'Date': [], 'Start': [], 'End': [], 'Close': []}
            ticker = self.kwargs['ticker']
            self.set_fname('csv', ticker=ticker)
            self.set_path(self.yolo.pred)
            index = 'Date'

        elif mode == 'stock':
            empty = {
                'Date': [],
                'Open': [], 'Close': [],
                'High': [], 'Low': [],
                'Volume': []
            }
            ticker = self.kwargs['ticker']
            self.set_fname('csv', ticker=ticker)
            self.set_path(self.common.stock)
            index = 'Date'

        elif (mode == 'Kospi') or (mode == 'Kosdaq'):
            empty = {'Ticker': [], 'Name': []}
            self.set_fname('csv', market=mode)
            self.set_path(self.common.top['stock'])

        else:
            return

        self.load[mode] = self.path
        super().load_df(empty, index=index)
        self.modedf[mode] = self.df

    def set_df(self, mode, df):
        super().set_df(df)
        self.modedf[mode] = self.df

    def save_df(self, mode):
        self.path = self.load[mode]
        self.df = self.modedf.get(mode)
        super().save_df()

    def set_image(self):
        style = self.kwargs.get('style')
        offset = self.kwargs.get('offset')
        market = self.kwargs.get('market')
        size = self.kwargs.get('size')
        candle = self.kwargs.get('candle')
        feature = self.kwargs.get('feature')
        linespace = self.kwargs.get('linespace')
        candlewidth = self.kwargs.get('candlewidth')

        self.common.set_image(
            feature, offset,
            market, style,
            size, candle,
            linespace, candlewidth,
        )

    def set_labeling(self):
        offset = self.kwargs.get('offset')
        market = self.kwargs.get('market')
        self.classification.set_labeling(offset, market)

    def set_dataset(self):
        '''
        folder: 'Classification' 또는 'Yolo' 폴더
        '''
        name = self.kwargs.get('name')
        folder = self.kwargs.get('folder')

        if folder == 'Classification':
            self.classification.set_dataset(name)
        elif folder == 'Yolo':
            model = self.kwargs.get('model')
            self.yolo.set_dataset(name, model)
        else:
            print('There is no such folder')

    def set_prediction(self):
        candle = self.kwargs.get('candle')
        market = self.kwargs.get('market')
        self.yolo.set_prediction(candle, market)

    def set_stock(self):
        market = self.kwargs.get('market')
        self.common.set_stock(market)

    def set_server(self):
        server = self.kwargs.get('server')
        self.common.set_server(server)

    def make_dir(
        self,
        classification=None, labeling=None, dataset=None,
        yolo=None, signal=True,
        common=None, stock=None, prediction=None, image=None, server=None
    ):
        if classification:
            self.classification.make_classification(
                labeling=labeling,
                dataset=dataset
            )
        if yolo:
            self.yolo.make_yolo(dataset=dataset, signal=signal)
        if common:
            self.common.make_common(
                stock=stock,
                prediction=prediction,
                image=image,
                server=server
            )


if __name__ == '__main__':
    ROOT = Path('/home/ubuntu/2022_VAIV_Cho/VAIV')
    vaiv = VAIV(ROOT)
    vaiv.make_dir(classification=True, yolo=True, common=True)
