import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import sys

ROOT = Path('/home/ubuntu/2022_VAIV_Cho/VAIV')
sys.path.append(ROOT)
sys.path.append(ROOT / 'Common' / 'Code')

# from manager import VAIV  # noqa: E402
from utils.mpf import candlestick_ohlc  # noqa: E402


fig = plt.figure(figsize=(224/100, 224/100))
ax1 = fig.add_subplot(111)

test = pd.DataFrame({
    'Date': [0, 1],
    'Open': [8010, 7800],
    'High': [8250, 7850],
    'Low': [7750, 7680],
    'Close': [7760, 7810],
})

buy = pd.DataFrame({
    'Date': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Open': [3000, 1600, 2400, 1000, 2000, 850, 2000, 1300, 2200, 700, 1000],
    'High': [3200, 2600, 2600, 3200, 2200, 1800, 2200, 1560, 2400, 1600, 2000],
    'Low': [800, 1400, 1400, 800, 800, 750, 800, 700, 800, 500, 700],
    'Close': [1000, 2400, 1600, 3000, 1000, 900, 1000, 1500, 1200, 900, 1800],
})

sell = pd.DataFrame({
    'Date': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Open': [2000, 2400, 2400, 1000, 1000, 1850, 1000, 2000, 1200, 2200, 2200],
    'High': [2400, 2600, 2600, 3200, 2200, 2800, 1800, 2400, 2400, 2600, 2400],
    'Low': [600, 1400, 1400, 800, 800, 1750, 800, 900, 600, 2000, 1100],
    'Close': [1000, 1600, 1600, 3000, 2000, 1900, 1600, 1800, 2200, 2400, 1400],
})

lines, patches = candlestick_ohlc(
    ax1, test.values, width=0.7,
    colorup='#77d879', colordown='#db3f3f', alpha=None
)

ax1.grid(False)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)
ax1.axis('off')

plt.tight_layout(pad=0)
fig.set_constrained_layout_pads(w_pad=0, h_pad=0)

fig.savefig('test.png')

pil_image = Image.open('test.png')
rgb_image = pil_image.convert('RGB')
rgb_image.save('test.png')
