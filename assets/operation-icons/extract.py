"""Split the FigureGen sheet into the eight runtime icons (requires Pillow)."""
from pathlib import Path
from PIL import Image, ImageChops, ImageOps

folder = Path(__file__).resolve().parent
source = Image.open(folder / 'figuregen-operations01.png').convert('RGB')
names = ['linear', 'attention', 'convolution', 'normalization', 'activation', 'pooling', 'dropout', 'recurrent']
output = folder.parent.parent / 'public' / 'assets' / 'operations'
output.mkdir(parents=True, exist_ok=True)
width, height = source.width // 4, source.height // 2
for index, name in enumerate(names):
    x, y = index % 4 * width, index // 4 * height
    tile = source.crop((x, y, x + width, y + height))
    # Find the ink bounds, then keep even whitespace around every pictogram.
    ink = ImageChops.difference(tile, Image.new('RGB', tile.size, 'white')).convert('L').point(lambda p: 255 if p > 80 else 0)
    tile = tile.crop(ink.getbbox())
    side = max(tile.size) + 24
    tile = ImageOps.pad(tile, (side, side), color='white')
    tile.resize((192, 192), Image.Resampling.LANCZOS).save(output / f'{name}.png', optimize=True)
print(f'Extracted {len(names)} icons.')
