from itertools import chain
from math import ceil
from PIL import Image

m, n = (5, 5)                                   # Checker dimension (x, y)
w, h = (100, 100)                               # Final image width and height

c1 = 0      # (224, 64, 64)                     # First color
c2 = 255    # (128, 128, 128)                   # Second color
mode = 'L' if isinstance(c1, int) else 'RGB'    # Mode from first color

# Generate pixel-wise checker, even x dimension
if m % 2 == 0:
    pixels = [[c1, c2] for i in range(int(m/2))] + \
             [[c2, c1] for i in range(int(m/2))]
    pixels = [list(chain(*pixels)) for i in range(ceil(n/2))]

# Generate pixel-wise checker, odd x dimension
else:
    pixels = [[c1, c2] for i in range(ceil(m*n/2))]

# Generate final Pillow-compatible pixel values
pixels = list(chain(*pixels))[:(m*n)]

# Generate Pillow image from pixel values, resize to final image size, and save
checker = Image.new(mode, (m, n))
checker.putdata(pixels)
checker = checker.resize((w, h), Image.NEAREST)
checker.save('img_checker.png')
