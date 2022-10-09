import numpy as np
from PIL import Image


def make_grid(images):
    cols = len(images) // 2
    rows = cols + 1
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return np.array(grid)