import numpy as np
import glob
from PIL import Image
import re

for f in glob.glob('./*.npy'):
    dat = np.load('label_100.npy')
    # dat = dat[..., np.newaxis]

    m = re.match('[^0-9]+(?P<n>[0-9]+).npy', f)

    im = Image.fromarray(dat.astype(np.uint8))
    im.save('rec1487417411_export_' + m['n'] + ".png")

