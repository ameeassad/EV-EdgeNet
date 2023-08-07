import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils.Loader_v2 import Loader

loader = Loader('../datasets/processed-fixed', problemType='edges', n_classes=30, width=640, height=480,
                median_frequency=0.00, channels=3, channels_events=6)

x, y, mask = loader.get_batch(size=1, train=True)
loader.batchex_printer(x, y, mask, rgb= True, predicted=False)