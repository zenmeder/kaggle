#!/usr/local/bin/ python3
# -*- coding:utf-8 -*-
# __author__ = "zenmeder"

import numpy as np
import pandas as pd
from PIL import Image

labels = pd.read_csv('labels.csv')
labels = {_[0]:_[1] for _ in labels.values}
import subprocess
pictures = subprocess.check_output(['ls','./train']).decode('utf-8').split('\n')
X,y = [], []
for picture in pictures:
    X.append(np.array(Image.open('train/{0}'.format(picture))))
    y.append(labels[picture[:-4]])

