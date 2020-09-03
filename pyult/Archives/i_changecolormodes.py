import argparse
parser = argparse.ArgumentParser(description='Converts color modes from grayscale to RGB if a source file is defined by grayscale. Otherwise, the source image is converted from RGB to grayscale.')
parser.add_argument('path', help='Path to the image file whose color modes you want to change.')
args = parser.parse_args()

import cv2
import numpy as np
import pyult.pyult as pyult
import pyult.splfit as splfit
from pathlib import Path

pdir = str(Path(args.path).parent)
fname = str(Path(args.path).stem)
sfx = str(Path(args.path).suffix)

def is_grayscale ( rgb_mat ):
    return np.all(np.apply_along_axis(lambda x: len(set(x))==1, 2, rgb_mat))

upc = pyult.UltPicture()
img = upc.read_img(args.path, grayscale=False)
if is_grayscale(img):
    img = upc.read_img(args.path, grayscale=True)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    opath = '{}/{}_color{}'.format(pdir, fname, sfx)
    flg = True
    cnt = 1
    while flg:
        if Path(opath).exists():
            opath = '{}/{}_color-{}{}'.format(pdir, fname, cnt, sfx)
            cnt = cnt + 1
        else:
            flg = False
else:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    opath = '{}/{}_gray{}'.format(pdir, fname, sfx)
    flg = True
    cnt = 1
    while flg:
        if Path(opath).exists():
            opath = '{}/{}_gray-{}{}'.format(pdir, fname, cnt, sfx)
            cnt = cnt + 1
        else:
            flg = False

upc.save_img(path=opath, img=img)
