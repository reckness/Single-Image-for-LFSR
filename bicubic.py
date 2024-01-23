from fileinput import filename
from utils.imresize import *
import os

from skimage import io

PATH = 'D:\workspace\Aberration_correction_metasensor\Data\sub_image\RGB_1'
SAVE_PATH = 'D:\workspace\Aberration_correction_metasensor\Data\sub_image\RGB_1_bicubic_4'
filenames = os.listdir(PATH)

os.mkdir(SAVE_PATH)

for f in filenames:
    img = io.imread(os.path.join(PATH, f))
    tmp = imresize(img, scalar_scale=4)
    io.imsave(os.path.join(SAVE_PATH, f), tmp)
    

# TMP_LF[u, v, :, :, :] = imresize(LF[u, v, :, :, :], scalar_scale=scale_factor)
