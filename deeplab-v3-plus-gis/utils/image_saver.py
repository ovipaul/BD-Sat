import torch
from matplotlib import pyplot as plt
import numpy as np
  
def image_saver(tar_img, index, img_name):
    if tar_img.shape[0] == 2:
        tar_img = tar_img[index:index+1,:,:]
    tar_img = np.squeeze(tar_img, axis=0)
    tar_img = decode_segmap(tar_img)
    plt.imsave('Images/Target/'+img_list[img_counter]+'.png',tar_img)

    return rgb
    