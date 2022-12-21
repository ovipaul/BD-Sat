import torch
from matplotlib import pyplot as plt
import numpy as np
  
def decode_segmap(image, nc=6):

# 0=Unrecognized                          
# 1=Forest, 2=Built-Up, 3=Water, 4=Farmland, 5=Meadow
  label_colors = np.array([(0, 0, 0), (0,255,255), (255,0,0), (0,0,255), (0,255,0), (255,255,0)])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
    
  rgb = np.stack([r, g, b], axis=2)
  return rgb