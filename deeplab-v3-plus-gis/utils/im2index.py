import numpy as np
def im2index(im):
    
    color2index = {
        (0 , 0, 0) : 0,
        (0, 255, 255) : 1,
        (255, 0, 0) : 2,
        (0, 0, 255) : 3,
        (0, 255, 0) : 4,
        (255, 255, 0) : 5
    }

    """
    turn a 3 channel RGB image to 1 channel index image
    """
    assert len(im.shape) == 3
    height, width, ch = im.shape
    assert ch == 3
    m_lable = np.zeros((height, width, 1), dtype=np.uint8)
    for w in range(width):
        for h in range(height):
            b, g, r = im[h, w, :]
            m_lable[h, w, :] = color2index[(r, g, b)]
    return m_lable