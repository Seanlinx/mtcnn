import numpy as np

def transform(im):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :return: [batch, channel, height, width]
    """
    im_tensor = im.transpose(2, 0, 1)
    im_tensor = im_tensor[np.newaxis, :]
    im_tensor = (im_tensor - 127.5)*0.0078125
    return im_tensor
