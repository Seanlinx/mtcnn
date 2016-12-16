import mxnet as mx
import negativemining
from config import config

def P_Net(mode='train'):
    """
    Proposal Network
    input shape 3 x 12 x 12
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")

    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=10, name="conv1")
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    pool1 = mx.symbol.Pooling(data=prelu1, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")

    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3, 3), num_filter=16, name="conv2")
    prelu2 = mx.symbol.LeakyReLU(data=conv2, act_type="prelu", name="prelu2")

    conv3 = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), num_filter=32, name="conv3")
    prelu3 = mx.symbol.LeakyReLU(data=conv3, act_type="prelu", name="prelu3")

    conv4_1 = mx.symbol.Convolution(data=prelu3, kernel=(1, 1), num_filter=2, name="conv4_1")
    conv4_2 = mx.symbol.Convolution(data=prelu3, kernel=(1, 1), num_filter=4, name="conv4_2")

    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=conv4_1, mode="channel", name="cls_prob")
        bbox_pred = conv4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])

    else:
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1, label=label,
                                           multi_output=True, use_ignore=True,
                                           out_grad=True, name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = conv4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     grad_scale=1, out_grad=True, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, bbox_pred=bbox_pred,
                               label=label, bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        group = mx.symbol.Group([out])
    return group


def R_Net(mode='train'):
    """
    Refine Network
    input shape 3 x 24 x 24
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")

    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=28, name="conv1")
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    pool1 = mx.symbol.Pooling(data=prelu1, pool_type="max", kernel=(3, 3), stride=(2, 2), name="pool1")

    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3, 3), num_filter=48, name="conv2")
    prelu2 = mx.symbol.LeakyReLU(data=conv2, act_type="prelu", name="prelu2")
    pool2 = mx.symbol.Pooling(data=prelu2, pool_type="max", kernel=(3, 3), stride=(2, 2), name="pool2")

    conv3 = mx.symbol.Convolution(data=pool2, kernel=(2, 2), num_filter=64, name="conv3")
    prelu3 = mx.symbol.LeakyReLU(data=conv3, act_type="prelu", name="prelu3")

    fc1 = mx.symbol.FullyConnected(data=prelu3, num_hidden=128, name="fc1")
    prelu4 = mx.symbol.LeakyReLU(data=fc1, act_type="prelu", name="prelu4")

    fc2 = mx.symbol.FullyConnected(data=prelu4, num_hidden=2, name="fc2")
    fc3 = mx.symbol.FullyConnected(data=prelu4, num_hidden=4, name="fc3")

    cls_prob = mx.symbol.SoftmaxOutput(data=fc2, label=label, use_ignore=True,
                                       out_grad=True, name="cls_prob")
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxOutput(data=fc2, label=label, use_ignore=True, name="cls_prob")
        bbox_pred = fc3
        group = mx.symbol.Group([cls_prob, bbox_pred])
    else:
        bbox_pred = mx.symbol.LinearRegressionOutput(data=fc3, label=bbox_target,
                                                     out_grad=True, grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, bbox_pred=bbox_pred, label=label,
                               bbox_target=bbox_target, op_type='negativemining', name="negative_mining")

        group = mx.symbol.Group([out])
    return group


def O_Net(mode="train"):
    """
    Refine Network
    input shape 3 x 48 x 48
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")

    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=32, name="conv1")
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    pool1 = mx.symbol.Pooling(data=prelu1, pool_type="max", kernel=(3, 3), stride=(2, 2), name="pool1")

    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3, 3), num_filter=64, name="conv2")
    prelu2 = mx.symbol.LeakyReLU(data=conv2, act_type="prelu", name="prelu2")
    pool2 = mx.symbol.Pooling(data=prelu2, pool_type="max", kernel=(3, 3), stride=(2, 2), name="pool2")

    conv3 = mx.symbol.Convolution(data=pool2, kernel=(3, 3), num_filter=64, name="conv3")
    prelu3 = mx.symbol.LeakyReLU(data=conv3, act_type="prelu", name="prelu3")
    pool3 = mx.symbol.Pooling(data=prelu3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3")

    conv4 = mx.symbol.Convolution(data=pool3, kernel=(2, 2), num_filter=128, name="conv4")
    prelu4 = mx.symbol.LeakyReLU(data=conv4, act_type="prelu", name="prelu4")

    fc1 = mx.symbol.FullyConnected(data=prelu4, num_hidden=256, name="fc1")
    prelu5 = mx.symbol.LeakyReLU(data=fc1, act_type="prelu", name="prelu5")

    fc2 = mx.symbol.FullyConnected(data=prelu5, num_hidden=2, name="fc2")
    fc3 = mx.symbol.FullyConnected(data=prelu5, num_hidden=4, name="fc3")

    cls_prob = mx.symbol.SoftmaxOutput(data=fc2, label=label, use_ignore=True, out_grad=True, name="cls_prob")
    if mode == "test":
        bbox_pred = fc3
        group = mx.symbol.Group([cls_prob, bbox_pred])
    else:
        bbox_pred = mx.symbol.LinearRegressionOutput(data=fc3, label=bbox_target,
                                                     grad_scale=1, out_grad=True, name="bbox_pred")
        out = mx.symbol.Custom(cls_prob=cls_prob, bbox_pred=bbox_pred, label=label,
                               bbox_target=bbox_target, op_type='negativemining', name="negative_mining")
        group = mx.symbol.Group([out])
    return group
