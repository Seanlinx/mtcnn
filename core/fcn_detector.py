import mxnet as mx
import numpy as np

from config import config

class FcnDetector(object):
    def __init__(self, symbol, ctx=None,
                 arg_params=None, aux_params=None):
        self.symbol = symbol
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        self.arg_params = arg_params
        self.aux_params = aux_params
        self.output_dict = None

    def predict(self, databatch):
        data_shape = {'data': databatch.shape}
        self.arg_params['data'] = mx.nd.array(databatch, self.ctx)

        arg_shapes, out_shape, aux_shapes = self.symbol.infer_shape(**data_shape)
        arg_shapes_dict = dict(zip(self.symbol.list_arguments(), arg_shapes))

        self.executor = self.symbol.bind(self.ctx, self.arg_params, args_grad=None,
                                         grad_req='null', aux_states=self.aux_params)

        self.executor.forward(is_train=False)
        outputs = self.executor.outputs

        return outputs
