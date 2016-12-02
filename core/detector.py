import mxnet as mx
import numpy as np

from config import config

class Detector(object):
    def __init__(self, symbol, data_size, batch_size, ctx=None,
                 arg_params=None, aux_params=None):
        self.symbol = symbol
        self.data_size = data_size
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        self.arg_params = arg_params
        self.aux_params = aux_params

        self.batch_size = batch_size
        data_shapes = {'data': (self.batch_size, 3, self.data_size, self.data_size)}
        executor = self.symbol.simple_bind(self.ctx, grad_req='null', **dict(data_shapes))
        executor.copy_params_from(self.arg_params, self.aux_params)
        self.executor = executor

        self.output_dict = None
        self.data_shape = data_shapes
        self.t = 0


    def predict(self, databatch):
        # access data
        # databatch: N x 3 x data_size x data_size
        scores = []
        batch_size = self.batch_size

        minibatch = []
        cur = 0
        n = databatch.shape[0]
        while cur < n:
            minibatch.append(databatch[cur:min(cur+batch_size, n), :, :, :])
            cur += batch_size

        data_arrays = self.executor.arg_dict['data']
        out_list = [[] for _ in range(len(self.executor.outputs))]

        for idx, data in enumerate(minibatch):
            m = data.shape[0]
            real_size = self.batch_size
            if m < batch_size:
                keep_inds = np.arange(m)
                gap = self.batch_size - m
                while gap >= len(keep_inds):
                    gap -= len(keep_inds)
                    keep_inds = np.concatenate((keep_inds, keep_inds))
                if gap != 0:
                    keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
                data = data[keep_inds]
                real_size = m

            data_arrays[:] = data
            self.executor.forward(is_train=False)

            for o_list, o_nd in zip(out_list, self.executor.outputs):
                o_list.append(o_nd[0:real_size].asnumpy())

        out = list()

        for o in out_list:
            out.append(np.vstack(o))

        return out
