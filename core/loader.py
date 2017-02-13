import mxnet as mx
import numpy as np
import minibatch
from config import config

class TestLoader(mx.io.DataIter):
    def __init__(self, imdb, batch_size=1, shuffle=False):
        self.imdb = imdb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = len(imdb)
        self.index = np.arange(self.size)

        self.cur = 0
        self.data = None
        self.label = None

        self.data_names = ['data']
        self.label_names = []

        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_names, self.label)]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        imdb = [self.imdb[self.index[i]] for i in range(cur_from, cur_to)]
        data, label = minibatch.get_testbatch(imdb)
        self.data = [mx.nd.array(data[name]) for name in self.data_names]
        self.label = [mx.nd.array(label[name]) for name in self.label_names]

class ImageLoader(mx.io.DataIter):
    def __init__(self, imdb, im_size, batch_size=config.BATCH_SIZE, shuffle=False, ctx=None, work_load_list=None):

        super(ImageLoader, self).__init__()

        self.imdb = imdb
        self.batch_size = batch_size
        self.im_size = im_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list

        self.cur = 0
        self.size = len(imdb)
        self.index = np.arange(self.size)
        self.num_classes = 2

        self.batch = None
        self.data = None
        self.label = None

        self.label_names= ['label', 'bbox_target']
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [('data', self.data[0].shape)]
      #  return [(k, v.shape) for k, v in zip(self.data_name, self.data)]


    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_names, self.label)]


    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        imdb = [self.imdb[self.index[i]] for i in range(cur_from, cur_to)]
        data, label = minibatch.get_minibatch(imdb, self.num_classes, self.im_size)
        self.data = [mx.nd.array(data['data'])]
        self.label = [mx.nd.array(label[name]) for name in self.label_names]
