from collections import OrderedDict
import theano
import numpy
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams

theano.config.floatX = 'float32'

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk],name=kk) #将params的参数设为共享变量，在每个线程上均能使用
                               #在程序中一般把神经网络的参数W、b等定义为共享变量，因为网络的参数，基本上是每个线程都需要访问的。
    return tparams

def norm_weight(nin, nout=None, scale=0.9, ortho=True):
    if nout is None:
        nout = nin
    W =  scale*numpy.random.randn(nin, nout) # 任意生成一个nin*nout的numpy矩阵，将里面的所有项scale化
    return W.astype(theano.config.floatX)

params = OrderedDict()
params['Wemb'] = norm_weight(9, 2)  # 随机生成一个float32数据类型的 6*2的 所有项被scale化 编码器的嵌入矩阵
tparams = init_tparams(params)

x = numpy.array([[0, 1,2,3 ],
                 [4,5,6,7]])
x = theano.shared(x)

emb = tparams['Wemb'][x.flatten()]
emb = emb.reshape([2, 4, 2])  # xt
print(emb.eval())

# emb1 = RandomStreams.shuffle_row_elements(emb,0)
# emb = tensor.permute_row_elements(emb,[[0,1]])
# emb1 = RandomStreams.permutation(emb,size=(3,2,2))
# print(emb1.eval())

# emb = emb.eval()
# numpy.random.shuffle(emb)
# emb = numpy.random.permutation(emb)

W1 = norm_weight(2, 2)
params['encoder_W1'] = W1  # Wji
params['encoder_b1'] = numpy.zeros((2,)).astype(theano.config.floatX)  # θj

W2 = norm_weight(2, 2)
params['encoder_W2'] = W2  # Wji
params['encoder_b2'] = numpy.zeros((2,)).astype(theano.config.floatX)  # θj

state_below = tensor.dot(emb, params['encoder_W1']) + params['encoder_b1']
state_below_ = tensor.tanh(state_below)  # 采用tanh()作为激活函数
state_below_ = tensor.dot(state_below_,params['encoder_W2']) +params['encoder_b2']
state_below__ = tensor.tanh(state_below_)  # 3*2*2
# print(state_below__.eval())
context_x = state_below__.mean(axis=0)
#print(context_x.eval())

Wx = norm_weight(2, 2)
params['encoder_Wx'] = Wx
params['encoder_bx'] = numpy.zeros((2,)).astype(theano.config.floatX)

context_x = tensor.dot(context_x,params['encoder_Wx']) +params['encoder_bx']
print(tensor.nnet.sigmoid(context_x).eval())
emb = emb * tensor.nnet.sigmoid(context_x)
print(emb.eval())