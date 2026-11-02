"""
In deep learning, rows represent samples and columns represent features.
axis=0 is the sample dimension.
axis=1 and higher describe the contents of each sample.
B : batch_size
V : vocab_size
H : hidden_size
N : corpus_length
W : window_size
C = 2W : context_length

corpus: (N,),  NDArray[np.int64]
    You say goodbye and I say hello.
    -> corpus = [0 1 2 3 4 1 5 6]    
contexts: (B, C), NDArray[np.int64]
target: (B,), NDArray[np.int64]
W_in: (V,H), NDArray[np.float32]
"""
from common.np import np
from common.layers import Embedding, NegativeSamplingLoss

class CBOW:

    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        
        V, H = vocab_size, hidden_size
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(V, H).astype('f')

        self.in_layers = []
        for _ in range(2 * window_size):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)
        
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in
        
    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            # if window_size=2, then the contexts = [[3, 1, 5, 8]
            # Rows are samples; columns are context words.      [2, 3, 4, 1]]
            h += layer.forward(contexts[:, i])
            # layer.forward(...) : (B, H)
            # h                  : (B, H)
            # contexts[:, i] -> (batch_size, )
        h /= len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss
    
    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout /= len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None
