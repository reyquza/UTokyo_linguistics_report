"""A small MLP for the plausibility experiment.

Reuses layers and optimization code from the book-based implementations
at the repository root (Deep Learning from Scratch 2).
"""

import numpy as np

from common.layers import Affine, Sigmoid, Softmax_with_Loss


class TwoLayerMLP:
    """A two-layer classifier: Affine -> Sigmoid -> Affine."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int = 2):
        # Xavier-style initialization for sigmoid activations
        w1 = (
            np.random.randn(input_size, hidden_size)
            * np.sqrt(1.0 / input_size)
        ).astype(np.float32)
        b1 = np.zeros(hidden_size, dtype=np.float32)
        w2 = (
            np.random.randn(hidden_size, output_size)
            * np.sqrt(1.0 / hidden_size)
        ).astype(np.float32)
        b2 = np.zeros(output_size, dtype=np.float32)

        self.layers = [
            Affine(w1, b1),
            Sigmoid(),
            Affine(w2, b2),
        ]
        self.loss_layer = Softmax_with_Loss()

        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x: np.ndarray, labels: np.ndarray) -> float:
        logits = self.predict_logits(x)
        return float(self.loss_layer.forward(logits, labels))

    def backward(self, dout: float = 1.0) -> None:
        grad = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

