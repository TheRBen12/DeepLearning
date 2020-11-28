from modell.operation import Operation
import numpy as np
from numpy import exp, ndarray


class Sigmoid(Operation):

    def __init__(self) -> None:
        super().__init__()

    def _output(self) -> ndarray:
        return 1.0 / (1.0 + np.exp(-1.0 * self.input_))

    def _input_grad(self, out_put_grad: ndarray) -> ndarray:
        sigmoid_backward = self.output * (1.0 * -self.output)
        input_grad = sigmoid_backward * out_put_grad
        return input_grad
