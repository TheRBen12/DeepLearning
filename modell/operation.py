import numpy as np
from numpy import ndarray


class Operation(object):
    '''
    Basisklasse für eine Operation in einem neuronalen Netz
    '''

    def __init__(self):
        pass

    def forward(self, input_: ndarray) -> ndarray:
        self.input_ = input_

        self.output = self._output()

        return self.output

    def backward(self, out_put_grad: ndarray) -> ndarray:
        self.input_grad = self._input_grad(out_put_grad)
        return self.input_grad

    def _output(self) -> ndarray:
        raise NotImplementedError()

    def _input_grad(self, out_put_grad: ndarray) -> ndarray:
        '''muss für jede Operation implementiert werden'''
        raise NotImplementedError()
