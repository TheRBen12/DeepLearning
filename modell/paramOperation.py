from numpy import ndarray

from modell.operation import Operation


class ParamOperation(Operation):

    def __init__(self, param: ndarray):
        super.__init__()
        self.param = param

    def backward(self, out_put_grad: ndarray) -> ndarray:
        self.input_grad = self._input_grad(out_put_grad)
        self.param_grad = self._param_grad(out_put_grad)
        return self.input_grad

    def _param_grad(self, out_put_grad: ndarray) -> ndarray:
        raise NotImplementedError()

        pass

