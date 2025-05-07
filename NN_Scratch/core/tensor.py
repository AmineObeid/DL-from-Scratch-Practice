import numpy as np

class Tensor:
    # So far accepting only numpy arrays of dtype float 32 for simplicity
    def __init__(self, data, grad_flag: bool = False, grad = None):

        if isinstance(data, np.ndarray) and data.dtype == np.float32:
            self.data = data
            self.shape = data.shape
            self.dtype = np.float32
        else:
            raise TypeError("Data must be a numpy array of dtype float32")
              
        self.grad_flag = grad_flag
        self.grad = grad

    def __repr__(self):
        pass #TODO: Implement