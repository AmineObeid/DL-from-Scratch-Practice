import numpy as np

class Tensor:
    # So far accepting only numpy arrays of dtype float 32 for simplicity
    def __init__(self, data, grad_flag: bool = False, grad = None, parents = None, op = None):

        if isinstance(data, np.ndarray) and data.dtype == np.float32:
            self.data = data
            self.shape = data.shape
            self.dtype = np.float32
        else:
            raise TypeError("Data must be a numpy array of dtype float32")
              
        self.grad_flag = grad_flag
        self.grad = grad
        self.parents = parents
        self.op = op
        self.backward_fn = None

    def __add__(self, other):
        data = np.add(self.data, other.data)
        parents = [self, other]
        op = "add"

        result = Tensor(data=data,
                        parents=parents,
                        op=op)
        
        return result

    def __mul__(self, other):
        data = np.mul(self.data, other.data)
        parents = [self, other]
        op = "mul"

        result = Tensor(data=data,
                        parents=parents,
                        op=op)
        
        return result
    
    def __matmul__(self, other):
        data = np.matmul(self.data, other.data)
        parents = [self, other]
        op = "matmul"

        result = Tensor(data=data,
                        parents=parents,
                        op=op)
        
        return result

    def __neg__(self):
        data = np.negative(self.data)
        parents = [self]
        op = "neg"

        result = Tensor(data=data,
                        parents=parents,
                        op=op)
        
        return result
    
    def __sub__(self, other):
        return self.__add__(self, self.__neg__(other))
        
    def __repr__(self):
        pass #TODO: Implement