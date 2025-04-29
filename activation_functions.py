import numpy as np

class Activation_Fn:    
    def __init__(self, x):
        self.x = x
    def forward(self, x):
        raise NotImplementedError
    def backward(self, x):
        raise NotImplementedError

class ReLU(Activation_Fn):
    def forward(self):
        return np.maximum(0, self.x)
    def backward(self, x):
        return super().backward(x)
    
class Sigmoid(Activation_Fn):
    def forward(self):
        return 1/(1 + np.exp(-(self.x)))
    def backward(self, x):
        return super().backward(x)

class Softmax(Activation_Fn):
    pass

class Tanh(Activation_Fn):
    def forward(self, x):
        return (np.exp(self.x) - np.exp(-(self.x))) / (np.exp(self.x) + np.exp(-(self.x)))
    def backward(self, x):
        return super().backward(x)

