import numpy as np

class Activation_Fn:    
    def __init__(self, x):
        self.x = x
    def forward(self):
        raise NotImplementedError
    def backward(self): # To use in the backpropagation
        raise NotImplementedError

class ReLU(Activation_Fn):
    def forward(self):
        return np.max(0, self.x) #TODO: add axis
    def backward(self):
        return np.where(self.x > 0, 1, 0)
    
class Sigmoid(Activation_Fn):
    def forward(self):
        return 1/(1 + np.exp(-(self.x)))
    def backward(self):
        forward_value = self.forward() # To compute it once and not twice, efficiency
        return forward_value * (1 - forward_value)

class Softmax(Activation_Fn):
    def forward(self):
        e_x = np.exp(self.x)
        return e_x / np.sum(e_x) #TODO: add axis
    def backward(self):
        return super().backward() #TODO: Calculate and implement

class Tanh(Activation_Fn):
    def forward(self):
        return np.tanh(self.x)
    def backward(self):
        return 1/np.cosh(self.x)**2

