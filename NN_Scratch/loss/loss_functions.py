import numpy as np

class Loss_Fn:
    def __init__(self, predictions, target):
        self.predictions = predictions
        self.target = target
    
    def calculate_loss(self):
        raise NotImplementedError

class MSE(Loss_Fn):
    def calculate_loss(self):
        return (1 / self.predictions.shape[0]) * np.sum(((np.subtract(self.target - self.predictions))**2)) #add axis
    
class MAE(Loss_Fn):
    def calculate_loss(self):
        return (1 / self.predictions.shape[0]) * np.sum(((np.absolute(self.target - self.predictions)))) #add axis
    
class BCE(Loss_Fn):
    pass

class CrossEntropyLoss(Loss_Fn):
    pass

class HuberLoss(Loss_Fn):
    pass

class HingeLoss(Loss_Fn):
    pass
    