import numpy as np

class Loss_Fn:
    def __init__(self, predictions, target):
        self.predictions = predictions
        self.target = target
    
    def calculate_loss(self):
        raise NotImplementedError

class MSE(Loss_Fn):
    def calculate_loss(self):
        return (1 / self.predictions.shape[0]) * np.sum(((np.subtract(self.target - self.predictions))**2)) #TODO: add axis
    
class MAE(Loss_Fn):
    def calculate_loss(self):
        return (1 / self.predictions.shape[0]) * np.sum(((np.absolute(self.target - self.predictions)))) #TODO: add axis
    
class BCE(Loss_Fn):
    def calculate_loss(self):
        return (-1 / self.predictions.shape[0]) * np.sum((self.target * np.log(self.predictions) + (1 - self.target) * np.log(1-self.predictions))) #TODO: add axis

class CrossEntropyLoss(Loss_Fn):
    def calculate_loss(self):
        return super().calculate_loss() #TODO: Implement

class HuberLoss(Loss_Fn):
    def calculate_loss(self):
        return super().calculate_loss() #TODO: Implement

class HingeLoss(Loss_Fn):
    def calculate_loss(self):
        return np.max(0, 1 - self.target * self.predictions) #TODO: add axis
    