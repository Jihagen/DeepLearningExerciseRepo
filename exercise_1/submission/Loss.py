import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, predictions, targets):
        """
        Compute the cross-entropy loss between predictions and targets.
        
        :param predictions: Predicted probabilities (softmax output).
        :param targets: True labels (one-hot encoded).
        :return: Computed cross-entropy loss.
        """
        
        # Clip predictions to avoid log(0)
        epsilon = 1e-12
        predictions = np.clip(predictions, epsilon, 1 - epsilon) #remember predictions are btw 0-1 anyways
        self.predictions = predictions
        # Compute cross-entropy loss
        loss = -np.sum(targets * np.log(predictions)) / predictions.shape[0]
        return loss
    
    def backward(self,label_tensor):
        """
        Compute the gradient of the cross-entropy loss with respect to predictions.
        
        :param predictions: Predicted probabilities (softmax output).
        :param targets: True labels (one-hot encoded).
        :return: Gradient of the loss with respect to predictions.
        """
        return (self.predictions - label_tensor) / self.predictions.shape[0] 
        #remember: 
        # cross entropy + softmax trick
        # cross entropy only focuses on the correct class, by increasing the probability of the correct class, 
        # we decrease probabilities of the other classes automatically -> which is decreasing loss overall
        # trick: gradient = predictions-label_tensor (one-hot)
     