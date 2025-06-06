from Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    """
    FullyConnected layer class.
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0, 1, (self.input_size, self.output_size))
    
    def forward(self, input_tensor):
        """
        Forward pass through the fully connected layer.
        
        Args:
            input_tensor (np.ndarray): Input tensor of shape (batch_size, input_size).
        
        Returns:
            np.ndarray: Output tensor of shape (batch_size, output_size).
        """
        self.input_tensor = input_tensor  # Store input for potential backpropagation
        return np.dot(input_tensor, self.weights)
    
    # Note: properties are protected attributes that can be accessed like attributes but are actually methods.
    # @property is used to define a getter for the optimizer property,
    # @optimizer.setter is used to define a setter for the optimizer property.

    @property 
    def optimizer(self):
        """
        Getter for the optimizer property.
        
        Returns:
            Optimizer: The optimizer associated with this layer.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """
        Setter for the optimizer property.
        
        Args:
            optimizer: The optimizer to be associated with this layer.
        """
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        """
        Getter for the gradient_weights property.
        
        Returns:
            np.ndarray: The gradient with respect to the weights.
        """
        return self._gradient_weights

    def backward(self, error_tensor):
        """
        Backward pass through the fully connected layer.
        
        Args:
            error_tensor (np.ndarray): Error tensor from the next layer.
        
        Returns:
            np.ndarray: Gradient with respect to the input tensor.
        """
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if hasattr(self, '_optimizer') and self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        return np.dot(error_tensor, self.weights.T)