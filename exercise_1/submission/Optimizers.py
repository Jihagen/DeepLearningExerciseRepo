class SDG:
    def __init__(self, learning_rate:float = 0.01):
        self.learning_rate = learning_rate
    
    def calculate_update(self, weight_tensor,gradient_tensor):
        """
        Calculate the update for the weight tensor based on the gradient tensor.
        
        Args:
            weight_tensor (torch.Tensor): The tensor containing the weights.
            gradient_tensor (torch.Tensor): The tensor containing the gradients.
        
        Returns:
            torch.Tensor: The updated weight tensor.
        """
        return weight_tensor - self.learning_rate * gradient_tensor