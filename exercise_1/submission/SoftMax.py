import numpy as np

class SoftMax:
    def __init__(self):
        self.input_tensor = None
        self.output_tensor = None

    def forward(self, input_tensor):
        """
        Forward pass through the softmax layer.

        Args:
            input_tensor (np.ndarray): Input tensor of shape (batch_size, num_classes).

        Returns:
            np.ndarray: Output tensor of shape (batch_size, num_classes) after applying softmax.
        """
        self.input_tensor = input_tensor
        exp_values = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True)) #logits to probability with e^zi / sum(e^zj)
        self.output_tensor = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output_tensor
    
    def backward(self, error_tensor):
        """
        Vectorized backward pass through the softmax layer.

        Args:
            error_tensor (np.ndarray): Gradient of the loss w.r.t. the softmax output, shape (B, C)

        Returns:
            np.ndarray: Gradient of the loss w.r.t. the softmax input, shape (B, C)
        """
        # s: (B, C)
        s = self.output_tensor  # softmax output
        B, C = s.shape # batch size, number of classes

        # Step 1: Compute the outer product of softmax output with itself
        s_reshaped = s[..., np.newaxis]  # shape (B, C, 1)
        s_outer = s_reshaped * s_reshaped.transpose(0, 2, 1)  # shape (B, C, C)

        # Step 2: Create identity matrix scaled by softmax values (diagonal: s_i * (1 - s_i))
        eye = np.eye(C)[np.newaxis, :, :]              # shape (1, C, C) → broadcast to (B, C, C) 
        diag = eye * s_reshaped.squeeze(-1)            # shape (B, C, C)  

        # Step 3: Construct Jacobian for each sample: J = diag - outer 
        # [Needed because any change in logit z affects all other logits]
        jacobian = diag - s_outer                      # shape (B, C, C)

        # Step 4: Apply Jacobian to error tensor (B, C) → (B, C)
        grad_input = np.einsum('bij,bj->bi', jacobian, error_tensor)

        return grad_input
        

