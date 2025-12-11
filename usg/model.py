"""
Neural Network Model Definition

This module contains the custom TensorFlow/Keras model used for
building attribute prediction.
"""

import tensorflow as tf
from tensorflow.keras import layers


class ScaledInputMaskedNN(tf.keras.Model):
    """
    Custom neural network that handles masked inputs for predicting
    missing building attributes.
    
    The model takes two inputs:
    - x: The feature values (with 0 for missing values)
    - mask: Binary mask indicating which features are known (1) vs missing (0)
    """
    
    def __init__(self, input_dim, hidden_units=[128, 64], **kwargs):
        """
        Initialize the masked neural network.
        
        Args:
            input_dim: Number of input features
            hidden_units: List of hidden layer sizes
            **kwargs: Additional keras.Model arguments
        """
        super().__init__(**kwargs)
        
        # Create hidden layers
        self.hidden_layers = [
            layers.Dense(units, activation='relu', name=f'hidden_{i}')
            for i, units in enumerate(hidden_units)
        ]
        
        # Output layer (same dimension as input)
        self.output_layer = layers.Dense(
            input_dim,
            activation='linear',
            name='output'
        )
    
    def call(self, inputs):
        """
        Forward pass through the network.
        
        Args:
            inputs: Tuple of (x, mask)
                - x: Input features
                - mask: Binary mask for known features
        
        Returns:
            Predicted feature values
        """
        x, mask = inputs
        
        # Apply mask to input
        masked_x = x * mask
        
        # Pass through hidden layers
        for layer in self.hidden_layers:
            masked_x = layer(masked_x)
        
        # Generate output
        output = self.output_layer(masked_x)
        
        return output