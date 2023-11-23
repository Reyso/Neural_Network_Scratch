from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class Neural:
    def __init__(self, layers: List[int], epochs: int, 
                 learning_rate: float = 0.001, batch_size: int=32,
                 validation_split: float = 0.2, verbose: int=1,
                 activation_function: str = 'relu'):
        self._layer_structure: List[int] = layers
        self._batch_size: int = batch_size
        self._epochs: int = epochs
        self._learning_rate: float = learning_rate
        self._validation_split: float = validation_split
        self._verbose: int = verbose
        self._losses: Dict[str, List[float]] = {"train": [], "validation": []}
        self._is_fit: bool = False
        self._layers = None
        self._activation_function = activation_function

        
    def initialize_weights(self, input_size: int, output_size: int) -> np.ndarray:
        # Initialize weights between -0.1 and 0.1
        weights = np.random.uniform(low=-0.1, high=0.1, size=(input_size, output_size))
        return weights

    def initialize_layers(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        layers = []
        for i in range(1, len(self._layer_structure)):
            weights = self.initialize_weights(self._layer_structure[i - 1], self._layer_structure[i])
            # biases = np.ones((1, self._layer_structure[i]))
            biases = np.random.uniform(low=0, high=1, size=(1, self._layer_structure[i]))


            layers.append((weights, biases))
        return layers
    
    
    def set_activation_function(self, activation_function: str) -> None:
        self._activation_function = activation_function

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Validation split
        X, X_val, y, y_val = train_test_split(X, y, test_size=self._validation_split, random_state=42)
        # Initialization of layers
        self._layers = self.initialize_layers()
        
        # Print initial parameters
        self.print_parameters()

        for epoch in range(self._epochs):
            epoch_losses = []

            for i in range(1, len(self._layers)):
                # Forward pass
                x_batch = X[i:(i+self._batch_size)]
                y_batch = y[i:(i+self._batch_size)]
                pred, hidden = self.forward(x_batch)

                # Calculate loss
                loss = self.calculate_loss(y_batch, pred)
                epoch_losses.append(np.mean(loss ** 2))

                # Backward
                self.backward(hidden, loss)

            # Validation
            valid_preds, _ = self.forward(X_val)
            train_loss = np.mean(epoch_losses)
            valid_loss = np.mean(self.calculate_mse(valid_preds, y_val))

            self._losses["train"].append(train_loss)
            self._losses["validation"].append(valid_loss)

            if self._verbose:
                print(f"Epoch: {epoch} Train MSE: {train_loss} Validation MSE: {valid_loss}")

        self._is_fit = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise Exception("Model has not been trained yet.")
        pred, _ = self.forward(X)
        return pred



    def forward(self, batch: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        hidden_layers = [batch.copy()]

        for i, layer in enumerate(self._layers):
            weights, biases = layer
            batch = np.matmul(batch, weights) + biases

            if i < len(self._layers) - 1:
                batch = self._apply_activation(batch)

            hidden_layers.append(batch.copy())

        return batch, hidden_layers

    def backward(self, hidden_layers: List[np.ndarray], grad: np.ndarray) -> None:
        for i in range(len(self._layers) - 1, -1, -1):
            weights, biases = self._layers[i]

            if i != len(self._layers) - 1:
                grad = np.multiply(grad, np.heaviside(hidden_layers[i + 1], 0))

            w_grad = hidden_layers[i].T @ grad
            b_grad = np.mean(grad, axis=0)

            self._layers[i] = (weights - w_grad * self._learning_rate, biases - b_grad * self._learning_rate)

            grad = grad @ weights.T

    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        if self._activation_function == 'relu':
            return np.maximum(x, 0)
        elif self._activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self._activation_function == 'tanh':
            return np.tanh(x)
        elif self._activation_function == 'linear':
            return x
        else:
            raise ValueError(f"Unsupported activation function: {self._activation_function}")

    def accuracy(self, y_true, y_pred):
        y_pred_binary = np.round(y_pred).astype(int)
        return accuracy_score(y_true, y_pred_binary)

    def print_parameters(self):
        for i, layer in enumerate(self._layers):
            weights, biases = layer
            print(f"Layer {i + 1} - Weights:\n{weights}\nBiases:\n{biases}\n")


    def plot_learning_curve(self) -> None:
        plt.plot(self._losses["train"], label="Training")
        plt.plot(self._losses["validation"], label="Validation")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.title("Learning Curve")
        plt.show()
        
        
    def calculate_loss(self, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        return predicted - actual

    def calculate_mse(self, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        return (actual - predicted) ** 2
