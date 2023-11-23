from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class Neural:
    def __init__(self, layers: List[int], epochs: int, 
                 learning_rate: float = 0.001, batch_size: int=32,
                 validation_split: float = 0.2, verbose: int=1,
                 activation_function: str = 'relu',random_state: int = None):
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
        self._random_state = random_state  # Adiciona o atributo random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        """
        Inicializa a classe Neural com os parâmetros necessários para a rede neural.

        Parâmetros:
        - layers: Lista contendo o número de neurônios em cada camada da rede.
        - epochs: Número de épocas de treinamento.
        - learning_rate: Taxa de aprendizado para a otimização.
        - batch_size: Tamanho do lote para treinamento em mini-lotes.
        - validation_split: Proporção dos dados a serem usados para validação.
        - verbose: Controla a exibição de informações durante o treinamento.
        - activation_function: Função de ativação a ser utilizada nas camadas ocultas.
        """
        
        
    def initialize_weights(self, input_size: int, output_size: int) -> np.ndarray:
        """
        Inicializa os pesos da camada entre -0.1 e 0.1.

        Parâmetros:
        - input_size: Número de neurônios na camada de entrada.
        - output_size: Número de neurônios na camada de saída.

        Retorna:
        - Matriz de pesos inicializados.
        """
        weights = np.random.uniform(low=-0.1, high=0.1, size=(input_size, output_size))
        return weights

    def initialize_layers(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Inicializa as camadas da rede neural com pesos e vieses aleatórios.

        Retorna:
        - Lista de tuplas, cada uma contendo os pesos e vieses de uma camada.
        """
        
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
        """
        Realiza o treinamento da rede neural.

        Parâmetros:
        - X: Dados de entrada.
        - y: Rótulos correspondentes.
        """
        # Divide os dados em conjuntos de treinamento e validação
        X, X_val, y, y_val = train_test_split(X, y, test_size=self._validation_split, random_state=42)

        # Inicializa as camadas
        self._layers = self.initialize_layers()

        # Exibe os parâmetros iniciais
        self.print_parameters()

        for epoch in range(self._epochs):
            epoch_losses = []

            for i in range(0, len(X), self._batch_size):
                # Seleciona um lote de dados
                x_batch = X[i:(i+self._batch_size)]
                y_batch = y[i:(i+self._batch_size)]

                # Passe para forward
                pred, hidden = self.forward(x_batch)

                # Calcula a perda
                loss = self.calculate_loss(y_batch, pred)
                epoch_losses.append(np.mean(loss ** 2))

                # Passe para o backpropagation
                self.backward(hidden, loss)

            # Validação
            valid_preds, _ = self.forward(X_val)
            train_loss = np.mean(epoch_losses)
            valid_loss = np.mean(self.calculate_mse(valid_preds, y_val))

            # Armazena as perdas
            self._losses["train"].append(train_loss)
            self._losses["validation"].append(valid_loss)

            # Exibe informações se verbose for verdadeiro
            if self._verbose:
                print(f"Epoch: {epoch} Train MSE: {train_loss} Validation MSE: {valid_loss}")

        # Define a flag de treinamento concluído
        self._is_fit = True
    
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza a predição usando a rede neural treinada.

        Parâmetros:
        - X: Dados de entrada.

        Retorna:
        - Saída prevista pela rede neural.
        """
        # Verifica se o modelo foi treinado
        if not self._is_fit:
            raise Exception("Model has not been trained yet.")
        # Realiza o passe para frente
        pred, _ = self.forward(X)
        return pred



    def forward(self, batch: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Realiza o passe para frente na rede neural.

        Parâmetros:
        - batch: Lote de dados de entrada.

        Retorna:
        - Saída da rede e lista de camadas ocultas.
        """
        # Lista para armazenar saídas de cada camada
        hidden_layers = [batch.copy()]

        for i, layer in enumerate(self._layers):
            weights, biases = layer
            # Calcula a saída da camada
            batch = np.matmul(batch, weights) + biases

            if i < len(self._layers) - 1:
                # Aplica a função de ativação
                batch = self._apply_activation(batch)

            # Armazena a saída da camada oculta
            hidden_layers.append(batch.copy())

        return batch, hidden_layers
    

    def backward(self, hidden_layers: List[np.ndarray], grad: np.ndarray) -> None:
        """
        Realiza o passe para trás para ajustar os pesos da rede neural.

        Parâmetros:
        - hidden_layers: Lista de camadas ocultas.
        - grad: Gradiente da última camada.
        """
        for i in range(len(self._layers) - 1, -1, -1):
            weights, biases = self._layers[i]

            if i != len(self._layers) - 1:
                # Aplica a função de ativação para o gradiente
                grad = np.multiply(grad, np.heaviside(hidden_layers[i + 1], 0))

            # Calcula os gradientes em relação aos pesos e vieses
            w_grad = hidden_layers[i].T @ grad
            b_grad = np.mean(grad, axis=0)

            # Atualiza os pesos e vieses usando o gradiente descendente
            self._layers[i] = (weights - w_grad * self._learning_rate, biases - b_grad * self._learning_rate)

            # Calcula o gradiente para a próxima camada
            grad = grad @ weights.T


    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        """
        Aplica a função de ativação apropriada à entrada.

        Parâmetros:
        - x: Entrada.

        Retorna:
        - Saída após a aplicação da função de ativação.
        """
        # Aplica a função de ativação especificada
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
        """
        Calcula a acurácia entre os rótulos verdadeiros e previstos.

        Parâmetros:
        - y_true: Rótulos verdadeiros.
        - y_pred: Rótulos previstos.

        Retorna:
        - Acurácia.
        """
        # Converte as previsões para rótulos binários e calcula a acurácia
        y_pred_binary = np.round(y_pred).astype(int)
        return accuracy_score(y_true, y_pred_binary)
    

    def print_parameters(self):
        """
        Imprime os pesos e vieses de cada camada da rede.
        """
        for i, layer in enumerate(self._layers):
            weights, biases = layer
            print(f"Layer {i + 1} - Weights:\n{weights}\nBiases:\n{biases}\n")


    def plot_learning_curve(self) -> None:
        """
        Plota a curva de aprendizado (MSE ao longo das épocas) para o conjunto de treinamento e validação.
        """
        # Plota as curvas de treinamento e validação
        plt.plot(self._losses["train"], label="Training")
        plt.plot(self._losses["validation"], label="Validation")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.title("Learning Curve")
        plt.show()


    def calculate_loss(self, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        """
        Calcula a perda como a diferença entre a previsão e o valor real.

        Parâmetros:
        - actual: Valores reais.
        - predicted: Valores previstos.

        Retorna:
        - Perda calculada.
        """
        # Calcula a diferença entre a previsão e o valor real
        return predicted - actual


    def calculate_mse(self, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        """
        Calcula o erro quadrático médio entre a previsão e o valor real.

        Parâmetros:
        - actual: Valores reais.
        - predicted: Valores previstos.

        Retorna:
        - Erro quadrático médio.
        """
        # Calcula o erro quadrático médio
        return (actual - predicted) ** 2