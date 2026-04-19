import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Hebbian Network Class
class HebbianNetwork:
    def __init__(self, input_size, output_size, learning_rate=0.1):
        self.weights = np.random.rand(input_size, output_size)
        self.learning_rate = learning_rate
    
    def train(self, inputs, outputs):
        for i in range(len(inputs)):
            input_vector = inputs[i]
            output_vector = outputs[i]
            # Hebbian weight update rule
            self.weights += self.learning_rate * np.outer(input_vector, output_vector)
    
    def predict(self, input_vector):
        return np.dot(input_vector, self.weights)

# Training data
inputs = np.array([
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0]
])

outputs = np.array([
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0]
])

# Create network
network = HebbianNetwork(input_size=2, output_size=2)

# Train network
network.train(inputs, outputs)

# Test prediction
test_input = np.array([1, 0])
predicted_output = network.predict(test_input)
print("Predicted output for input", test_input, ":", predicted_output)

# Plot weight matrix heatmap
hebbian_weights = network.weights

plt.figure(figsize=(6,5))
sns.heatmap(hebbian_weights, annot=True, cmap="coolwarm", fmt=".3f", 
            xticklabels=["Output Neuron 1", "Output Neuron 2"], 
            yticklabels=["Input Neuron 1", "Input Neuron 2"])
plt.title("Hebbian Learning Weight Matrix")
plt.xlabel("Output Neurons")
plt.ylabel("Input Neurons")
plt.show()
