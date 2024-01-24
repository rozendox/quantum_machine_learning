from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
import numpy as np

# Quantum activation function simulation
def quantum_activation(circuit, q_input, theta):
    circuit.ry(theta, q_input)
    return circuit

# Quantum perceptron-like model
def quantum_perceptron(theta, input_data):
    # Create a quantum circuit
    circuit = QuantumCircuit(1, 1)

    # Apply quantum activation function
    quantum_activation(circuit, 0, theta)

    # Measure the quantum state
    circuit.measure(0, 0)

    # Simulate the quantum circuit
    backend = Aer.get_backend('qasm_simulator')
    t_circuit = transpile(circuit, backend)
    qobj = assemble(t_circuit)
    result = backend.run(qobj).result()

    # Extract measurement result
    counts = result.get_counts()
    probability_0 = counts.get('0', 0) / 1024

    return probability_0

# Binary classification loss function
def binary_classification_loss(predictions, labels):
    return -np.mean(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))

# Training data and labels
training_data = np.array([0.1, 0.2, 0.3, 0.4])
labels = np.array([0, 0, 1, 1])

# Initial parameter
theta_initial = 0.1

# Learning rate
learning_rate = 0.01

# Number of iterations
num_iterations = 1000

# Training the quantum perceptron-like model
theta = theta_initial
for iteration in range(num_iterations):
    # Calculate the quantum perceptron output for training data
    predictions = [quantum_perceptron(theta, data) for data in training_data]

    # Calculate the loss
    loss = binary_classification_loss(predictions, labels)

    # Calculate the gradient of the loss with respect to the parameter theta
    gradient = np.mean([(prediction - label) * data for prediction, label, data in zip(predictions, labels, training_data)])

    # Update the parameter using gradient descent
    theta -= learning_rate * gradient

    # Display progress every 100 iterations
    if iteration % 100 == 0:
        print(f"Iteration {iteration}: Loss = {loss}")

# Display the final parameter
print("Final parameter of the quantum perceptron:", theta)
