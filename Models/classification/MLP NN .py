import numpy as np 

def sigmoid(x):
	return 1/(1+np.exp(-x))
	
def sigmoid_derivative(x):
	return x * (1 - x)
	
x = np.array([[0,0],
			 [0,1],
			 [1,0],
			 [1,1]])
			 
y = np.array([[0], 
			 [1],
			 [1],
			 [0]])

# Network architecture
input_size = 2
hidden_size = 4
output_size = 1

# Weights initialization
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Training loop
learning_rate = 0.1
for epoch in range(10000):
    # Forward pass
    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Loss (Binary Cross-Entropy can be used, here we use MSE for simplicity)
    loss = np.mean((y - a2) ** 2)

    # Backpropagation
    error = y - a2
    d_output = error * sigmoid_derivative(a2)

    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(a1)

    # Update weights and biases
    W2 += a1.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    W1 += x.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} - Loss: {loss:.4f}")

# Final Output
print("\nTrained Output:")
print(np.round(a2, 2))
