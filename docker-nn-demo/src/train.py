import numpy as np
import yaml
from pathlib import Path

# Load config
config = yaml.safe_load(Path("config.yaml").read_text())

hidden_units = config["model"]["hidden_units"]
lr = config["model"]["learning_rate"]
epochs = config["model"]["epochs"]
seed = config["model"]["seed"]

rng = np.random.default_rng(seed)

# XOR dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]], dtype=np.float32)

y = np.array([[0],
              [1],
              [1],
              [0]], dtype=np.float32)

# 2 -> hidden -> 1 network
W1 = rng.normal(0, 1, size=(2, hidden_units)).astype(np.float32)
b1 = np.zeros((1, hidden_units), dtype=np.float32)
W2 = rng.normal(0, 1, size=(hidden_units, 1)).astype(np.float32)
b2 = np.zeros((1, 1), dtype=np.float32)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    return a * (1 - a)

for epoch in range(epochs):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    yhat = sigmoid(z2)

    loss = np.mean((yhat - y) ** 2)

    d_yhat = 2 * (yhat - y) / y.shape[0]
    d_z2 = d_yhat * sigmoid_deriv(yhat)
    dW2 = a1.T @ d_z2
    db2 = np.sum(d_z2, axis=0, keepdims=True)

    d_a1 = d_z2 @ W2.T
    d_z1 = d_a1 * sigmoid_deriv(a1)
    dW1 = X.T @ d_z1
    db1 = np.sum(d_z1, axis=0, keepdims=True)

    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 1000 == 0:
        print(f"epoch={epoch} loss={loss:.6f}")

print("\nFinal predictions:")
pred = (yhat > 0.5).astype(int)
for i in range(len(X)):
    print(f"{X[i].astype(int).tolist()} -> {int(pred[i,0])}")
