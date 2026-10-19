
import numpy as np


def sigmoid(Z):
    # Clip values to avoid overflow in exp.
    Z = np.clip(Z, -500, 500)
    return 1.0 / (1.0 + np.exp(-Z))


def relu(Z):
    return np.maximum(0, Z)


def sigmoid_backward(dA, Z):
    A = sigmoid(Z)
    return dA * A * (1.0 - A)


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def init_layers(nn_architecture, seed=99):
    rng = np.random.default_rng(seed)
    param_values = {}

    for idx, layer in enumerate(nn_architecture, start=1):
        layer_input = layer["input_dim"]
        layer_output = layer["output_dim"]

        # He initialization for ReLU layers; Xavier for sigmoid.
        if layer["activation"] == "relu":
            scale = np.sqrt(2.0 / layer_input)
        elif layer["activation"] == "sigmoid":
            scale = np.sqrt(1.0 / layer_input)
        else:
            raise ValueError(
                f"Unsupported activation: {layer['activation']}"
            )

        param_values[f"W{idx}"] = (
            rng.standard_normal((layer_output, layer_input)) * scale
        )
        param_values[f"b{idx}"] = np.zeros((layer_output, 1))

    return param_values


def single_layer_forward_propagation(
    A_prev, W_curr, b_curr, activation="relu"
):
    Z_curr = W_curr @ A_prev + b_curr

    if activation == "relu":
        A_curr = relu(Z_curr)
    elif activation == "sigmoid":
        A_curr = sigmoid(Z_curr)
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    return A_curr, Z_curr


def full_forward_propagation(X, param_values, nn_architecture):
    memory = {}
    A_curr = X

    for idx, layer in enumerate(nn_architecture, start=1):
        A_prev = A_curr

        W_curr = param_values[f"W{idx}"]
        b_curr = param_values[f"b{idx}"]

        A_curr, Z_curr = single_layer_forward_propagation(
            A_prev,
            W_curr,
            b_curr,
            layer["activation"],
        )

        memory[f"A{idx - 1}"] = A_prev
        memory[f"Z{idx}"] = Z_curr

    return A_curr, memory


def convert_prob_into_class(Y_hat):
    return (Y_hat > 0.5).astype(int)


def get_cost_value(Y_hat, Y):
    if Y_hat.shape != Y.shape:
        raise ValueError(
            f"Shape mismatch: predictions {Y_hat.shape}, labels {Y.shape}"
        )

    eps = 1e-12
    Y_hat = np.clip(Y_hat, eps, 1.0 - eps)
    m = Y.shape[1]

    cost = -np.sum(
        Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)
    ) / m

    return float(np.squeeze(cost))


def get_accuracy_value(Y_hat, Y):
    predictions = convert_prob_into_class(Y_hat)
    return float(np.mean(predictions == Y))


def single_layer_backward_propagation(
    A_prev, Z_curr, dA_curr, W_curr, b_curr, activation="relu"
):
    del b_curr  # Bias is not needed to calculate these gradients.

    m = A_prev.shape[1]

    if activation == "relu":
        dZ_curr = relu_backward(dA_curr, Z_curr)
    elif activation == "sigmoid":
        dZ_curr = sigmoid_backward(dA_curr, Z_curr)
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    dW_curr = (dZ_curr @ A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = W_curr.T @ dZ_curr

    return dA_prev, dW_curr, db_curr


def full_backward_propagation(
    nn_architecture, memory, param_values, Y_hat, Y
):
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    if Y.shape != Y_hat.shape:
        raise ValueError(
            f"Shape mismatch: predictions {Y_hat.shape}, labels {Y.shape}"
        )

    grad_values = {}
    Y_hat = np.clip(Y_hat, 1e-12, 1.0 - 1e-12)

    # For sigmoid output + binary cross-entropy, dZ = Y_hat - Y.
    dA_curr = -(Y / Y_hat) + (1.0 - Y) / (1.0 - Y_hat)
    dA_curr *= Y_hat * (1.0 - Y_hat)

    for layer_idx in reversed(range(1, len(nn_architecture) + 1)):
        layer = nn_architecture[layer_idx - 1]

        W_curr = param_values[f"W{layer_idx}"]
        b_curr = param_values[f"b{layer_idx}"]
        A_prev = memory[f"A{layer_idx - 1}"]
        Z_curr = memory[f"Z{layer_idx}"]

        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            A_prev,
            Z_curr,
            dA_curr,
            W_curr,
            b_curr,
            layer["activation"],
        )

        grad_values[f"dW{layer_idx}"] = dW_curr
        grad_values[f"db{layer_idx}"] = db_curr

        dA_curr = dA_prev

    return grad_values


def update(nn_architecture, param_values, grad_values, learning_rate):
    updated_params = {}

    for layer_idx in range(1, len(nn_architecture) + 1):
        updated_params[f"W{layer_idx}"] = (
            param_values[f"W{layer_idx}"]
            - learning_rate * grad_values[f"dW{layer_idx}"]
        )
        updated_params[f"b{layer_idx}"] = (
            param_values[f"b{layer_idx}"]
            - learning_rate * grad_values[f"db{layer_idx}"]
        )

    return updated_params


def train(
    X,
    Y,
    nn_architecture,
    epochs=1000,
    learning_rate=0.01,
    seed=99,
):
    if X.ndim != 2:
        raise ValueError("X must have shape (input_features, examples).")

    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    if Y.shape != (1, X.shape[1]):
        raise ValueError("Y must have shape (1, number_of_examples).")

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    expected_input_dim = X.shape[0]

    for layer in nn_architecture:
        if layer["input_dim"] != expected_input_dim:
            raise ValueError(
                f"Expected input_dim={expected_input_dim}, "
                f"got {layer['input_dim']}."
            )
        expected_input_dim = layer["output_dim"]

    if nn_architecture[-1]["activation"] != "sigmoid":
        raise ValueError(
            "Binary classification requires a sigmoid output layer."
        )

    param_values = init_layers(nn_architecture, seed)
    cost_history = []
    accuracy_history = []

    for _ in range(epochs):
        Y_hat, memory = full_forward_propagation(
            X, param_values, nn_architecture
        )

        cost_history.append(get_cost_value(Y_hat, Y))
        accuracy_history.append(get_accuracy_value(Y_hat, Y))

        gradients = full_backward_propagation(
            nn_architecture, memory, param_values, Y_hat, Y
        )

        param_values = update(
            nn_architecture, param_values, gradients, learning_rate
        )

    return param_values, cost_history, accuracy_history
