import numpy as np

nn_architecture = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 6, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
]

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;

def init_layers(nn_architecture, seed = 99):
    np.random.seed(seed)
    param_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input = layer["input_dim"]
        layer_output = layer["output_dim"]

        param_values["W" + str(layer_idx)] = np.random.radn(layer_output, layer_input) * 0.1
        param_values["b" + str(layer_idx)] = np.random.radn(layer_output, 1) * 0.1

    return param_values;

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    Z_curr = np.dot(A_prev, W_curr) + b_curr

    if(activation == "relu"):
        activation_func = relu
    if(activation == "sigmoid"):
         activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')

    return activation_func(Z_curr), Z_curr;

def full_forward_propagation(X, param_values, nn_architecture):
    memory = {} # helps store the activations A & Z
    A_curr = X

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr

        activation = layer["activation"]
        W_curr = param_values["W" + str(layer_idx)]
        b_curr = param_values["b" + str(layer_idx)]
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activation)

        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr

        return A_curr, memory; # revisit, why are we returning A_curr?

def convert_prob_into_class(Y_hat):
    if Y_hat > 0.5:
        return True
    else:
        return False

def get_cost_value(Y_hat, Y):
    m = Y_hat.hat[1]
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()

def single_layer_backward_propagation(A_prev, Z_curr, dA_curr, W_curr b_curr, activation = "relu"):
    m = A_prev.shape[1]

    if activation == "relu":
        backward_activation_func = relu_backward
    if activation == "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception("Non-supportec activation function")
    
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m # revisit
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr;

def full_backward_propagation(nn_architecture, memory, param_values, Y_hat, Y):
    grad_values = {}
    m = Y_hat.shape[1]
    Y = Y.reshape(Y_hat.shape)
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

    for prev_layer_idx, layer in reversed(list(enumerate(nn_architecture))):
        activation = layer["activation"]
        layer_idx = idx + 1

        W_curr = param_values["W" + str(layer_idx)]
        b_curr = param_values["b" + str(layer_idx)]
        A_prev = memory["A" + str(prev_layer_idx)]
        Z_curr = memory["Z" + str(layer_idx)]
        dA_curr = dA_curr;

        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(A_prev, Z_curr, dA_curr, W_curr, b_curr, activation)
        grad_values["dW" + str(layer_idx)] = dW_curr
        grad_values["db" + str(layer_idx)] = db_curr

        return grad_values;

def update(nn_architecture, param_values, grad_values, learning_rate):
    for idx, layer in enumerate(nn_architecture):
        param_values["W" + str(idx)] -= learning_rate * grad_values["dW" + str(idx)]
        param_values["b" + str(idx)] -= learning_rate * grad_values["db" + str(idx)]
    return param_values;

def train(X, Y, nn_architecture, epochs, learning_rate):
    param_values = init_layers(nn_architecture, 2)
    cost_history = []
    accuracy_history = []

    for i in range(epochs):
        Y_hat, cache = full_forward_propagation(X, param_values, nn_architecture)
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)

        grad_values = full_backward_propagation(nn_architecture, cache, param_values, Y_hat, Y)
        param_values = update(nn_architecture, param_values, grad_values, learning_rate)

        return param_values, cost_history, accuracy_history