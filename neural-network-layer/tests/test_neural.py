
import numpy as np
import pytest

from neural_network import (
    sigmoid,
    relu,
    sigmoid_backward,
    relu_backward,
    init_layers,
    single_layer_forward_propagation,
    full_forward_propagation,
    get_cost_value,
    get_accuracy_value,
    single_layer_backward_propagation,
    full_backward_propagation,
    update,
    train,
)


ARCHITECTURE = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 1, "activation": "sigmoid"},
]


def test_sigmoid():
    np.testing.assert_allclose(
        sigmoid(np.array([0.0])),
        [0.5],
    )


def test_relu():
    np.testing.assert_array_equal(
        relu(np.array([-2.0, 0.0, 3.0])),
        [0.0, 0.0, 3.0],
    )


def test_sigmoid_backward():
    np.testing.assert_allclose(
        sigmoid_backward(np.array([1.0]), np.array([0.0])),
        [0.25],
    )


def test_relu_backward():
    np.testing.assert_array_equal(
        relu_backward(
            np.ones(3),
            np.array([-1.0, 0.0, 2.0]),
        ),
        [0.0, 0.0, 1.0],
    )


def test_init_layer_shapes():
    params = init_layers(ARCHITECTURE)

    assert params["W1"].shape == (4, 2)
    assert params["b1"].shape == (4, 1)
    assert params["W4"].shape == (1, 6)
    assert params["b4"].shape == (1, 1)


def test_initialization_is_reproducible():
    first = init_layers(ARCHITECTURE, seed=42)
    second = init_layers(ARCHITECTURE, seed=42)

    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_forward_pass_shapes():
    X = np.random.default_rng(1).normal(size=(2, 5))
    params = init_layers(ARCHITECTURE)

    Y_hat, memory = full_forward_propagation(
        X, params, ARCHITECTURE
    )

    assert Y_hat.shape == (1, 5)
    assert memory["Z1"].shape == (4, 5)
    assert memory["Z4"].shape == (1, 5)
    assert np.all((Y_hat >= 0) & (Y_hat <= 1))


def test_single_layer_forward():
    A_prev = np.array([[1.0, 2.0]])
    W = np.array([[2.0]])
    b = np.array([[1.0]])

    output, Z = single_layer_forward_propagation(
        A_prev, W, b, "relu"
    )

    np.testing.assert_array_equal(Z, [[3.0, 5.0]])
    np.testing.assert_array_equal(output, [[3.0, 5.0]])


def test_cost_and_accuracy():
    predictions = np.array([[0.9, 0.1, 0.8, 0.2]])
    labels = np.array([[1, 0, 1, 0]])

    assert get_cost_value(predictions, labels) >= 0
    assert get_accuracy_value(predictions, labels) == 1.0


def test_cost_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        get_cost_value(
            np.array([[0.5, 0.5]]),
            np.array([[1, 0, 1]]),
        )


def test_backward_pass_shapes():
    X = np.random.default_rng(2).normal(size=(2, 5))
    Y = np.array([[1, 0, 1, 0, 1]])
    params = init_layers(ARCHITECTURE)

    Y_hat, memory = full_forward_propagation(
        X, params, ARCHITECTURE
    )
    grads = full_backward_propagation(
        ARCHITECTURE, memory, params, Y_hat, Y
    )

    for idx, layer in enumerate(ARCHITECTURE, start=1):
        assert grads[f"dW{idx}"].shape == (
            layer["output_dim"], layer["input_dim"]
        )
        assert grads[f"db{idx}"].shape == (layer["output_dim"], 1)
        assert np.isfinite(grads[f"dW{idx}"]).all()
        assert np.isfinite(grads[f"db{idx}"]).all()


def test_single_layer_backward():
    A_prev = np.array([[1.0, 2.0]])
    W = np.array([[1.0]])
    b = np.zeros((1, 1))
    Z = W @ A_prev + b
    dA = np.ones_like(Z)

    dA_prev, dW, db = single_layer_backward_propagation(
        A_prev, Z, dA, W, b, "relu"
    )

    np.testing.assert_allclose(dW, [[1.5]])
    np.testing.assert_allclose(db, [[1.0]])
    np.testing.assert_allclose(dA_prev, [[1.0, 1.0]])


def test_update_changes_parameters():
    params = init_layers(ARCHITECTURE)
    grads = {
        f"d{kind}{idx}": np.ones_like(params[f"{kind}{idx}"])
        for idx in range(1, len(ARCHITECTURE) + 1)
        for kind in ("W", "b")
    }

    updated = update(ARCHITECTURE, params, grads, 0.1)

    for key in params:
        assert not np.array_equal(updated[key], params[key])
        np.testing.assert_allclose(
            updated[key], params[key] - 0.1
        )


def test_training_returns_finite_history():
    X = np.array([
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
    ])
    Y = np.array([[0, 1, 1, 0]])

    params, costs, accuracies = train(
        X,
        Y,
        ARCHITECTURE,
        epochs=10,
        learning_rate=0.01,
    )

    assert len(costs) == 10
    assert len(accuracies) == 10
    assert np.isfinite(costs).all()
    assert all(0 <= a <= 1 for a in accuracies)
    assert params["W4"].shape == (1, 6)


def test_training_rejects_bad_dimensions():
    X = np.ones((3, 4))
    Y = np.zeros((1, 4))

    with pytest.raises(ValueError):
        train(X, Y, ARCHITECTURE, epochs=2)
