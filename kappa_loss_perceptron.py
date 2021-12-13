from dataclasses import dataclass
from jax import value_and_grad, jit, vmap
import jax.random
import jax.numpy as jnp
from jax.nn import softmax, tanh
from jax.nn.initializers import he_uniform
import numpy as np
import optax  # type: ignore
from typing import Callable, List
from sklearn.base import BaseEstimator  # type: ignore

# TODO: regularization?
# TODO: batch size params, currently 100%


@dataclass
class Shape:
    in_width: int
    out_width: int


@dataclass
class Layer:
    activation: Callable
    shape: Shape

    def dims(self):
        return (self.shape.in_width, self.shape.out_width)


# Need to subclass BaseEstimator for Yellowbrick to work
class KappaLossPerceptron(BaseEstimator):

    _estimator_type = "classifier"

    def __init__(
        self,
        num_classes: int,
        weight_matrix,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        early_stopping_min_improvement: float = 0.000001,
    ) -> None:
        self.num_classes = num_classes
        assert weight_matrix.shape[0] == weight_matrix.shape[1]
        assert weight_matrix.shape[0] == num_classes
        self.weight_matrix = weight_matrix
        # TODO: add warning if diagonal isn't all zeros
        self.rand_key = jax.random.PRNGKey(1)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.params: List = []
        self.loss_values: List[float] = []
        # If loss doesn't improve by at least this amount, stop training
        self.early_stopping_min_improvement = early_stopping_min_improvement

    def confusion_matrix_continuous(self, y_true, y_pred):
        """
        A confusion matrix that support continuous class probabilities, i.e.
        the output of a softmax layer.
        It also outputs a continuous valued confusion matrix. Since it's part
        of our loss function, this gives us a continous loss instead of a
        discrete one.
        """

        def f(i, j):
            """
            for each probability column j in y_true, sum the probabilities of
            of getting the expected answer for each expected answer i
            """
            return jnp.where(y_true == i, y_pred[:, j], 0).sum()

        vecs = vmap(vmap(f, in_axes=(0, None)), in_axes=(None, 0), out_axes=1)(
            jnp.arange(self.num_classes),
            jnp.arange(self.num_classes),
        )
        return vecs

    def kappa_continuous(self, y_true, y_pred):
        """
        A continuous version of Cohen's Kappa, given a y_pred is a vector of
        probabilities for each class
        """
        assert len(y_true) == len(y_pred)

        y_true = y_true.astype(int)

        observed = self.confusion_matrix_continuous(y_true, y_pred)
        num_scored_items = float(len(y_true))

        hist_true = jnp.bincount(
            y_true, minlength=self.num_classes, length=self.num_classes
        )
        hist_true = hist_true / num_scored_items
        preds = jnp.argmax(y_pred, axis=1)
        hist_pred = jnp.bincount(
            preds, minlength=self.num_classes, length=self.num_classes
        )
        hist_pred = hist_pred / num_scored_items
        expected = jnp.outer(hist_true, hist_pred)

        # Normalize observed array
        observed = observed / num_scored_items
        k = 1.0

        # If all weights are zero, that means no disagreements matter.
        if np.count_nonzero(self.weight_matrix):
            k -= sum(sum(self.weight_matrix * observed)) / sum(  # type: ignore
                sum(self.weight_matrix * expected)  # type: ignore
            )
        return k

    def init_layers(self, data_width: int):
        self.layers = [
            Layer(tanh, Shape(data_width, 5)),
            Layer(lambda x: softmax(x, axis=1), Shape(5, self.num_classes)),
        ]

    def init_params(self, data_width: int):
        self.init_layers(data_width)
        params = []
        he_generator = he_uniform()
        for layer in self.layers:
            params.append(
                dict(
                    weights=he_generator(self.rand_key, layer.dims()),
                    biases=jnp.zeros(layer.shape.out_width),
                )
            )
        self.params = params

    def forward_pass(self, X, W):
        for i, layer in enumerate(self.layers):
            layer_params = W[i]
            X = layer.activation(X @ layer_params["weights"] + layer_params["biases"])
        return X

    def weighted_kappa_loss(self, X, W, y, fn):
        y_hat = fn(X, W)
        # Positive kappa is good, so we want to minimize negative kappa
        # (maximize positive kappa)
        return -1 * self.kappa_continuous(y, y_hat)

    def loss(self, W, X, y):
        return self.weighted_kappa_loss(X, W, y, self.forward_pass)

    def fit(self, X, y, warm_start=False, max_iter=None, verbose=False):
        if y.min() != 0 or jnp.unique(jnp.diff(jnp.unique(y))) != jnp.array([1]):
            raise RuntimeError(
                "Class labels must be sequential integers starting at zero"
            )
        if not max_iter:
            max_iter = self.max_iter
        if not warm_start:
            self.loss_values = []
            self.init_params(X.shape[1])
        elif len(self.params) == 0:
            raise RuntimeError("Can't do warm start on untrained model.")
        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(self.params)  # type: ignore
        grad_func = jit(value_and_grad(self.loss))
        for _ in range(max_iter):
            loss_val, grads = grad_func(self.params, X, y)
            self.loss_values.append(loss_val)
            if verbose:
                print(loss_val)
            updates, opt_state = optimizer.update(grads, opt_state)
            self.params = optax.apply_updates(self.params, updates)  # type: ignore

            # early stopping
            improvement = sum(self.loss_values[-5:-1]) / 4 - loss_val
            if (
                len(self.loss_values) > 10
                and improvement < self.early_stopping_min_improvement
            ):
                print(
                    "Stopping early after {} iterations.".format(len(self.loss_values))
                )
                return

    def predict(self, X, one_hot=False):
        if one_hot:
            pred = self.forward_pass(X, self.params)
            z = jnp.zeros_like(pred)
            z = z.at[jnp.arange(len(pred)), pred.argmax(1)].set(1)
            return z
        else:
            return jnp.argmax(self.forward_pass(X, self.params), axis=1)

    def prediction_kappa(self, X, y_true):
        y_pred = self.predict(X, one_hot=True)
        return self.kappa_continuous(y_true, y_pred)
