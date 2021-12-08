from jax import value_and_grad, jit, vmap
from jax.nn import softmax
from jax.nn.initializers import glorot_uniform
import jax.random
import optax  # type: ignore
import jax.numpy as jnp
import numpy as np
from typing import List

# TODO: regularization?
# TODO: print()/verbose seems to wait until the end
# TODO: compile is slow, and also dependent on data size
# todo: double check "random" penalization of Cohen kappa
# todo: batch size params, currently 100%


class KappaLossPerceptron:
    def __init__(
        self,
        num_classes: int,
        weight_matrix,
        learning_rate: float = 0.01,
        epochs: int = 5,
    ) -> None:
        self.num_classes = num_classes
        assert weight_matrix.shape[0] == weight_matrix.shape[1]
        assert weight_matrix.shape[0] == num_classes
        self.weight_matrix = weight_matrix
        # TODO: add warning if diagonal isn't all zeros
        self.rand_key = jax.random.PRNGKey(1)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.params = None
        self.loss_values: List[float] = []

    def confusion_matrix_continuous(self, y_true, y_pred):
        """
        A confusion matrix that support continuous class probabilities, i.e.
        the output of a softmax layer.
        It also outputs a continuous valued confusion matrix. Since it's part
        of our loss function, this gives us a continous loss instead of a
        discrete one.
        """
        cm = jnp.zeros((self.num_classes, self.num_classes))
        for i in range(len(y_pred)):
            index_true = y_true[i]
            predicted_classes = y_pred[i]
            cm = cm.at[index_true, :].add(predicted_classes)
        return cm

    def kappa_continuous(self, y_true, y_pred):
        """
        A continuous version of Cohen's Kappa, given a y_pred is a vector of
        probabilities for each class
        """
        # Ensure that the lists are both the same length
        assert len(y_true) == len(y_pred)

        y_true = y_true.astype(int)

        # Build the observed/confusion matrix
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
        # If all weights are zero, that means no disagreements matter.
        k = 1.0
        if np.count_nonzero(self.weight_matrix):
            k -= sum(sum(self.weight_matrix * observed)) / sum(  # type: ignore
                sum(self.weight_matrix * expected)  # type: ignore
            )
        return k

    def softmax_layer(self, X, W):
        v = X @ W
        return softmax(v, axis=1)

    def weighted_kappa_loss(self, X, W, y, fn):
        y_hat = fn(X, W)
        return -1 * self.kappa_continuous(y, y_hat)

    def loss(self, W, X, y):
        return self.weighted_kappa_loss(X, W, y, self.softmax_layer)

    # TODO: I think there is an sklearn convention for "clean"
    def fit(self, X, y, clean=True, epochs=None, verbose=True):
        if not epochs:
            epochs = self.epochs
        if clean:
            self.loss_values = []
            param_gen = glorot_uniform()
            self.params = param_gen(self.rand_key, (X.shape[1], self.num_classes))
        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(self.params)  # type: ignore
        grad_func = jit(value_and_grad(self.loss))
        for _ in range(epochs):
            loss_val, grads = grad_func(self.params, X, y)
            if verbose:
                print(loss_val)
            self.loss_values.append(loss_val)
            updates, opt_state = optimizer.update(grads, opt_state)
            self.params = optax.apply_updates(self.params, updates)  # type: ignore
            # print(loss)

    def predict(self, X, one_hot=False):
        if one_hot:
            pred = self.softmax_layer(X, self.params)
            z = jnp.zeros_like(pred)
            z = z.at[jnp.arange(len(pred)), pred.argmax(1)].set(1)
            return z
        else:
            return jnp.argmax(self.softmax_layer(X, self.params), axis=1)

    def prediction_kappa(self, X, y_true):
        y_pred = self.predict(X, one_hot=True)
        return self.kappa_continuous(y_true, y_pred)
