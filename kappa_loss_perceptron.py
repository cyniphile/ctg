from jax import value_and_grad, jit, vmap
from jax.nn import softmax
from jax.nn.initializers import glorot_uniform
import jax.random
import optax  # type: ignore
import jax.numpy as jnp
import numpy as np
from typing import List

# TODO: regularization? early stopping
# TODO: print()/verbose seems to wait until the end
# TODO: compile is slow, and also dependent on data size
# todo: double check "random" penalization of Cohen kappa
# todo: batch size params, currently 100%


class KappaLossPerceptron:

    _estimator_type = "classifier"

    def __init__(
        self,
        num_classes: int,
        weight_matrix,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        early_stopping_min_improvement: float = 0.0001,
    ) -> None:
        self.num_classes = num_classes
        assert weight_matrix.shape[0] == weight_matrix.shape[1]
        assert weight_matrix.shape[0] == num_classes
        self.weight_matrix = weight_matrix
        # TODO: add warning if diagonal isn't all zeros
        self.rand_key = jax.random.PRNGKey(1)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.params = None
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

    def softmax_layer(self, X, W):
        v = X @ W
        return softmax(v, axis=1)

    def weighted_kappa_loss(self, X, W, y, fn):
        y_hat = fn(X, W)
        # Positive kappa is good, so we want to minimize negative kappa
        # (maximize positive kappa)
        return -1 * self.kappa_continuous(y, y_hat)

    def loss(self, W, X, y):
        return self.weighted_kappa_loss(X, W, y, self.softmax_layer)

    def fit(self, X, y, warm_start=False, max_iter=None, verbose=False):
        # TODO: add check that y follows sequential classes starting at zero
        if not max_iter:
            max_iter = self.max_iter
        if not warm_start:
            self.loss_values = []
            param_gen = glorot_uniform()
            self.params = param_gen(self.rand_key, (X.shape[1], self.num_classes))
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
            pred = self.softmax_layer(X, self.params)
            z = jnp.zeros_like(pred)
            z = z.at[jnp.arange(len(pred)), pred.argmax(1)].set(1)
            return z
        else:
            return jnp.argmax(self.softmax_layer(X, self.params), axis=1)

    def prediction_kappa(self, X, y_true):
        y_pred = self.predict(X, one_hot=True)
        return self.kappa_continuous(y_true, y_pred)
