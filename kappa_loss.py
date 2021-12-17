from dataclasses import dataclass
from jax import value_and_grad, grad, jit, vmap, hessian
from jax._src.api import grad
from jax._src.flatten_util import ravel_pytree
import jax.random
import jax.numpy as jnp
from jax.nn import softmax, tanh
from jax.nn.initializers import he_uniform
import numpy as np  # type: ignore
import optax  # type: ignore
from typing import Callable, List, Optional, Union, Dict
from sklearn.base import BaseEstimator  # type: ignore
from lightgbm import LGBMClassifier  # type: ignore

# TODO: Add batching functionality, currently batch size == 100%


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


def confusion_matrix_continuous(y_true, y_pred, num_classes):
    """
    A confusion matrix that support continuous class probabilities, i.e.
    the output of a softmax layer.
    It also outputs a continuous valued confusion matrix, which makes for
    a smoother loss function.
    """

    def f(i, j):
        """
        for each probability column j in y_true, sum the probabilities of
        of getting the expected answer for each expected answer i
        """
        return jnp.where(y_true == i, y_pred[:, j], 0).sum()

    vecs = vmap(vmap(f, in_axes=(0, None)), in_axes=(None, 0), out_axes=1)(
        jnp.arange(num_classes),
        jnp.arange(num_classes),
    )
    return vecs


def kappa_continuous(y_true, y_pred, num_classes, weight_matrix):
    """
    A continuous version of Cohen's Kappa, given each row of `y_pred` is
    a vector of probabilities for each class
    """
    assert len(y_true) == len(y_pred)

    y_true = y_true.astype(int)

    observed = confusion_matrix_continuous(y_true, y_pred, num_classes)
    num_scored_items = float(len(y_true))

    hist_true = jnp.bincount(y_true, minlength=num_classes, length=num_classes)
    hist_true = hist_true / num_scored_items
    preds = jnp.argmax(y_pred, axis=1)
    hist_pred = jnp.bincount(preds, minlength=num_classes, length=num_classes)
    hist_pred = hist_pred / num_scored_items
    expected = jnp.outer(hist_true, hist_pred)

    # Normalize observed array
    observed = observed / num_scored_items
    k = 1.0
    k -= sum(sum(weight_matrix * observed)) / sum(  # type: ignore
        sum(weight_matrix * expected)  # type: ignore
    )
    return k


# Need to subclass BaseEstimator for Yellowbrick to work
class KappaLossNN(BaseEstimator):

    # Needed to get stratification from Sklearn data splitters
    _estimator_type = "classifier"

    def __init__(
        self,
        num_classes: int,
        weight_matrix,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        alpha: float = 0,
        early_stopping_min_improvement: float = 0.00001,
        hidden_layer_shapes=[5],
        hidden_layer_actvation=tanh,
    ) -> None:
        self.num_classes = num_classes
        assert weight_matrix.shape[0] == weight_matrix.shape[1]
        assert weight_matrix.shape[0] == num_classes
        # TODO: add warning if diagonal isn't all zeros
        self.weight_matrix = weight_matrix
        self.alpha = alpha
        self.rand_key = jax.random.PRNGKey(1)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.params: List = []
        self.loss_values: List[float] = []
        # each item is the number of neurons in a hidden layer
        self.hidden_layer_shapes: List[int] = hidden_layer_shapes
        self.hidden_layer_actvation = hidden_layer_actvation
        # If training loss doesn't improve by at least this amount, stop
        # TODO: Add true early stopping based on validation set
        self.early_stopping_min_improvement = early_stopping_min_improvement

    def init_layers(self, data_width: int):
        self.layers = []
        if len(self.hidden_layer_shapes):
            self.layers.append(
                Layer(
                    self.hidden_layer_actvation,
                    Shape(data_width, self.hidden_layer_shapes[0]),
                ),
            )
            for in_size, out_size in zip(
                self.hidden_layer_shapes[:-1], self.hidden_layer_shapes[1:]
            ):
                if in_size and out_size:
                    self.layers.append(
                        Layer(
                            self.hidden_layer_actvation,
                            Shape(in_size, out_size),
                        ),
                    )
            self.layers.append(
                Layer(
                    lambda x: softmax(x, axis=1),
                    Shape(self.hidden_layer_shapes[-1], self.num_classes),
                ),
            )
        else:
            # If no hidden layers, make a softmax perceptron
            self.layers.append(
                Layer(
                    lambda x: softmax(x, axis=1),
                    Shape(data_width, self.num_classes),
                ),
            )

    def init_params(self, data_width: int):
        self.init_layers(data_width)
        params = []
        he_generator = he_uniform()
        self.rand_key, *subkeys = jax.random.split(self.rand_key, len(self.layers) + 1)
        for layer, key in zip(self.layers, subkeys):
            params.append(
                dict(
                    weights=he_generator(key, layer.dims()),
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
        return -1 * kappa_continuous(y, y_hat, self.num_classes, self.weight_matrix)

    def regularization_penalty(self, W):
        flattened, _ = ravel_pytree(W)
        return self.alpha * jnp.sqrt(jnp.sum(jnp.power(flattened, 2)))

    def loss(self, W, X, y):
        model_loss = self.weighted_kappa_loss(X, W, y, self.forward_pass)
        if self.alpha > 0:
            return model_loss + self.regularization_penalty(W)
        else:
            return model_loss

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
                if verbose:
                    print(
                        "Stopping early after {} iterations.".format(
                            len(self.loss_values)
                        )
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
        return kappa_continuous(y_true, y_pred, self.num_classes, self.weight_matrix)


class KappaLGBM(LGBMClassifier):
    def __init__(
        self,
        # Wrap in a list as a hack to work with LGBM types
        # TODO: is there a better way? Currently makes many warnings:
        # `[LightGBM] [Warning] Unknown parameter: ]`
        weight_matrix,
        num_classes: int,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        class_weight: Optional[Union[Dict, str]] = None,
        min_split_gain: float = 0,
        min_child_weight: float = 0.001,
        min_child_samples: int = 20,
        subsample: float = 1,
        subsample_freq: int = 0,
        colsample_bytree: float = 1,
        reg_alpha: float = 0,
        reg_lambda: float = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        n_jobs: int = -1,
        silent: Union[bool, str] = "warn",
        importance_type: str = "split",
        **kwargs
    ):
        self.num_classes = num_classes
        objective = self.boosting_loss
        self.weight_matrix = weight_matrix
        super().__init__(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            objective=objective,
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            silent=silent,
            importance_type=importance_type,
            **kwargs
        )

    def get_weight_matrix(self):
        return np.array(self.weight_matrix)

    def stacked_kappa_loss(self, y_pred, y):
        """
        wrapped to switch argument order, make negative,
        and accept a column vector in LightGBM format where
        classes for all examples are stacked.
        """
        y_pred_shaped = y_pred.reshape(self.num_classes, -1).T
        return -1 * kappa_continuous(
            y, y_pred_shaped, self.num_classes, self.get_weight_matrix()
        )

    def boosting_loss(self, y, y_pred):
        grad_func = jit(grad(self.stacked_kappa_loss))
        grads = np.array(grad_func(y_pred, y))
        # Cohen's kappa is TODO: is it? linear in all variables, so hessian is
        # going to always be zero. Apparantly setting hessian to 1 (LGBM only
        # takes the diagonal) reverse to simple gradient descent
        # https://github.com/microsoft/LightGBM/issues/2128
        hess = np.ones(len(grads))
        return grads, hess

    def predict_proba(
        self,
        X,
        raw_score=False,
        start_iteration=0,
        num_iteration=None,
        pred_leaf=False,
        pred_contrib=False,
        **kwargs
    ):
        """
        need to wrap best class selector when using custom loss
        """
        return jnp.argmax(
            super().predict_proba(
                X,
                raw_score=raw_score,
                start_iteration=start_iteration,
                num_iteration=num_iteration,
                pred_leaf=pred_leaf,
                pred_contrib=pred_contrib,
                **kwargs
            ),
            axis=1,
        )
