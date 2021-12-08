import jax.numpy as jnp
from numpy.testing import assert_almost_equal
from kappa_loss_perceptron import KappaLossPerceptron
from sklearn.metrics import confusion_matrix as sk_confusion_matrix  # type: ignore
from skll.metrics import kappa as skll_kappa  # type: ignore
import pytest


@pytest.fixture
def KLP(test_weights):
    return KappaLossPerceptron(num_classes=3, weight_matrix=test_weights, epochs=1)


@pytest.fixture
def test_weights():
    return jnp.array(
        # fmt: off
        [
# Predicted   N    S    P     # True 
            [0.0, 1.0, 1.0],  # N
            [1.0, 0.0, 1.0],  # S
            [1.0, 1.0, 0.0],  # P
        ]
        # fmt: on
    )


@pytest.fixture
def y_true():
    return jnp.array([0, 1, 0, 0, 1, 2, 2])
    # return jnp.array([0, 1, 0, 0, 1,])


@pytest.fixture
def X():
    return jnp.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ]
    )


@pytest.fixture
def y_pred():
    return jnp.array([1, 1, 0, 0, 1, 0, 0])


@pytest.fixture
def y_pred_one_hot():
    return jnp.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
        ]
    )


def test_confusion_matrix(KLP, y_true, y_pred, y_pred_one_hot):
    sk_answer = sk_confusion_matrix(y_true, y_pred)
    klp_answer = KLP.confusion_matrix_continuous(y_true, y_pred_one_hot)
    assert_almost_equal(klp_answer, sk_answer)


def test_kappa_continuous(KLP, y_true, y_pred, y_pred_one_hot):
    answer_KLP = KLP.kappa_continuous(y_true, y_pred_one_hot)
    answer_skll_kappa = skll_kappa(y_true, y_pred)
    assert_almost_equal(answer_KLP, answer_skll_kappa)
    y_pred_2 = jnp.array(
        [
            [0.1, 0.9, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
        ]
    )
    KLP.kappa_continuous(y_true, y_pred_2)


def test_fit_predict(KLP, y_true, X):
    """
    ensure score is getting better with more training
    """
    KLP.fit(X, y_true)
    kappa1 = KLP.discretized_kappa(X, y_true)
    KLP.fit(X, y_true, clean=False, epochs=55)
    kappa2 = KLP.discretized_kappa(X, y_true)
    # score improves
    assert kappa2 > kappa1
    # reached perfect score
    assert kappa2 == 1
