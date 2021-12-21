import jax.numpy as jnp
import numpy as np  # type: ignore
from sklearn.model_selection import cross_validate  # type: ignore
from sklearn.metrics import make_scorer  # type: ignore
from numpy.testing import assert_almost_equal  # type: ignore
from kappa_loss import (
    KappaLossNN,
    KappaLossLGBM,
    confusion_matrix_continuous,
    kappa_continuous,
)
from sklearn.metrics import confusion_matrix as sk_confusion_matrix  # type: ignore
from skll.metrics import kappa as skll_kappa  # type: ignore
import pytest


@pytest.fixture
def KNeuralNet(test_weights):
    return KappaLossNN(
        num_classes=3, weight_matrix=test_weights, max_iter=1, alpha=0.001
    )


@pytest.fixture
def KLGBM(test_weights):
    return KappaLossLGBM(
        weight_matrix=list(test_weights), num_classes=3, validation_size=0.1
    )


@pytest.fixture
def test_weights():
    return jnp.array(
        # fmt: off
        [
# Predicted   N    S    P     # True 
            [0.0, 1.0, 1.0],  # N
            [1.0, 0.0, 1.0],  # S
            [1.1, 1.0, 0.0],  # P
        ]
        # fmt: on
    )


@pytest.fixture
def y_true():
    return jnp.array([0, 1, 0, 0, 1, 2, 2])


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


def test_confusion_matrix(y_true, y_pred, y_pred_one_hot):
    sk_answer = sk_confusion_matrix(y_true, y_pred)
    klp_answer = confusion_matrix_continuous(y_true, y_pred_one_hot, num_classes=3)
    assert_almost_equal(klp_answer, sk_answer)


def test_kappa_continuous(y_true, y_pred, y_pred_one_hot, test_weights):
    answer_KLP = kappa_continuous(
        y_true, y_pred_one_hot, weight_matrix=test_weights, num_classes=3
    )
    answer_skll_kappa = skll_kappa(y_true, y_pred, weights=test_weights)
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
    # works with non-discrete predictions
    kappa_continuous(y_true, y_pred_2, num_classes=3, weight_matrix=test_weights)


def test_stacked_kappa(KLGBM, y_true, y_pred_one_hot, test_weights):
    answer_KLP = kappa_continuous(
        y_true, y_pred_one_hot, num_classes=3, weight_matrix=test_weights
    )
    y_pred_one_hot_stacked = y_pred_one_hot.T.ravel()
    answer_stacked = -1 * KLGBM.stacked_kappa_loss(y_true, y_pred_one_hot_stacked)
    assert_almost_equal(answer_KLP, answer_stacked)


def test_fit_predict_nn(KNeuralNet, y_true, X):
    KNeuralNet.fit(X, y_true, verbose=False)
    kappa1 = KNeuralNet.prediction_kappa(X, y_true)
    KNeuralNet.fit(X, y_true, warm_start=True, max_iter=25, verbose=True)
    kappa2 = KNeuralNet.prediction_kappa(X, y_true)
    # score improves
    assert kappa2 > kappa1
    # reached perfect score
    assert kappa2 == 1


def test_lgbm_kappa(KLGBM, y_true, X):
    # augment data size, otherwise get errors from LGBM
    MULTIPLIER = 11
    new_X = jnp.concatenate([X for _ in range(MULTIPLIER)])
    new_y = jnp.concatenate([y_true for _ in range(MULTIPLIER)])
    KLGBM.fit(new_X, new_y)
    preds = KLGBM.predict(X)
    assert skll_kappa(preds, y_true) == 1


def test_multi_fit(test_weights, y_true, X, KLGBM):

    new_X = jnp.concatenate([X for _ in range(20)])
    new_y = jnp.concatenate([y_true for _ in range(20)])

    def weightedKappa(x, y):
        return skll_kappa(x, y, weights=test_weights)

    kappaScorer = make_scorer(weightedKappa)

    score = cross_validate(
        KLGBM,
        X=new_X,
        y=new_y,
        cv=2,
        scoring=kappaScorer,
        n_jobs=-1,
        verbose=10,
    )
    return score
