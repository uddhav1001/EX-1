
import pytest

from model_utils import train_model


@pytest.fixture(scope="module")
def trained_model():

    model, X_test, y_test = train_model()

    return model, X_test, y_test


def test_model_training(trained_model):

    model, X_test, y_test = trained_model

    assert model is not None


def test_prediction_count(trained_model):

    model, X_test, y_test = trained_model

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_prediction_classes(trained_model):

    model, X_test, y_test = trained_model

    predictions = model.predict(X_test)

    assert set(predictions).issubset({0, 1, 2})


def test_model_accuracy(trained_model):

    model, X_test, y_test = trained_model

    accuracy = model.score(X_test, y_test)

    assert accuracy >= 0.80
