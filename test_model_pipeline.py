from model_pipeline import prepare_data


def test_prepare_data_returns_consistent_splits():
    x_train, x_test, y_train, y_test = prepare_data()
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_train) > 0
    assert len(x_test) > 0
