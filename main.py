import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    predict,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Prepare the data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--predict", action="store_true", help="Predict on test data")
    parser.add_argument("--save", action="store_true", help="Train and save the model")
    parser.add_argument("--load", action="store_true", help="Load the model")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    args = parser.parse_args()

    # --all active toutes les étapes
    if args.all:
        args.prepare = True
        args.train = True
        args.save = True
        args.evaluate = True
        args.predict = True
        args.load = True

    x_train = x_test = y_train = y_test = None
    model = None

    if args.prepare:
        x_train, x_test, y_train, y_test = prepare_data()
        print("Data prepared.")

    if args.train or args.save:
        if x_train is None:
            x_train, x_test, y_train, y_test = prepare_data()
        model = train_model(x_train, y_train)
        save_model(model)
        print("Model trained and saved.")

    if args.load:
        model = load_model()
        print("Model loaded.")

    if args.evaluate:
        if x_test is None:
            x_train, x_test, y_train, y_test = prepare_data()
        if model is None:
            model = load_model()
        evaluate_model(model, x_test, y_test)

    if args.predict:
        if x_test is None:
            x_train, x_test, y_train, y_test = prepare_data()
        print(predict(x_test[:10]))


if __name__ == "__main__":
    main()
