__data_scientist__ = "osomorog"
__date__ = "09/11/2022"
__iteration__ = 3


import os
import argparse
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from azureml.core.run import Run


# Split the dataframe into test and train data
def split_data(df):
    X = df.drop('Y', axis=1).values
    y = df['Y'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}}
    return data


# Train the model, return the model
def train_model(data, ridge_args):
    reg_model = Ridge(**ridge_args)
    reg_model.fit(data["train"]["X"], data["train"]["y"])
    return reg_model


# Evaluate the metrics for the model
def get_model_metrics(model, data):
    preds = model.predict(data["test"]["X"])
    mse = mean_squared_error(preds, data["test"]["y"])
    metrics = {"mse": mse}
    return metrics


def main():
    print("Running train.py")
    run = Run.get_context()

    # Define training parameters
    ridge_args = {"alpha": 0.5}

    sample_data = load_diabetes()
    train_df = pd.DataFrame(
        data=sample_data.data,
        columns=sample_data.feature_names)
    train_df['Y'] = sample_data.target

    data = split_data(train_df)

    # Train the model
    model = train_model(data, ridge_args)

    # Log the metrics for the model
    metrics = get_model_metrics(model, data)
    for (k, v) in metrics.items():
        print(f"{k}: {v}")
        run.log(k, v)
        # run.parent.log(k, v)

    os.makedirs(step_output_path, exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    output_path = os.path.join('outputs', 'diabetes_model.pkl')
    model_output_path = os.path.join(step_output_path, 'diabetes_model.pkl')
    joblib.dump(value=model, filename=output_path)
    joblib.dump(value=model, filename=model_output_path)

    run.complete()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("train")

    parser.add_argument(
        "--step_output",
        type=str,
        help=("output for passing data to next step")
    )

    args = parser.parse_args()
    step_output_path = args.step_output

    main()
