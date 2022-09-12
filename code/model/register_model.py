import json
import os
import sys
import argparse
import traceback
import joblib
from azureml.core import Run, Experiment, Workspace, Dataset
from azureml.core.model import Model as AMLModel


def main():

    run = Run.get_context()
    exp = run.experiment

    parser = argparse.ArgumentParser("register")

    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the Model",
        default="diabetes_model.pkl",
    )

    parser.add_argument(
        "--step_input",
        type=str,
        help=("input from previous steps")
    )

    args = parser.parse_args()
    run_id = run.parent.id
    model_name = args.model_name
    model_path = args.step_input

    print("Getting registration parameters")

    model_tags = {}
    for tag in ["mse"]:
        try:
            mtag = run.parent.get_metrics()[tag]
            model_tags[tag] = mtag
        except KeyError:
            print(f"Could not find {tag} metric on parent run.")

    # load the model
    print("Loading model from " + model_path)
    model_file = os.path.join(model_path, model_name)
    model = joblib.load(model_file)
    parent_tags = run.parent.get_tags()
    try:
        build_id = parent_tags["BuildId"]
    except KeyError:
        build_id = None
        print("BuildId tag not found on parent run.")
        print(f"Tags present: {parent_tags}")
    try:
        build_uri = parent_tags["BuildUri"]
    except KeyError:
        build_uri = None
        print("BuildUri tag not found on parent run.")
        print(f"Tags present: {parent_tags}")

    if (model is not None):
        if (build_id is None):
            register_aml_model(
                model_file,
                model_name,
                model_tags,
                exp,
                run_id)
        elif (build_uri is None):
            register_aml_model(
                model_file,
                model_name,
                model_tags,
                exp,
                run_id,
                build_id)
        else:
            register_aml_model(
                model_file,
                model_name,
                model_tags,
                exp,
                run_id,
                build_id,
                build_uri)
    else:
        print("Model not found. Skipping model registration.")
        sys.exit(0)


def model_already_registered(model_name, exp, run_id):
    model_list = AMLModel.list(exp.workspace, name=model_name, run_id=run_id)
    if len(model_list) >= 1:
        e = ("Model name:", model_name, "in workspace",
             exp.workspace, "with run_id ", run_id, "is already registered.")
        print(e)
        raise Exception(e)
    else:
        print("Model is not registered for this run.")


def register_aml_model(
    model_path,
    model_name,
    model_tags,
    exp,
    run_id,
    build_id: str = 'none',
    build_uri=None
):
    try:
        tagsValue = {"area": "diabetes_regression",
                     "run_id": run_id,
                     "experiment_name": exp.name}
        tagsValue.update(model_tags)
        if (build_id != 'none'):
            model_already_registered(model_name, exp, run_id)
            tagsValue["BuildId"] = build_id
            if (build_uri is not None):
                tagsValue["BuildUri"] = build_uri

        model = AMLModel.register(
            workspace=exp.workspace,
            model_name=model_name,
            model_path=model_path,
            tags=tagsValue,)
        os.chdir("..")
        print(
            "Model registered: {} \nModel Description: {} "
            "\nModel Version: {}".format(
                model.name, model.description, model.version
            )
        )
    except Exception:
        traceback.print_exc(limit=None, file=None, chain=True)
        print("Model registration failed")
        raise


if __name__ == '__main__':
    main()
