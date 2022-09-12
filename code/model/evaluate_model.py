from azureml.core import Run
import argparse
from azureml.core.model import Model

run = Run.get_context()

exp = run.experiment
ws = run.experiment.workspace

parser = argparse.ArgumentParser("evaluate")

parser.add_argument(
    "--model_name",
    type=str,
    help="Name of the Model",
    default="diabetes_model.pkl",
)

parser.add_argument(
    "--allow_run_cancel",
    type=str,
    help="Set this to false to avoid evaluation step from cancelling run after an unsuccessful evaluation",  # NOQA: E501
    default="true",
)

args = parser.parse_args()
run_id = run.parent.id
model_name = args.model_name
metric_eval = "mse"

allow_run_cancel = args.allow_run_cancel

firstRegistration = False
tag_name = 'experiment_name'

try:
    model = Model(ws, model_name)
except:
    model = None

if (model is not None):
    production_model_mse = 10000
    if (metric_eval in model.tags):
        production_model_mse = float(model.tags[metric_eval])
    try:
        new_model_mse = float(run.parent.get_metrics().get(metric_eval))
    except TypeError:
        new_model_mse = None
    if (production_model_mse is None or new_model_mse is None):
        print("Unable to find ", metric_eval, " metrics, "
              "exiting evaluation")
        if((allow_run_cancel).lower() == 'true'):
            run.parent.cancel()
    else:
        print(
            "Current Production model {}: {}, ".format(
                metric_eval, production_model_mse) +
            "New trained model {}: {}".format(
                metric_eval, new_model_mse
            )
        )

    if (new_model_mse < production_model_mse):
        print("New trained model performs better, "
              "thus it should be registered")
    else:
        print("New trained model metric is worse than or equal to "
              "production model so skipping model registration.")
        if((allow_run_cancel).lower() == 'true'):
            run.parent.cancel()
else:
    print("This is the first model, "
          "thus it should be registered")
