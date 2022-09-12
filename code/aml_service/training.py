import os

from azureml.core import Workspace, Environment, Experiment
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core.authentication import AzureCliAuthentication


def main():

    WORKSPACE_NAME = os.environ.get("WORKSPACE_NAME")
    SUBSCRIPTION_ID = os.environ.get("SUBSCRIPTION_ID")
    RESOURCE_GROUP = os.environ.get("RESOURCE_GROUP")
    ENV_NAME = "AzureML-sklearn-0.24-ubuntu18.04-py37-cpu"

    cli_auth = AzureCliAuthentication()

    ws = Workspace.get(
        name=WORKSPACE_NAME,
        subscription_id=SUBSCRIPTION_ID,
        resource_group=RESOURCE_GROUP,
        auth=cli_auth
    )

    compute_name = "aml-cluster"
    compute_target = ws.compute_targets[compute_name]

    gen_run_config = RunConfiguration()
    gen_run_config.target = compute_target

    try:
        environments = Environment.list(workspace=ws)
        for env in environments:
            if env == ENV_NAME:
                curated_environment = environments[ENV_NAME]

        gen_run_config.environment = curated_environment
    except Exception as e:
        print("Cannot get a curated environment, because:")
        print(e)

    pipeline_data = PipelineData(
        "pipeline_data", datastore=ws.get_default_datastore()
    )

    source_dir = '.'

    train_step = PythonScriptStep(
        name="Train Model",
        script_name='model/train.py',
        source_directory=source_dir,
        outputs=[pipeline_data],
        arguments=[
            "--step_output", pipeline_data
        ],
        compute_target=compute_target,
        runconfig=gen_run_config,
        allow_reuse=False
    )

    evaluate_step = PythonScriptStep(
        name="Evaluate Model",
        script_name='model/evaluate_model.py',
        compute_target=compute_target,
        runconfig=gen_run_config,
        allow_reuse=False
    )

    register_step = PythonScriptStep(
        name="Register Model",
        script_name='model/register_model.py',
        source_directory=source_dir,
        inputs=[pipeline_data],
        arguments=[
            "--step_input", pipeline_data
        ],
        compute_target=compute_target,
        runconfig=gen_run_config,
        allow_reuse=False
    )

    print("Include evaluation step before register step.")
    evaluate_step.run_after(train_step)
    register_step.run_after(evaluate_step)

    ppl = Pipeline(workspace=ws, steps=[
                   train_step, evaluate_step, register_step])
    run = Experiment(ws, 'diabetes_regression').submit(ppl)
    run.wait_for_completion()


if __name__ == "__main__":
    main()