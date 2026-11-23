import argparse
import subprocess
import sys

from prefect import flow, task, get_run_logger

from model_pipeline import (
    prepare_data,
    train_model,
    save_model,
    load_model,
    evaluate_model,
)

# Les etapes qualite/securite ne ciblent que ces 3 fichiers (pas le venv)
PROJECT_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def run_cmd(cmd, check=True):
    """Execute une commande et envoie sa sortie dans les logs Prefect."""
    logger = get_run_logger()
    logger.info("Commande : %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Echec ({result.returncode}) : {' '.join(cmd)}")
    return result.returncode


# ------------------------------------------------------------ TASKS (data/modele)
@task(name="install_dependencies")
def install_dependencies():
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="prepare_data")
def prepare_data_task():
    return prepare_data()


@task(name="train_model")
def train_model_task(x_train, y_train):
    return train_model(x_train, y_train)


@task(name="save_model")
def save_model_task(model):
    save_model(model)


@task(name="load_model")
def load_model_task():
    return load_model()


@task(name="evaluate_model")
def evaluate_model_task(model, x_test, y_test):
    return evaluate_model(model, x_test, y_test)


# ------------------------------------------------------------ TASKS (suivi du code)
@task(name="format_code")
def format_code():
    run_cmd([sys.executable, "-m", "black", *PROJECT_FILES])


@task(name="quality_code")
def quality_code():
    # rapport affiche dans les logs, sans bloquer le pipeline
    run_cmd(
        [
            sys.executable,
            "-m",
            "flake8",
            "--max-line-length=88",
            "--extend-ignore=E203,W503",
            *PROJECT_FILES,
        ],
        check=False,
    )


@task(name="security_code")
def security_code():
    # rapport affiche dans les logs, sans bloquer le pipeline
    run_cmd([sys.executable, "-m", "bandit", "-q", *PROJECT_FILES], check=False)


@task(name="unit_tests")
def unit_tests():
    # ce test bloque le pipeline s'il echoue
    run_cmd([sys.executable, "-m", "pytest", "-v", "test_model_pipeline.py"])


# ------------------------------------------------------------ FLOWS
@flow(name="code")
def code_flow():
    install_dependencies()
    format_code()
    quality_code()
    security_code()
    unit_tests()


@flow(name="all")
def all_flow():
    # etapes de code, puis etapes 2-3-4-6
    install_dependencies()
    format_code()
    quality_code()
    security_code()
    unit_tests()
    x_train, x_test, y_train, y_test = prepare_data_task()
    model = train_model_task(x_train, y_train)
    save_model_task(model)
    evaluate_model_task(model, x_test, y_test)


@flow(name="train")
def train_flow():
    x_train, x_test, y_train, y_test = prepare_data_task()
    model = train_model_task(x_train, y_train)
    save_model_task(model)


@flow(name="evaluate")
def evaluate_flow():
    _, x_test, _, y_test = prepare_data_task()
    model = load_model_task()
    evaluate_model_task(model, x_test, y_test)


FLOWS = {
    "all": all_flow,
    "code": code_flow,
    "train": train_flow,
    "entrainement": train_flow,
    "evaluate": evaluate_flow,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", choices=FLOWS.keys(), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
