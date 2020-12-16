from typing import Union

import mlflow
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, f1_score, precision_score)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MultiLabelBinarizer


def eval_metrics(y_test: Union[list, np.array], y_preds: Union[list, np.array]) ->  dict:
    """
    Get the necessary evaluation metrics on an unseen evaluation test set
    :param y_test: the binarized (0/1) labels of the test set
    :param y_preds: the binarized (0/1) labels predicted by the model
    :return: Dictionary with accuracy, f1 score, avg_precision and precision
    """
    accuracy = accuracy_score(y_test, y_preds)
    f1 = f1_score(y_test, y_preds, average="weighted")
    avg_precision = average_precision_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds, average="weighted")
    return {"accuracy": accuracy,
            "f1": f1,
            "avg_precision": avg_precision,
            "precision": precision}

# Adapted from https://gist.github.com/liorshk/9dfcb4a8e744fc15650cbd4c2b0955e5
def log_run(gridsearch: Union[GridSearchCV, RandomizedSearchCV], python_model: mlflow.pyfunc.PythonModel,
            binarizer: MultiLabelBinarizer, y_test_binarized: Union[list, np.array],
            experiment_name: str, model_name: str, run_index: int, conda_env: dict,
            tags: dict={}) -> None:
        """
        Logs cross validation results of a hyperparameter optimization to mlflow tracking server

        :param gridsearch: Any hyperparameter optimization object like GridSearch, RandomizedSearch, as long as it implements
        a `.fit()` method and has a `cv_results`_ attribute. BayesSeach from sklopt could also work, but it hasn't been tested
        :param binarizer: a MultiLabelBinarizer to binarize the Y lablels, which will be stored in conjuction with the model,
        and will be used to produce a classification report.
        :param y_test_binarized: binarized y-labels for the evaluation set (holdout, not used in cross-validation)
        :param experiment_name (str): experiment name
        :param model_name (str): Name of the model
        :param run_index (int): Index of the run (in Gridsearch)
        :param conda_env (dict): A dictionary that describes the conda environment (MLFlow Format)
        :param tags (dict): Dictionary of extra data and tags (usually features)
        """

        cv_results = gridsearch.cv_results_
        with mlflow.start_run(run_name=str(run_index)) as run:

            mlflow.log_param("folds", gridsearch.cv)

            print("Logging parameters")
            params = cv_results.get("params").keys()
            for param in params:
                mlflow.log_param(param, cv_results["param_%s" % param][run_index])

            print("Logging metrics")
            for score_name in [score for score in cv_results if "mean_test" in score]:
                mlflow.log_metric(score_name, cv_results[score_name][run_index])
                mlflow.log_metric(score_name.replace("mean","std"), cv_results[score_name.replace("mean","std")][run_index])

            y_test_pred_binarized = gridsearch.best_estimator_.predict(X_test)
            class_report = classification_report(
                                        y_test_binarized,
                                        y_test_pred_binarized,
                                        target_names=binarizer.classes_,
                                        zero_division=1
                                    )
            # Write results to file
            class_report_path = "metrics/classification_report.txt"
            cv_results_path = "metrics/{}-cv_results.csv".format(model_name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pd.DataFrame(cv_results).to_csv(filename, index=False)
                with open(class_report_path, 'w') as f:
                    f.write(class_report)

            ## LOG MODEL AND BINARIZER
            print("Logging model")
            pipeline_path = "models/classifier.pkl"
            binarizer_path = "models/binarizer.pkl"

            with open(pipeline_path, 'wb') as f:
                dill.dump(gridsearch.best_estimator_, f)
            with open(binarizer_path, 'wb') as f:
                dill.dump(binarizer, f)

            artifacts = {
                "pipeline": pipeline_path,
                "binarizer": binarizer_path,
                "classification_report": class_report_path,
                "cv_results": cv_results_path
            }

            mlflow_pyfunc_model_path = "so_pyfunc_model"
            mlflow.pyfunc.log_model(
                artifact_path=mlflow_pyfunc_model_path,
                python_model=python_model,
                artifacts=artifacts,
                conda_env=conda_env,
            )


            print("Logging extra data related to the experiment")
            mlflow.set_tags(tags)

            run_id = run.info.run_uuid
            experiment_id = run.info.experiment_id
            print(mlflow.get_artifact_uri())
            print("runID: %s" % run_id)
            mlflow.end_run()


def log_results(gridsearch: Union[GridSearchCV, RandomizedSearchCV], python_model: mlflow.pyfunc.PythonModel,
                binarizer: MultiLabelBinarizer, y_test_binarized: Union[list, np.array],
                experiment_name: str, model_name: str, tags:dict = {},
                conda_env: dict, log_only_best: bool=False) -> None:
    """Logging of cross validation results to mlflow tracking server

    Args:
    :param gridsearch: Any hyperparameter optimization object like GridSearch, RandomizedSearch, as long as it implements
    a `.fit()` method and has a `cv_results`_ attribute. BayesSeach from sklopt could also work, but it hasn't been tested
    :param binarizer: a MultiLabelBinarizer to binarize the Y lablels, which will be stored in conjuction with the model,
    and will be used to produce a classification report.
    :param y_test_binarized: binarized y-labels for the evaluation set (holdout, not used in cross-validation)
    :param experiment_name (str): experiment name
    :param model_name (str): Name of the model
    :param conda_env (dict): A dictionary that describes the conda environment (MLFlow Format)
    :param log_only_best (bool): Whether to log only the best model in the gridsearch or all the other models as well
    """

    best = gridsearch.best_index_

    if log_only_best:
        log_run(gridsearch, python_model, binarizer, y_test_binarized,
                experiment_name, model_name, best, conda_env, tags)
    else:
        for i in range(len(gridsearch.cv_results_['params'])):
            log_run(gridsearch, experiment_name, model_name, i, conda_env, tags)


def get_default_conda_env():
    """
    Extract current conda environment for the minimum packages we want to register.
    """
    from pkg_resources import get_distribution
    return {
            'name': 'so-classifier-env',
            'channels': ['defaults', "anaconda", "conda-firge"],
            'dependencies': [
                f"python={mlflow.pyfunc.PYTHON_VERSION}",
                f"scikit-learn={get_distribution('scikit-learn').version}",
                f"pip={get_distribution('pip').version}",
                f"setuptools={get_distribution('setuptools').version}",
                {'pip':
                 [f"boto3=={get_distribution('boto3').version}",
                  f"dill=={get_distribution('dill').version}",
                  f"mlflow={get_distribution('mlflow').version}",
                  f"nltk[stopwords]=={get_distribution('nltk').version}",
                  f"nltk[punkt]=={get_distribution('nltk').version}",
                  f"numpy=={get_distribution('numpy').version}",
                  f"pandas=={get_distribution('pandas').version}",
                  f"scikit-optimize=={get_distribution('scikit-optimize').version}",
                  f"scipy=={get_distribution('scipy').version}"]
                }
            ]
        }
