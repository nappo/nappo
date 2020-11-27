from .scheme import Scheme
from .base.worker import default_remote_config


def get_scheme(
        scheme_name,
        algo_factory,
        actor_factory,
        storage_factory,
        train_envs_factory,
        test_envs_factory,
        col_workers=0,
        grad_workers=0,
        local_device=None,
        col_worker_resources=default_remote_config,
        grad_worker_resources=default_remote_config,
        col_specs={"fraction_samples": 1.0, "fraction_workers": 1.0},
):

    """ _ """
    params = {}

    # general params
    params.update({
        "algo_factory":algo_factory,
        "actor_factory":actor_factory,
        "storage_factory": storage_factory,
        "train_envs_factory":train_envs_factory,
        "test_envs_factory":test_envs_factory,
        "local_device": local_device,
        "col_specs": col_specs,
        "col_worker_resources": col_worker_resources,
        "grad_worker_resources": grad_worker_resources,
    })

    # scheme specific params
    if scheme_name == "cscs":
        params.update({
            "col_workers": 0,
            "grad_workers": 0,
            "col_communication": "synchronous",
            "grad_communication": "synchronous",
            "update_execution": "centralised",
        })
    elif scheme_name == "csca":
        params.update({
            "col_workers": 0,
            "grad_workers": 0,
            "col_communication": "synchronous",
            "grad_communication": "asynchronous",
            "update_execution": "centralised",
        })
    elif scheme_name == "csda":
        params.update({
            "col_workers": 0,
            "grad_workers": grad_workers,
            "col_communication": "synchronous",
            "grad_communication": "asynchronous",
            "update_execution": "centralised",
        })
    elif scheme_name == "dacs":
        params.update({
            "col_workers": col_workers,
            "grad_workers": 0,
            "col_communication": "asynchronous",
            "grad_communication": "synchronous",
            "update_execution": "centralised",
        })
    elif scheme_name == "dads":
        params.update({
            "col_workers": col_workers,
            "grad_workers": grad_workers,
            "col_communication": "asynchronous",
            "grad_communication": "synchronous",
            "update_execution": "centralised",
        })
    elif scheme_name == "dada":
        params.update({
            "col_workers": col_workers,
            "grad_workers": grad_workers,
            "col_communication": "asynchronous",
            "grad_communication": "asynchronous",
            "update_execution": "centralised",
        })

    return Scheme(**params)
