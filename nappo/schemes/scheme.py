from .collection.c_worker_set import CWorkerSet
from .gradients.g_worker_set import GWorkerSet
from .updates.u_worker import UWorker


class Scheme:
    """
    Class to define training schemes and handle creation and operation
    of its workers.

    Parameters
    ----------
    algo_factory : func
        A function that creates an algorithm class.
    actor_factory : func
        A function that creates a policy.
    storage_factory : func
        A function that create a rollouts storage.
    train_envs_factory : func
        A function to create train environments.
    test_envs_factory : func
        A function to create test environments.
    """
    def __init__(self,

                 # core
                 algo_factory,
                 actor_factory,
                 storage_factory,
                 train_envs_factory,
                 test_envs_factory=lambda x, y, c: None,

                 # collection
                 col_remote_workers=0,
                 col_communication="synchronous",
                 col_worker_resources={"num_cpus": 1, "num_gpus": 0.5},
                 col_specs={"fraction_samples": 1.0, "fraction_workers": 1.0},

                 # gradients
                 grad_remote_workers=0,
                 grad_communication="synchronous",
                 grad_worker_resources={"num_cpus": 1, "num_gpus": 0.5},

                 # update
                 local_device=None,
                 update_execution="centralised"):

        # TODO. Add that core components check.

        col_execution="decentralised" if col_remote_workers > 0 else "centralised"
        grad_execution="decentralised" if grad_remote_workers > 0 else "centralised"

        col_workers_factory = CWorkerSet.create_factory(

            # core modules
            algo_factory=algo_factory,
            actor_factory=actor_factory,
            storage_factory=storage_factory,
            test_envs_factory=test_envs_factory,
            train_envs_factory=train_envs_factory,

            # col specs
            num_workers=col_remote_workers,
            col_worker_resources=col_worker_resources,
            fraction_samples=col_specs.get("fraction_samples"),
        )

        grad_workers_factory = GWorkerSet.create_factory(

            # col specs
            col_execution=col_execution,
            col_communication=col_communication,
            col_workers_factory=col_workers_factory,

            # grad_specs
            num_workers=grad_remote_workers,
            grad_worker_resources=grad_worker_resources
        )

        self._update_worker = UWorker(

            # grad specs
            grad_execution=grad_execution,
            grad_communication=grad_communication,
            grad_workers_factory=grad_workers_factory,
            fraction_workers=col_specs.get("fraction_workers"),

            # update specs
            local_device=local_device,
            update_execution=update_execution,
        )

    def update_worker(self):
        """Return local worker"""
        return self._update_worker