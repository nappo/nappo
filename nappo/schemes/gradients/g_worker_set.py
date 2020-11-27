from .g_worker import GWorker
from ..base.worker_set import WorkerSet as WS
from ..base.worker import default_remote_config


class GWorkerSet(WS):
    """Class to better handle the operations of ensembles of GWorkers."""

    def __init__(self,
                 col_specs,
                 num_workers,
                 local_device,
                 add_local_worker,
                 col_execution,
                 col_communication,
                 col_workers_factory,
                 grad_worker_resources,
                 initial_weights=None):

        self.worker_class = GWorker
        self.num_workers = num_workers
        default_remote_config.update(grad_worker_resources)
        self.remote_config = default_remote_config

        self.worker_params = {
            "col_specs": col_specs,
            "col_execution": col_execution,
            "col_communication": col_communication,
            "col_workers_factory": col_workers_factory,
        }

        super(GWorkerSet, self).__init__(
            worker=self.worker_class,
            local_device=local_device,
            num_workers=self.num_workers,
            initial_weights=initial_weights,
            worker_params=self.worker_params,
            add_local_worker=add_local_worker,
            worker_remote_config=self.remote_config)

    @classmethod
    def create_factory(cls,
                       num_workers,
                       col_workers_factory,
                       col_execution="distributed",
                       col_communication="synchronous",
                       grad_worker_resources=default_remote_config,
                       col_specs={"fraction_samples": 1.0, "fraction_workers": 1.0}):

        def grad_worker_set_factory(device, add_local_worker, initial_weights=None):
            return cls(
                col_specs=col_specs,
                local_device=device,
                num_workers=num_workers,
                initial_weights=initial_weights,
                add_local_worker=add_local_worker,
                col_execution=col_execution,
                col_communication=col_communication,
                col_workers_factory=col_workers_factory,
                grad_worker_resources=grad_worker_resources)

        return grad_worker_set_factory
