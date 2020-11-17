import os
import ray
import time
from functools import partial
from collections import defaultdict, deque
from torch.utils.tensorboard import SummaryWriter

class Learner:
    """
    Task learner class.

    Class to manage the training process. It pushes forward training calling
    the update workers and tracks progress.

    Parameters
    ----------
    update_workers : Worker or WorkerSet
        The set of workers handling actor_critic updates.
    target_steps : int
        Number of environment steps to reach to complete training.
    log_dir : str
        Target directory for Tensorboard logs.

    Attributes
    ----------
    update_workers : Worker or WorkerSet
        The set of workers handling actor_critic updates.
    target_steps : int
        Number of environment steps to reach to complete training.
    log_dir : str
        Target directory for Tensorboard logs.
    num_samples_collected : int
        Number of train environment transition samples so far in the
        training process.
    metrics : collections.defaultdict
        Collection of deque's, tracking recent training metrics (as moving
        averages).
    start : float
        Training start time.
    """

    def __init__(self, workers, target_steps, log_dir=None):

        # Input attributes
        self.log_dir = log_dir
        self.update_workers = workers.updater_workers()
        self.target_steps = target_steps

        # Counters and metrics
        self.num_samples_collected = 0
        self.metrics = defaultdict(partial(deque, maxlen=100))

        # Define summary writer
        if log_dir:
            tb_log_dir = "{}/tensorboard_logs".format(log_dir)
            os.makedirs(tb_log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_log_dir)
        else: self.writer = None

        # Record starting time
        self.start = time.time()

    def step(self):
        """Takes a logical synchronous optimization step."""

        # Update step
        info = self.update_workers.step()
        info["scheme/metrics/fps"] = int(self.num_samples_collected / (time.time() - self.start))

        # Update counters
        self.num_samples_collected += info.pop("collected_samples")

        # Update and log metrics
        for k, v in info.items():
            self.metrics[k].append(v)
            if self.writer: self.writer.add_scalar(k, v, self.num_samples_collected)

    def done(self):
        """
        Return True if training has finished (target_steps reached).

        Returns
        -------
        flag : bool
            True if training has reached the target number of steps.
        """
        flag = self.num_samples_collected >= self.target_steps
        if flag: self.update_workers.stop()
        return flag

    def get_metrics(self):
        """Returns metrics averaged across number of updates."""
        return {k: sum(v) / len(v) for k, v in self.metrics.items()}

    def print_info(self):
        """Print relevant information about the training process"""
        s = "Update {}, num samples collected {}, FPS {}".format(
            self.update_workers.num_updates, self.num_samples_collected,
            int(self.num_samples_collected / (time.time() - self.start)))
        s += "\n"
        for k, v in self.get_metrics().items():
            if k.split("/")[0] == "algo":
                s += "{} {}, ".format(k.split("/")[-1], v)
        print(s[:-2], flush=True)

    def update_algo_parameter(self, parameter_name, new_parameter_value):
        """
        If `parameter_name` is an attribute of the algorithm used for training,
        change its value to `new_parameter_value value`.

        Parameters
        ----------
        parameter_name : str
            Worker.algo attribute name
        new_parameter_value : int or float
            New value for `parameter_name`.
        """
        self.update_workers.update_algo_parameter(parameter_name, new_parameter_value)

    def save_model(self):
        """
        Save currently learned actor_critic version.

        Returns
        -------
        save_name : str
            Path to saved file.
        """
        fname = os.path.join(self.log_dir, "actor_critic.state_dict")
        save_name = self.update_workers.save_model(fname)
        return save_name
