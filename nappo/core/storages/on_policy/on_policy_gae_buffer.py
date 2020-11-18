import torch
from .on_policy_buffer import OnPolicyBuffer as B


class OnPolicyGAEBuffer(B):
    """
    Storage class for On-Policy algorithms with Generalized Advantage
    Estimator (GAE). https://arxiv.org/abs/1506.02438

    Parameters
    ----------
    size : int
        Storage capacity along time axis.
    gae_lambda : float
        GAE lambda parameter.
    device: torch.device
        CPU or specific GPU where data tensors will be placed and class
        computations will take place. Should be the same device where the
        actor critic model is located.

    Attributes
    ----------
    on_policy_data_fields : tuple
        Accepted data fields. If the samples inserted contain other fields,
        an AssertionError will be raised.
    max_size : int
        Storage capacity along time axis.
    device: torch.device
        CPU or specific GPU where data tensors will be placed and class
        computations will take place.
    gae_lambda : float
        GAE lambda parameter.
    """

    on_policy_data_fields = ("obs", "obs2", "rhs", "act", "rew", "val", "logp", "done")

    def __init__(self, size, gae_lambda=0.95, device=torch.device("cpu")):

        super(OnPolicyGAEBuffer, self).__init__(
            size=size,
            device=device)

        self.gae_lambda = gae_lambda

    @classmethod
    def create_factory(cls, size, gae_lambda=0.95):
        """
        Returns a function that creates OnPolicyGAEBuffer instances.

        Parameters
        ----------
        size : int
            Storage capacity along time axis.
        gae_lambda : float
            GAE lambda parameter.

        Returns
        -------
        create_buffer_instance : func
            creates a new OnPolicyBuffer class instance.
        """
        def create_buffer_instance(device):
            """Create and return a OnPolicyGAEBuffer instance."""
            return cls(size, gae_lambda, device)
        return create_buffer_instance

    def before_update(self, actor_critic, algo):
        """
        Before updating actor policy model, compute returns and advantages.

        Parameters
        ----------
        actor_critic : ActorCritic
            An actor_critic class instance.
        algo : an algorithm class
            An algorithm class instance.
        """
        with torch.no_grad():
            _ = actor_critic.get_action(
                self.data["obs"][self.step - 1],
                self.data["rhs"][self.step - 1],
                self.data["done"][self.step - 1])
            next_value = actor_critic.get_value(self.data["obs"][self.step - 1])

        self.data["val"][self.step] = next_value
        self.compute_returns(algo.gamma)
        self.compute_advantages()

    def compute_returns(self, gamma):
        """
        Compute return values.

        Parameters
        ----------
        gamma : float
            Algorithm discount factor parameter.
        """
        len = self.step if self.step != 0 else self.max_size
        gae = 0
        for step in reversed(range(len)):
            delta = (self.data["rew"][step] + gamma * self.data["val"][step + 1] * (
                1.0 - self.data["done"][step + 1]) - self.data["val"][step])
            gae = delta + gamma * self.gae_lambda * (1.0 - self.data["done"][step + 1]) * gae
            self.data["ret"][step] = gae + self.data["val"][step]
