## Nappo: A PyTorch Library for distributed Reinforcement Learning

Nappo is a pytorch-based library for RL that focuses on distributed implementations, yet flexible enough to allow for method experimentation.

### Installation
```

    conda create -y -n nappo
    conda activate nappo
    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

    pip install nappo
    pip install git+git://github.com/openai/baselines.git

```

### Documentation

NAPPO documentation can be found [here](http://nappo.readthedocs.io/).

### Minimal code example


```
import ray
from nappo import Learner
from nappo.schemes import Scheme
from nappo.core.algos import PPO
from nappo.core.envs import VecEnv
from nappo.core.storages import OnPolicyGAEBuffer
from nappo.core.actors import OnPolicyActorCritic, get_feature_extractor
from nappo.envs import make_pybullet_train_env

# 0. init ray
ray.init(address="auto")

```

The first part in any Nappo training script consists in defining the core components, the lower level modules. All core components have a `create_factory` method, which returns a function that allows to later create independent instances in different workers if required by the training scheme.

We can start with **the VecEnv** (vector environment). Nappo supports by default pybullet, atari and mujoco environments, but it is easy to extend it to any other environment. A detailed explanation about how to do it can be found [here](http://nappo.readthedocs.io/).

```
# 1. Define Train Vector of Envs
train_envs_factory, action_space, obs_space = VecEnv.create_factory(
    vec_env_size=1, log_dir="/tmp/train_example", env_fn=make_pybullet_train_env,
    env_kwargs={"env_id": "HalfCheetahBulletEnv-v0"})
```

We can continue by defining an on-policy or off-policy set of **Actor** (or ActorCritic), **Algo** and **Storage** core components.

```
# 2. Define RL Actor
actor_factory = OnPolicyActorCritic.create_factory(
    obs_space, action_space, feature_extractor_network=get_feature_extractor("MLP"))

# 3. Define RL training algorithm
algo_factory = PPO.create_factory(
    lr=1e-4, num_epochs=4, clip_param=0.2, entropy_coef=0.01,
    value_loss_coef=.5, max_grad_norm=.5, num_mini_batch=4,
    use_clipped_value_loss=True, gamma=0.99)

# 4. Define rollouts storage
storage_factory = OnPolicyGAEBuffer.create_factory(size=1000, gae_lambda=0.95)
```

One of the main ideas behind Nappo is to allow single components to be replaced for experimentation without the need to change anything else. Since in RL not all components are compatible with each other (e.g. an on-policy storage with an off-policy algorithm is not expected to work), some libraries advocate or higher level implementations with a single function call that accept many parameters and handle creation of components under the hood. This approach might be generally more suitable to generate benchmarks and to use out-of-the-box solutions in industry, but less so for researchers trying to improve the state-of-the-art by switching and changing components. Furthermore, to a certain extend some components can be reused in a different set. If the components within the defined set do not match, a NotImplementedError error will be raised during execution.

We encourage users to create their own core components to extend current functionality, following the base.py templates associated with each one of them. Neural networks used as function approximators in the actor components can also be modified by the user. A more detailed explanation about how to do it can be found [here](http://nappo.readthedocs.io/).

Following, we instantiate the training scheme of our choice.

Worker components were designed to work for any combination of core components.

```
# 5. Define workers

# Core components params
scheme_parameters = {
    "algo_factory": algo_factory,
    "actor_factory": actor_factory,
    "storage_factory": storage_factory,
    "train_envs_factory": train_envs_factory}

# Collection operation params
scheme_parameters.update({
    "col_remote_workers": 0, # only local workers
    "col_communication": "synchronous"})

# Gradient computation operation params
scheme_parameters.update({
    "grad_remote_workers": 0, # only local workers
    "col_communication": "synchronous"})

# Update operation params
scheme_parameters.update({
    "update_execution": "centralised"})

scheme = Scheme(**scheme_parameters)
```

Finally, we create a Learner class instance and define the training loop.

```
# 6. Define learner
learner = Learner(scheme, target_steps=1000000, log_dir="/tmp/train_example")

# 7. Define train loop
iterations = 0
while not learner.done():
    learner.step()
    if iterations % 1 == 0:
        learner.print_info()
    if iterations % 100 == 0:
        save_name = learner.save_model()
    iterations += 1
```

### Available core components

* Core components
    * envs: VecEnv
    * algos:
        * On-policy: PPO
        * Off-policy: SAC
    * actors:
        * On-policy: OnPolicyActorCritic
        * Off-policy: OffPolicyActorCritic
    * storages:
        * On-policy: OnPolicyBuffer, OnPolicyGAEBuffer, OnPolicyVTraceBuffer
        * Off-policy: ReplayBuffer: HindsightExperienceReplayBuffer

### Scheme options

The following images shows how nappo schemes are structured

![alt text](https://github.com/nappo/nappo/blob/master/images/nappo_overview.jpg)

* Distributed schemes summary
    * Data collection operations can be
        * centralised (1 local workers)
        * decentralised (M remote workers), which can coordinate
            * synchronous
            * asynchronous
    * Gradient computation operations can be
        * centralised (1 local workers)
        * decentralised (N remote workers), which can coordinate
            * synchronous
            * asynchronous
    * Model update operations can occur
        * centralised (in 1 local workers with a central network version)
        * decentralised (in the N remote workers)

A more detailed explanation of the training scheme possibilities can be found [here](http://nappo.readthedocs.io/).

The parameters we used to create our Scheme instance in the training example above correspond to the simplest non-distributed scheme.

![alt text](https://github.com/nappo/nappo/blob/master/images/nappo_single_threaded.jpg)

### Current limitations


### Citing Nappo

```
@misc{nappo2020rl,
  author = {Bou, Albert},
  title = {Nappo: A PyTorch Library for distributed Reinforcement Learning},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/nappo/nappo}},
}
```