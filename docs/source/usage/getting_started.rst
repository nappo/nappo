Getting started
===============

Running experiments with NAPPO takes 5 simple steps.

1. Define an environment
------------------------

First, we need to define a function that creates an environment for the agent to interact with. NAPPO already contains some basic environments, such as ``Pybullet``, ``Atari`` and ``MuJoCo``, so we can import one of those.

.. code-block:: python

    from nappo.envs import make_pybullet_train_env

However, here we have a slightly simplified version of the code we just imported.

.. code-block:: python

    import gym
    import pybullet_envs

    def make_pybullet_train_env(env_id, seed=0):
        env = gym.make(env_id)
        env.seed(seed)
        return env

As we can see, the function returns a ``gym.Env`` instance, so defining a version of this function for any other environment adapted to work with the OpenAI Gym interface is straightforward. Any ``gym.Wrappers`` can be added here as well.

2. Define a RL agent
--------------------

NAPPO subscribes the idea that composable agents are the best option to enable method experimentation. Individual components are easier to read, understand and modify. They also allow for method flexibility, as they can be combined in different ways.

NAPPO distinguished between 4 types of core components: the ``VecEnv``, allowing to stack multiple independent environments into a single one, the ``Algo``, which manages loss and gradient computation, the ``Actor``, implementing the deep neural networks used as function approximators, and the ``Storage``, which handles data storage, processing and retrieval. Selecting an instance of each class and combining them we can create an agent.

.. code-block:: python

    from nappo.core.vec_env import VecEnv

    # Define Environment Vector
    train_envs_factory, action_space, obs_space = VecEnv.create_factory(
        vec_env_size=1,
        log_dir="/tmp/train_example",
        env_fn=make_pybullet_train_env,
        env_kwargs={"env_id": "HalfCheetahBulletEnv-v0"})

.. note::
   The ``VecEnv`` class accepts an optional parameter called ``log_dir``. If provided, a ``gym.Monitor`` wrapper will be used to generate json log files for each individual environment in the vector.

.. code-block:: python

    from nappo.core.algos import PPO
    from nappo.core.storages import OnPolicyGAEBuffer
    from nappo.core.actors import OnPolicyActorCritic

    # Define RL Actor
    actor_factory = OnPolicyActorCritic.create_factory(obs_space, action_space)

    # Define RL training algorithm
    algo_factory = PPO.create_factory(
        lr=1e-4, num_epochs=4, clip_param=0.2, entropy_coef=0.01,
        value_loss_coef=.5, max_grad_norm=.5, num_mini_batch=4,
        use_clipped_value_loss=True, gamma=0.99)

    # Define rollouts storage
    storage_factory = OnPolicyGAEBuffer.create_factory(size=1000, gae_lambda=0.95)

.. note::
    Being able to scale to distributed regimes can require RL agent components to be instantiated multiple times in different processes. To do that, all NAPPO core components contain a specifically named class method, called ``create_factory``, which returns a function allowing to create component instances, a ``component factory``.

    Instead of directly defining a single RL agent instance, we can define a ``component factory`` for each component, and that will allow us to define later the training scheme with more flexibility.

3. Customize training scheme
----------------------------

Agent + Scheme = Trainable Agent

4. Train
--------

5. Check results
----------------