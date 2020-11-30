Minimal example
===============

Imports
-------

.. code-block:: python
    
    import ray
    from nappo import Learner
    from nappo.schemes import Scheme
    from nappo.core.algos import PPO
    from nappo.core.envs import VecEnv
    from nappo.core.storages import OnPolicyGAEBuffer
    from nappo.core.actors import OnPolicyActorCritic, get_feature_extractor
    from nappo.envs import make_pybullet_train_env

Specifying ray resources
------------------------

.. code-block:: python

    # 0. init ray
    ray.init(address="auto")

Defining core components
------------------------

.. code-block:: python

    # 1. Define Train Vector of Envs
    train_envs_factory, action_space, obs_space = VecEnv.create_factory(
        vec_env_size=1, log_dir="/tmp/train_example", env_fn=make_pybullet_train_env,
        env_kwargs={"env_id": "HalfCheetahBulletEnv-v0"})
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

Defining training scheme
------------------------

.. code-block:: python

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

Executing training loop
-----------------------

.. code-block:: python

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
