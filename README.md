## Nappo: A PyTorch library for distributed Reinforcement Learning

Deep Reinforcement learning (DRL) has been very successful in recent years but current methods still require vast amounts of data to solve non-trivial environments.  Scaling to solve more complex tasks requires frameworks that are flexible enough to allow prototyping and testing of new ideas, yet avoiding the impractically slow experimental turnaround times associated to single-threaded implementations.  NAPPO is a pytorch-based library for DRL that allows to easily assemble RL agents using a set of core reusable and easily extendable sub-modules as building blocks.  To reduce training times, NAPPO allows scaling agents with a parameterizable component called Scheme, that permits to define distributed architectures with great flexibility by specifying which operations should be decoupled, which should be parallelized, and how parallel tasks should be synchronized.

### Installation

```
    conda create -y -n nappo
    conda activate nappo

    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
    pip install git+git://github.com/openai/baselines.git

    pip install nappo
```

### Documentation

NAPPO documentation can be found [here](http://nappo.readthedocs.io/).

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
