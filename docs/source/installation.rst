Installation
============

Follow this steps for a successful installation in a conda environment::

    conda create -y -n nappo
    conda activate nappo
    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

    pip install nappo
    pip install git+git://github.com/openai/baselines.git
