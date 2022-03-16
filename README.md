# deepmodel_attention


1. Generate a conda env 
First, create and activate a basic conda env from the [env_gat.yml](./env/env_gat.yml) file. 

Run 
```
    conda env create -f ./env/env_gat.yml
```

then 

```
    conda activate env_gat
```

NB: it can be good to change the conda name env into [env_basic_conda.yml](./env/env_basic_conda.yml) file.


2. Install poetry and package dependencies

To install package dependencies with poetry, 

```
    poetry install
```

To update package dependencies, 
```
    poetry update
```

## Testing 

For running all the tests:

```
    poetry run pytest 
```

For running a specific test: [TO COMPLETE]


See test coverage : [TO COMPLETE]

## Start a training and see results

To start a training : 

```
python commander.py
```

To see metrics :

```
tensorboard --logdir your_path_logs
```
The following line will appear

```
TensorFlow installation not found - running with reduced feature set.

NOTE: Using experimental fast data loading logic. To disable, pass
    "--load_fast=false" and report issues on GitHub. More details:
    https://github.com/tensorflow/tensorboard/issues/4784

Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.8.0 at http://localhost:6006/ (Press CTRL+C to quit)

```

Open the path 'http://localhost:6006/' in Firefox.

## Docs

- [More about the numerical features generator](docs/models/numericalFeature.md): the numerical features generator allows to convert and harmonize information. In the specific context of sketches, it converts and harmonizes all the parameters characterizing the nodes (primitives) or the edges (constraints) of a sketch.

## Good pratices 

### PEP8

Use the pep8 norm to format all the code. Specific pep8 parameters are given into the [pyproject.toml](pyproject.toml) file.

```
autopep8 --in-place --aggressive --aggressive ./
```




