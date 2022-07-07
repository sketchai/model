# deepmodel_attention

## Installation

1. Generate a conda env 
First, create and activate a basic conda env from the [env_gnn.yml](./env/env_gnn.yml) file. 

Run 
```
    conda env create -f ./env/env_gnn.yml
```

then 

```
    conda activate env_gnn
```


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

Run all the tests:

```
    poetry run pytest 
```

Run a specific test:

```
    poetry run pytest tests/model/test_messagepassing.py
```

See test coverage : [TO COMPLETE]

## Start a training and see results

Configure the path to your data in the [configuration file](config/gat.yml).

Start training: 

```
python commander.py
```

To see the metrics :

```
tensorboard --logdir data/gat_logs
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

Open the path 'http://localhost:6006/' in your browser.

## Reproduce results

Run the evaluation loop on the test dataset:

```
python testloop.py --path path/to/the/ckpt
```

## Notebooks

Play with your trained model in the [inference](notebook/inference.ipynb) notebook.

You will need to install [sam](https://github.com/sketchai/sam) and [ipywidgets](https://pypi.org/project/ipywidgets/) to visualize the results.

Use the following notebook to inspect the input data :
```
jupyter-notebook ./notebook/data_input.ipynb
```

## Docs

- [More about the numerical features generator](docs/models/numericalFeature.md): the numerical features generator allows to convert and harmonize information. In the specific context of sketches, it converts and harmonizes all the parameters characterizing the nodes (primitives) or the edges (constraints) of a sketch.

## Good pratices 

### PEP8

Use the pep8 norm to format all the code. Specific pep8 parameters are given into the [pyproject.toml](pyproject.toml) file.

```
autopep8 --in-place --aggressive --aggressive ./
```




