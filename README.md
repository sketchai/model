# deepmodel_attention

In this package, we propose a ML model dedicated to constraint suggestions. It uses the [SAM preprocessing pipeline](https://github.com/sketchai/preprocessing) to prepare and filter sketches data. The predictions are in SAM format. As far SketchGraphs original format, the predictions can be converted using [the following package](https://github.com/sketchai/sketchgraph_vs_sam).

![CAD_ML_inf](https://user-images.githubusercontent.com/103726832/184371778-3a82f8f1-198b-40b5-81d2-d3b5f8c8fc5a.gif)

## Installation

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

Download the input data [here](https://huggingface.co/datasets/sketchai/sam-dataset)

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

Download pretrained models [here](https://huggingface.co/sketchai/sketch-gnn)

Run the evaluation loop on the test dataset:

```
python testloop.py --path path/to/the/ckpt
```

## Notebooks

Play with your trained model in the [inference](notebook/inference.ipynb) notebook.

You will need to install [sam](https://github.com/sketchai/sam) and [ipywidgets](https://pypi.org/project/ipywidgets/) to visualize the results.


## Docs

- [More about the numerical features generator](docs/models/numericalFeature.md): the numerical features generator allows to convert and harmonize information. In the specific context of sketches, it converts and harmonizes all the parameters characterizing the nodes (primitives) or the edges (constraints) of a sketch.

## Good pratices 

### PEP8

Use the pep8 norm to format all the code. Specific pep8 parameters are given into the [pyproject.toml](pyproject.toml) file.

```
autopep8 --in-place --aggressive --aggressive ./
```




