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

2. Follow the recommended installation for [pytorch](https://pytorch.org/) then [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) and [pytorch-lightning](https://www.pytorchlightning.ai/)


3. Install additional package dependencies via poetry

```
    poetry install
```

## Testing 

Run all the tests:

```
    pytest 
```

Run a specific test:

```
    pytest tests/model/test_messagepassing.py
```

See test coverage : [TO COMPLETE]

## Start a training and see results

Configure the path to your data in the [configuration file](config/gat.yml).

Data is available at [huggingface dev](https://huggingface.co/datasets/sketchai/sam-dataset/tree/dev)

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
python evaluationloop.py --path path/to/the/ckpt --conf path/to/the/hparams.yml
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




