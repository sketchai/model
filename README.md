# deepmodel_attention


1. Generate a conda env 
First, create and activate a basic conda env from the [env_deep.yml](./env/env_deep.yml) file. 

Run 
```
    conda env create -f ./env/env_deep.yml
```

then 

```
    conda activate basic_conda
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


## Good pratices 

### PEP8

Use the pep8 norm to format all the code. Specific pep8 parameters are given into the [pyproject.toml](pyproject.toml) file.

```
autopep8 --in-place --aggressive --aggressive ./
```


### FLAKE8