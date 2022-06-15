# Numerical Features

The [numerical_features](sketch_gnn/models/numerical_features) folder gathers a set of methods to convert constraint and type data of a CAD model into numerical vectors.

In particular, the classes [NumericalFeatureEncoding](sketch_gnn/models/numerical_features/encoding.py) and [NumericalFeaturesEmbedding](sketch_gnn/models/numerical_features/embedding.py) are directly extracted from the [numerical_features.py](https://gitlab.pam-retd.fr/cao_ml/sg/-/blob/master/sketchgraphs_models/graph/model/numerical_features.py) of SketchGraph model.



About the test: tests class can be found [here](tests/model/test_numericalFeatures.py).


## 1. Numerical features description

### 1.1 Numerical Feature encoding
[Class NumericalFeatureEncoding](sketch_gnn/models/numerical_features/encoding.py) 



### 1.2 Numerical Feature embedding
[Class NumericalFeaturesEmbedding](sketch_gnn/models/numerical_features/embedding.py)  is a ```torch.nn.Module``` that transform a sequence of vectors into a single one. This reduction of information is done by an average operation on each vector of the sequence.

About our case : 




### 1.3 Numerical Feature generator

The numerical features generator allows to convert and harmonize information. In the specific context of sketches, it converts and harmonizes all the parameters characterizing the nodes (primitives) or the edges (constraints) of a sketch.

__Description__  
Inputs:      
- d_features_dims (Dict) : a dict { elt : {component_1 : int, component_2 : int, ...}}
- embedding_dim (int)    : a integer representing the final size of the embedding vector

Outputs:  
- d_embedding (Dict) : a dict {elt.name : torch.nn.Module} where the torch.nn.Module encodes the feature into a vector of size embedding_dim

__How is it done?__  
The encoding and embedding methods are combined into a single function: the [numerical feature generator](sketch_gnn/models/numerical_features/generator.py).
Specifically, function ```generate_embedding(d_features_dims: Dict = {}, embedding_dim: int = None) -> Dict``` generates a dict of embedding layers whose keys are element names. 

__Examples?__  
Examples are proposed into the [tests](tests/model/test_numericalFeatures.py). 


Specifically, as far node (resp. edge) features, keys are string corresponding to the list of primitives (resp. constraints) chosen for the experiment. 
Dict values are a two-layers nn composed of first a NumericalFeatureEncoding layer then a NumericalFeaturesEmbedding layer.

__About the sketches specific case__  

- le dictionnaire doit être généré au moment du preprocessing (dans le fichier )

For encoding sketches information, d_features_dims is composed as follows:


