# This folder contains result on the dataset provided by Linzen'18 

* Testing data folder contains the processed linzen data (that is not in form of strings!) that can be directly used as the basic command line args. 
* Nomenclature need to be followed for the result and .pkl files : model_prefix_<Hidden_dim>_<Embedding_dim>
* Further details will be present in the individual files
* Moreover, we want to keep the Linzen testing away from the normal decay rnn model. Although this can be used in training, but refrain from using this for training. 
* If changes are made in the source code of the decay rnn or lstm (GPU CPU whatever, then please do the same changes in this folder file's too!)

