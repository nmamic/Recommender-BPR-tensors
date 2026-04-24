# Recommender-BPR-tensors

My code for a university project on recommender systems using tensor decomposition. Implemented Bayesian Personal Ranking models based on [1]. 

Included are the models themselves, and a short script comparing them on the Delicious dataset retrieved from the HetRec 2011 workshop dataset. It can be found at https://grouplens.org/datasets/hetrec-2011/

The package with the models can be installed using pip install -e . in the root folder. The original project notebook, without separation into a library is also included in the root folder.

## References

[1] Rendle, S., & Schmidt-Thieme, L. (2010). *Pairwise Interaction Tensor Factorization
for Personalized Tag Recommendation*. WSDM '10, 81–90.
https://doi.org/10.1145/1718487.1718498

