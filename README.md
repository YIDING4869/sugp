# Sugp Experiment Runner
This repository contains the implementation of the Sugp experiment runner, which is designed to process and run experiments on various datasets. It supports both CSV and Jackson file formats for the datasets. The datasets are stored in the sugp/data directory.

## How to Run
To run the experiments, find and execute the sugp/runner/run.py script.

## Datasets

### ImageNet
The ImageNet dataset may occasionally produce running warnings due to the algorithm implementation. These warnings occur when the upper and lower bounds in the calculations are equal to zero or infinitely close to zero.
Successfully reproduced

### Nightstreet
The Nightstreet dataset still has some issues regarding the label definitions. When attempting to print the labels, they appear to be undefined. This problem needs to be addressed before running experiments on this dataset.

### Beta, Ontonotes , Tacred
Successfully reproduced


## References
Some parts of the code are adapted from the Sugp GitHub repository. We have made modifications to the original code structure to improve its performance and usability. The parameters used in our experiments are inspired by the original tests presented in the corresponding paper.

## Notes
Please be aware that some parts of the code have been modified from the original Sugp implementation to better suit our requirements and improve the overall structure of the code.
