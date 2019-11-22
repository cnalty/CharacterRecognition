# A Basic Character Recognition CNN

## Introduction

This repo is a basic implementation of a CNN for character recognition on 28x28 signle channel images of characters.
The network has 4 convolutional layers, 2 batch normalization layers, 2 max pooling layers, and 2 fully connected layers. 

train.py is currently set up to train on the MINST digit recognition data set that has 10 digits (0-9) using cuda. 
If you wish to train on a data set with a different number of characters simply adjust the model decleration passing
the desired number of characters.

## Kaggle Results
### Digit Recognizer
| Date     | Score |
|----------|-------|
| 11/21/19 | **0.984** |
| 11/21/19 | 0.981 |
| 11/21/19 | 0.978 |
| 11/18/19 | 0.979 |
| 11/18/19 | 0.934 |
    
### Kannada MNIST
| Date     | Score |
|----------|-------|
| 11/21/19 | **0.962** |


## History of improvements

The network started out with no batch normalization layers. Without these layers the network had a much slower convergence
and a noticibly worse performance than after adding them. The network would stay stuck with it's starting loss of 2.3 
(as expected for Cross-Entropy Loss), and remained at this level for a third to half of the epochs before begining to converge.
It also caused the minimum loss to never surpase 0.1. After adding the batch normalization layers the models loss dropped
significantly at the very begining of training, converged much quicker, and reached a lower final loss of approximately 0.04.
This shows the importance of normalization for convolutional nets.

## Version History / Experiments
* 11/21/19 - Conv -> Batch Norm -> Conv -> Batch Norm -> Conv -> Batch Norm -> Max Pool -> Conv -> Batch Norm -> Conv -> Batch Norm -> Max Pool -> FC -> FC
  * Doubled most Conv layers number of filters
  * Slight Improvement in Accuracy to 0.984
* 11/21/19 - Conv -> Batch Norm -> Conv -> Batch Norm -> Conv -> Batch Norm -> Max Pool -> Conv -> Batch Norm -> Conv -> Batch Norm -> Max Pool -> FC -> FC
  * Slight Improvement in Accuracy to 0.981
* 11/21/19 - Conv -> Batch Norm -> Conv -> Batch Norm -> Max Pool -> Conv -> Batch Norm -> Conv -> Batch Norm -> Max Pool -> FC -> FC
  * Tested with shrinking last FC layer neurons, worse results than keeping constant neuron amount
  * Approximately same loss as 1 FC
* 11/21/19 - Conv -> Batch Norm -> Conv -> Batch Norm -> Max Pool -> Conv -> Batch Norm -> Conv -> Batch Norm -> Max Pool -> FC
  * Double filters in last Conv layer
  * Approximately Same accuracy of 0.978
* 11/21/19 - Conv -> Batch Norm -> Conv -> Max Pool -> Conv -> Batch Norm -> Conv -> Max Pool -> FC
  * Batch Norms moved to before relu
  * Slightly Faster convergence
* 11/18/19 - Conv -> Batch Norm -> Conv -> Max Pool -> Conv -> Batch Norm -> Conv -> Max Pool -> FC
  * Batch Norms were after relu
  * Much faster Convergence
  * Better Accuracy of 0.979
* 11/18/19 - 2 Convs -> Max Pool -> 2 Convs -> Max Pool -> FC
  * Slow Convergence
  * 0.934 Accuracy
* 11/18/19 - 2 Convs -> Max Pool -> 2 Convs -> Max Pool -> Conv -> FC
  * Result - No Convergence
