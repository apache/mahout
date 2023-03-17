<!--
 Licensed to the Apache Software Foundation (ASF) under one or more
 contributor license agreements.  See the NOTICE file distributed with
 this work for additional information regarding copyright ownership.
 The ASF licenses this file to You under the Apache License, Version 2.0
 (the "License"); you may not use this file except in compliance with
 the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->
---
layout: default
title: Multilayer Perceptron

    
---

Multilayer Perceptron
=====================

A multilayer perceptron is a biologically inspired feed-forward network that can 
be trained to represent a nonlinear mapping between input and output data. It 
consists of multiple layers, each containing multiple artificial neuron units and
can be used for classification and regression tasks in a supervised learning approach. 

Command line usage
------------------

The MLP implementation is currently located in the MapReduce-Legacy package. It
can be used with the following commands: 


# model training
    $ bin/mahout org.apache.mahout.classifier.mlp.TrainMultilayerPerceptron 
# model usage
    $ bin/mahout org.apache.mahout.classifier.mlp.RunMultilayerPerceptron


To train and use the model, a number of parameters can be specified. Parameters without default values have to be specified by the user. Consider that not all parameters can be used both for training and running the model. We give an example of the usage below.

### Parameters

| Command | Default | Description | Type |
|:---------|---------:|:-------------|:---------|
| --input -i | | Path to the input data (currently, only .csv-files are allowed) | |
| --skipHeader -sh | false | Skip first row of the input file (corresponds to the csv headers)| |
|--update -u | false | Whether the model should be updated incrementally with every new training instance. If this parameter is not given, the model is trained from scratch. | training |
| --labels -labels | | Instance labels separated by whitespaces. | training |
| --model -mo | | Location where the model will be stored / is stored (if the specified location has an existing model, it will update the model through incremental learning). | |
| --layerSize -ls | | Number of units per layer, including input, hidden and ouput layers. This parameter specifies the topology of the network (see [this image][mlp] for an example specified by `-ls 4 8 3`). | training |
| --squashingFunction -sf| Sigmoid | The squashing function to use for the units. Currently only the sigmoid fucntion is available. | training |
| --learningRate -l | 0.5 | The learning rate that is used for weight updates. | training |
| --momemtumWeight -m | 0.1 | The momentum weight that is used for gradient descent. Must be in the range between 0 ... 1.0 | training |
| --regularizationWeight -r | 0 | Regularization value for the weight vector. Must be in the range between 0 ... 0.1 | training |
| --format -f | csv | Input file format. Currently only csv is supported. | |
|--columnRange -cr | | Range of the columns to use from the input file, starting with 0 (i.e. `-cr 0 5` for including the first six columns only) | testing |
| --output -o | | Path to store the labeled results from running the model. | testing |

Example usage
-------------

In this example, we will train a multilayer perceptron for classification on the iris data set. The iris flower data set contains data of three flower species where each datapoint consists of four features.
The dimensions of the data set are given through some flower parameters (sepal length, sepal width, ...). All samples contain a label that indicates the flower species they belong to.

### Training

To train our multilayer perceptron model from the command line, we call the following command


    $ bin/mahout org.apache.mahout.classifier.mlp.TrainMultilayerPerceptron \
                -i ./mrlegacy/src/test/resources/iris.csv -sh \
                -labels setosa versicolor virginica \
                -mo /tmp/model.model -ls 4 8 3 -l 0.2 -m 0.35 -r 0.0001


The individual parameters are explained in the following.

- `-i ./mrlegacy/src/test/resources/iris.csv` use the iris data set as input data
- `-sh` since the file `iris.csv` contains a header row, this row needs to be skipped 
- `-labels setosa versicolor virginica` we specify, which class labels should be learnt (which are the flower species in this case)
- `-mo /tmp/model.model` specify where to store the model file
- `-ls 4 8 3` we specify the structure and depth of our layers. The actual network structure can be seen in the figure below.
- `-l 0.2` we set the learning rate to `0.2`
- `-m 0.35` momemtum weight is set to `0.35`
- `-r 0.0001` regularization weight is set to `0.0001`
 
|  |  |
|---|---|
| The picture shows the architecture defined by the above command. The topolgy of the network is completely defined through the number of layers and units because in this implementation of the MLP every unit is fully connected to the units of the next and previous layer. Bias units are added automatically. | ![Multilayer perceptron network][mlp] |

[mlp]: mlperceptron_structure.png "Architecture of a three-layer MLP"
### Testing

To test / run the multilayer perceptron classification on the trained model, we can use the following command


    $ bin/mahout org.apache.mahout.classifier.mlp.RunMultilayerPerceptron \
                -i ./mrlegacy/src/test/resources/iris.csv -sh -cr 0 3 \
                -mo /tmp/model.model -o /tmp/labelResult.txt
                

The individual parameters are explained in the following.

- `-i ./mrlegacy/src/test/resources/iris.csv` use the iris data set as input data
- `-sh` since the file `iris.csv` contains a header row, this row needs to be skipped
- `-cr 0 3` we specify the column range of the input file
- `-mo /tmp/model.model` specify where the model file is stored
- `-o /tmp/labelResult.txt` specify where the labeled output file will be stored

Implementation 
--------------

The Multilayer Perceptron implementation is based on a more general Neural Network class. Command line support was added later on and provides a simple usage of the MLP as shown in the example. It is implemented to run on a single machine using stochastic gradient descent where the weights are updated using one datapoint at a time, resulting in a weight update of the form:
$$ \vec{w}^{(t + 1)} = \vec{w}^{(t)} - n \Delta E_n(\vec{w}^{(t)}) $$

where *a* is the activation of the unit. It is not yet possible to change the learning to more advanced methods using adaptive learning rates yet. 

The number of layers and units per layer can be specified manually and determines the whole topology with each unit being fully connected to the previous layer. A bias unit is automatically added to the input of every layer. 
Currently, the logistic sigmoid is used as a squashing function in every hidden and output layer. It is of the form:

$$ \frac{1}{1 + exp(-a)} $$

The command line version **does not perform iterations** which leads to bad results on small datasets. Another restriction is, that the CLI version of the MLP only supports classification, since the labels have to be given explicitly when executing on the command line. 

A learned model can be stored and updated with new training instanced using the `--update` flag. Output of classification reults is saved as a .txt-file and only consists of the assigned labels. Apart from the command-line interface, it is possible to construct and compile more specialized neural networks using the API and interfaces in the mrlegacy package. 


Theoretical Background
-------------------------

The *multilayer perceptron* was inspired by the biological structure of the brain where multiple neurons are connected and form columns and layers. Perceptual input enters this network through our sensory organs and is then further processed into higher levels. 
The term multilayer perceptron is a little misleading since the *perceptron* is a special case of a single *artificial neuron* that can be used for simple computations [\[1\]][1]. The difference is that the perceptron uses a discontinous nonlinearity while for the MLP neurons that are implemented in mahout it is important to use continous nonlinearities. This is necessary for the implemented learning algorithm, where the error is propagated back from the output layer to the input layer and the weights of the connections are changed according to their contribution to the overall error. This algorithm is called backpropagation and uses gradient descent to update the weights. To compute the gradients we need continous nonlinearities. But let's start from the beginning!

The first layer of the MLP represents the input and has no other purpose than routing the input to every connected unit in a feed-forward fashion. Following layers are called hidden layers and the last layer serves the special purpose to determine the output. The activation of a unit *u* in a hidden layer is computed through a weighted sum of all inputs, resulting in 
$$ a_j = \sum_{i=1}^{D} w_{ji}^{(l)} x_i + w_{j0}^{(l)} $$
This computes the activation *a* for neuron *j* where *w* is the weight from neuron *i* to neuron *j* in layer *l*. The last part, where *i = 0* is called the bias and can be used as an offset, independent from the input.

The activation is then transformed by the aforementioned differentiable, nonlinear *activation function* and serves as the input to the next layer. The activation function is usually chosen from the family of sigmoidal functions such as *tanh* or *logistic sigmoidal* [\[2\]][2]. Often sigmoidal and logistic sigmoidal are used synonymous. Another word for the activation function is *squashing function* since the s-shape of this function class *squashes* the input.

For different units or layers, different activation functions can be used to obtain different behaviors. Especially in the output layer, the activation function can be chosen to obtain the output value *y*, depending on the learning problem:
$$ y_k = \sigma (a_k) $$

If the learning problem is a linear regression task, sigma can be chosen to be the identity function. In case of classification problems, the choice of the squashing functions depends on the exact task at hand and often softmax activation functions are used. 

The equation for a MLP with three layers (one input, one hidden and one output) is then given by

$$ y_k(\vec{x}, \vec{w}) = h \left( \sum_{j=1}^{M} w_{kj}^{(2)} h \left( \sum_{i=1}^{D} w_{ji}^{(1)} x_i + w_{j0}^{(1)} \right) + w_{k0}^{(2)} \right) $$ 

where *h* indicates the respective squashing function that is used in the units of a layer. *M* and *D* specify the number of incoming connections to a unit and we can see that the input to the first layer (hidden layer) is just the original input *x* whereas the input into the second layer (output layer) is the transformed output of layer one. The output *y* of unit *k* is therefore given by the above equation and depends on the input *x* and the weight vector *w*. This shows us, that the parameter that we can optimize during learning is *w* since we can not do anything about the input *x*. To facilitate the following steps, we can include the bias-terms into the weight vector and correct for the indices by adding another dimension with the value 1 to the input vector. The bias is a constant factor that is added to the weighted sum and that serves as a scaling factor of the nonlinear transformation. Including it into the weight vector leads to:

$$ y_k(\vec{x}, \vec{w}) = h \left( \sum_{j=0}^{M} w_{kj}^{(2)} h \left( \sum_{i=0}^{D} w_{ji}^{(1)} x_i \right) \right) $$ 

The previous paragraphs described how the MLP transforms a given input into some output using a combination of different nonlinear functions. Of course what we really want is to learn the structure of our data so that we can feed data with unknown labels into the network and get the estimated target labels *t*. To achieve this, we have to train our network. In this context, training means optimizing some function such that the error between the real labels *y* and the network-output *t* becomes smallest. We have seen in the previous pragraph, that our only knob to change is the weight vector *w*, making the function to be optimized a function of *w*. For simplicitly and because it is widely used, we choose the so called *sum-of-squares* error function as an example that is given by

$$ E(\vec{w}) = \frac{1}{2} \sum_{n=1}^N \left( y(\vec{x}_n, \vec{w}) - t_n \right)^2 $$

The goal is to minimize this function and thereby increase the performance of our model. A common method to achieve this is to use gradient descent and the so called technique of *backpropagation* where the goal is to compute the contribution of every unit to the overall error and changing the weight according to this contribution and into the direction of the gradient of the error function at this particular unit. In the following we try to give a short overview of the model training with gradient descent and backpropagation. A more detailed example can be found in [\[3\]][3] where much of this information is taken from.

The problem with minimizing the error function is that the error can only be computed at the output layers where we get *t*, but we want to update all the weights of all the units. Therefore we use the technique of backpropagation to propagate the error, that we first compute at the output layer, back to the units of the previous layers. For this approach we also need to compute the gradients of the activation function. 

Weights are then updated with a small step in the direction of the negative gradient, regulated by the learning rate *n* such that we arrive at the formula for weight update:

$$ \vec{w}^{(t + 1)} = \vec{w}^{(t)} - n \Delta E(\vec{w}^{(t)}) $$

A momentum weight can be set as a parameter of the gradient descent method to increase the probability of finding better local or global optima of the error function.





[1]: http://en.wikipedia.org/wiki/Perceptron "The perceptron in wikipedia"
[2]: http://en.wikipedia.org/wiki/Sigmoid_function "Sigmoid function on wikipedia"
[3]: http://research.microsoft.com/en-us/um/people/cmbishop/prml/ "Christopher M. Bishop: Pattern Recognition and Machine Learning, Springer 2009"

References

\[1\] http://en.wikipedia.org/wiki/Perceptron

\[2\] http://en.wikipedia.org/wiki/Sigmoid_function

\[3\] [Christopher M. Bishop: Pattern Recognition and Machine Learning, Springer 2009](http://research.microsoft.com/en-us/um/people/cmbishop/prml/)

