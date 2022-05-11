This is the repository for the DeepConcolic tool with further additions. To see the original project go here: https://github.com/TrustAI/DeepConcolic/

The additions and distribution of this project is purely for research reasons, which is allowed by the original work authors under their License.

The current additions include: l1-norm, Frodeus-norm, nuclear-norm. As well as added datasets: mnist, cifar10, cats and dogs; and appropriate deep neural networks (see research paper for further details).

# Installation

First of all, set up a conda environment

```sh
conda create --name deepconcolic python==3.7
conda activate deepconcolic
```
This should be followed by installing software dependencies:
```sh
conda install opencv nltk matplotlib
conda install -c pytorch torchvision
pip3 install numpy==1.19.5 scipy==1.4.1 tensorflow\>=2.4 pomegranate==0.14 scikit-learn scikit-image pulp keract np_utils adversarial-robustness-toolbox parse tabulate pysmt saxpy keras menpo patool z3-solver pyvis
```

# Examples
To run the tests use parameters accordingly. For instance to run a test for mnist dataset, together with simple neural network, with l-infinity norm, ssc criterion:
```
python3 -m deepconcolic.main --outputs outs/simple-mnist-ssc --dataset mnist --model saved_models/simple_DNN_for_mnist.h5 --criterion ssc --norm linf
```
It should be possible to replicate all the tests, however some of the DNNs are missing: this is due to the github file size limits. However all neural networks can be generated using DeepModels repository:https://github.com/sielos/DeepModels. Further instructions are provided there as well.
