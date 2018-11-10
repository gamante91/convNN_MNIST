# convNN_MNIST
A Neural Network with two convolutional layers and a fully connected output layer trained on the MNIST (http://yann.lecun.com/exdb/mnist/) dataset to classify digits.

The Neural Network is coded in PyTorch and has the following architecure:

layer1:
- Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
- BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
- ReLU()
- MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

layer2:
- Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
- BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
- ReLU()
- MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

output layer:
- Linear(in_features=1568, out_features=10, bias=True)

Choiche of hyper parameters:

Learning Rate: 0.001
Loss function: Cross Entropy Loss
Optimizer: Stochastic Gradient Descent

After being trained, the Neural Network reached an accuracy on the 10000 test images of 99.09%
