# Number Identifying Neural Network

This is a neural network that identifies handwritten numbers,
trained on MNIST data.

I made this following [this excellent guide](http://neuralnetworksanddeeplearning.com/chap1.html)

## To run

1. clone this repo
2. download the [data](https://github.com/MichalDanielDobrzanski/DeepLearningPython/blob/master/mnist.pkl.gz)
3. place the `mnist.pkl.gz` file in the cloned folder
4. download [mnist loader](https://github.com/MichalDanielDobrzanski/DeepLearningPython/blob/master/mnist_loader.py)
5. place that into cloned folder
6. make the following changes to `mnist_loader.py`

```python
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
```

7. in python shell run the following

```python
import mnist_loader
import main
training_data,validation_data,test_data = mnist_loader.load_data_wrapper()
net = main.Network([784,30,10])
net.sgd(training_data,30,10,3.0,test_data=test_data)
```
