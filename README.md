## Recognising human handwritten numbers

Making a neural network that can be trained to recognise human handwritten characters.

```
$ clang -Wall train.c common.c -lm -o train
$ ./train
Training data: mnist_dataset/mnist_train.csv
Size of training data 376800000 Bytes: 359.34 MB
epoch 0
epoch 1
epoch 2
epoch 3
epoch 4
Time spent:     385.85 seconds.
```

```
$ clang -Wall query.c common.c -lm -o query
$ ./query                                   
Testing data: mnist_dataset/mnist_test.csv
Size of testing data 62800000 Bytes: 59.89 MB
Size of Wih 1254400 Bytes: 1.20 MB
Size of Who 16000 Bytes: 15.62 KB
Performance score: 0.97
```

Note: a collection of images of handwritten numbers used in this project is called MNIST database of handwritten digits, and is available from the neural network researcher Yann LeCun’s website ​http://yann.lecun.com/exdb/mnist/.​

Training​ set ​http://www.pjreddie.com/media/files/mnist_train.csv

Test​ set ​http://www.pjreddie.com/media/files/mnist_test.csv



