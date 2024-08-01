## Recognising human handwritten numbers

Making a neural network that can be trained to recognise human handwritten characters.

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




