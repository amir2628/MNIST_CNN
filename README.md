# MNIST_CNN
A simple Convolutional Neural Network (CNN) to classify images from the Keras MNIST datset

## Dataset
The dataset is from Keras, called Fashion_MNIST. This dataset has a very poor quality and the result of the NN can be not so good as some images related to different classes can be mistaken for other classes.

## Result

```
              precision    recall  f1-score   support

           0       0.89      0.83      0.86      1000
           1       0.99      0.98      0.99      1000
           2       0.86      0.88      0.87      1000
           3       0.91      0.92      0.91      1000
           4       0.83      0.89      0.86      1000
           5       0.99      0.98      0.98      1000
           6       0.75      0.73      0.74      1000
           7       0.95      0.98      0.97      1000
           8       0.99      0.98      0.99      1000
           9       0.98      0.96      0.97      1000

    accuracy                           0.91     10000
   macro avg       0.91      0.91      0.91     10000
weighted avg       0.91      0.91      0.91     10000
```

![Figure_1](https://user-images.githubusercontent.com/56083377/221143005-21c71805-dfd2-4e57-b808-b49b2bdf0608.png)

![Figure_2](https://user-images.githubusercontent.com/56083377/221143033-121558b5-3fde-4905-bede-faa06900bcd7.png)
