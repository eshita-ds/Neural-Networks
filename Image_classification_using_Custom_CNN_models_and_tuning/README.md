# Sports Image classification using Custom CNN models and Fine Tuning

## Introduction
Image classification with deep learning trains neural networks to automatically recognize and categorize visual content into predefined classes. Using convolutional neural networks (CNNs), it processes spatial and pixel-level relationships in images through layers like convolutional, pooling, and fully connected layers. These layers extract features ranging from edges to complex objects, enabling models to generalize across variations in lighting, angles, and backgrounds.

The process involves training on labeled datasets to map images to their correct labels using optimization techniques like gradient descent. Once trained, the model can classify unseen images, with applications in medical imaging, autonomous driving, facial recognition, and more.

## Problem Statement
Train a neural network with different number of layers and neurons in each layer. Experiment with different combinations of Epochs, Batch Sizes, Activation functions, Different regularization techniques, different loss functions and optimization algorithms. Identify which combination gives the best result to establish which model configuration is best suited for this classification task.

## Dataset

The dataset contains images of different sports classes and is split into a training set and a test set. The training set consists of labeled images belonging to the following sports classes: cricket, wrestling, tennis, badminton, soccer, swimming, and karate. Each image is associated with a unique image ID and its corresponding class label. The test set contains unlabeled images for which you need to predict the class.

**Link**: `https://www.kaggle.com/datasets/sidharkal/sports-image-classification/data`

**Sample Images**

<img width="500" alt="image" src="https://github.com/user-attachments/assets/3df1fc17-c69f-4253-85f4-5171c3fc9f62" />


## Requirements

- Python 3.9 or higher
- Libraries
  - Pandas
  - Numpy
  - Scikit-Learn
  - Pytorch
 
## Modeling and Conclusion
We tried multiple configurations for CNN models with our pre-processed dataset and following are the outcomes with different base configurations.

Comparing the metrics for all models. We see that the best performing models are Setup 5, Setup 8 and Setup 9. We see the Setup 9 shows the best metrics with lowest loss and lowest training time. However, we see the model significantly overfits which will degrade the model’s performance on Unseen Data. Another model with good performance is Setup 8 which is again having high accuracy. However, it’s also significantly overfitting which can cause degradation of performance.
The most balanced model is Setup 5, which shows a consistent trend for decreasing losses for both training and validation. Overall Accuracy is high; however, the losses might be a bit higher as compared the other models. However, since it does not overfit, it generalizes well between training and unseen data.


|<img width="1126" alt="image" src="https://github.com/user-attachments/assets/ddf6f321-681e-47b6-8bc5-507a1d987457" />|<img width="350" alt="image" src="https://github.com/user-attachments/assets/13a3855e-711a-4ab2-84fb-2287df682d9c" />|
|---|---|

## License
MIT License

Copyright (c) 2024 Eshita Gupta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

