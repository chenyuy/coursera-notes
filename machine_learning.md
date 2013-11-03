# Machine Learning

## Week 1
### Introduction
- A computer program learn from experience E with respect to some task T and some performance measure P, if its performance on T as measured by P improves with experience E
- Supervised learning
	- In the dataset we are told the right answer of each data
	- Regression - predict value
	- Classification
- Unsupervised learning
	- Given date with no labels
	- Clustering

### Linear Regression with One Variable

#### Model Representation
- Training data ( (x1, y1), (x2, y2), ... )-> learning algorithm -> hypothesis
- A linear regression:

<div align="center" ><img src="http://latex.codecogs.com/gif.latex?h_{\theta}(x)%20=%20\theta_{0}%20+%20\theta_{1}x" /></div>

#### Cost Function
- thetas: parameters
- Different parameters give different hypothesis
- Come up with paramters that fit the data well - choose paramters so that h(x) is close to y for training example (x, y)
- One cost function:

<div align="center" ><img src="http://latex.codecogs.com/gif.latex?J(\theta_{0},%20\theta_{1})%20=%20\sum\limits_{i=1}^m%20\frac{1}{2m}(h_{\theta}(x^{i})%20-%20y^{i})^2" /></div>
- Goal:

<div align="center"><img src="http://latex.codecogs.com/gif.latex?\underset{\theta_{0},%20\theta{1}}{\test{minimize}}J(\theta_{0},%20\theta_{1})"></div>
- Intuition: sum of distance between hypothesis and the actual values
