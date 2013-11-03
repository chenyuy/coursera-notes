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
- A linear regression - hypothesis(x) = theta0 + theta1 * x = y (output)

#### Cost Function
- thetas: parameters
- Different parameters give different hypothesis
- Come up with paramters that fit the data well - choose paramters so that h(x) is close to y for training example (x, y)
- One cost function: J(theta0, theta1) = ( 1/2m sum((h(xi) - yi))^2 ) over i = 1 to i
- Goal: min(J(theta0, theta1))
- Intuition: sum of distance between hypothesis and the actual values
