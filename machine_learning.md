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

<div align="center"><img src="http://latex.codecogs.com/gif.latex?h_{\theta}(x)%20=%20\theta_{0}%20+%20\theta_{1}x" /></div>

#### Cost Function
- thetas: parameters
- Different parameters give different hypothesis
- Come up with paramters that fit the data well - choose paramters so that h(x) is close to y for training example (x, y)
- One cost function:

<div align="center"><img src="http://latex.codecogs.com/gif.latex?J(\theta_{0},%20\theta_{1})%20=%20\sum\limits_{i=1}^m%20\frac{1}{2m}(h_{\theta}(x^{i})%20-%20y^{i})^2" /></div>
- Goal:

<div align="center"><img src="http://latex.codecogs.com/gif.latex?\underset{\theta_{0},%20\theta{1}}{\test{minimize}}J(\theta_{0},%20\theta_{1})"></div>
- Intuition: sum of distance between hypothesis and the actual values

#### Gradient Descent
- Idea: start with some initial paramters; keep changing them to reduce cost
- Algorithm: repeat until convergence

<div align="center"><img src="http://latex.codecogs.com/gif.latex?\theta_{j}:=\theta_{j}-\alpha\frac{\partial}{\partial%20\theta{j}}J(\theta_{0},%20\theta_{1})" /></div>
- Derivates: tangent of the point in the function J
- Alpha: learning rate - how large the step it takes
	- Too small: the algorithm can be small
	- Too large: overshoot the minimum, may never converge
	- As we approach local minimum, gradient descent will automatically take smaller step; so no need to decrease alpha
- Simultaneously update parameters
- Apply gradient descent to cost function
- Batch gradient descent: use all training examples in each step

## Week 2
### Multiple Features (Multivariate Linear Regression)
- Hypothesis:

<div align="center"><img src="http://latex.codecogs.com/gif.latex?h_{\theta}(x)=\theta^\intercal%20X" /></div>
<div align="center"><img src="http://latex.codecogs.com/gif.latex?\theta=\begin{bmatrix}\theta_{0}\\\theta_{1}\\\vdots\\\theta_{n}\end{bmatrix}" />&nbsp;<img src="http://latex.codecogs.com/gif.latex?X=\begin{bmatrix}x_{0}\\x_{1}\\\vdots\\x_{n}\end{bmatrix}" /></div>
- x0 is equal to 1

### Gradient Descent
- Cost function:

<div align="center"><img src="http://latex.codecogs.com/gif.latex?J(\theta)%20=%20\sum\limits_{i=1}^m%20\frac{1}{2m}(h_{\theta}(x^{i})%20-%20y^{i})^2" /></div>
- Algorithm:

<div align="center"><img src="http://latex.codecogs.com/gif.latex?\theta_{j}:=\theta_{j}-\alpha\frac{1}{m}\sum\limits_{n=1}^m(h_{\theta}(x^{(i)})-y^{(i)})x^{(i)}_{j}" /></div>
- Simultaneously update all parameters
- Have the same form as gradient descent with single variable

### Feature Scaling
- Make sure features are on a similar scale
- It will take a long time for gradient descent to converge if they are not
- Get every feature into approximately [-1, 1]
- Can also use mean normalization
	- replace xi with (xi - mean) / [(max - min) | stdev]

### Learning Rate
- Making sure that gradient descent work correctly
	- Plot cost function versus number of iterations
	- The value should decrease after each iteration
	- Use automatic convergence test: declare convergence after the decrease is smaller than a threshold
- If it is not working correctly
	- Use small learning rate
	- But a small learning rate will make convergence slow
- In summary
	- If learning rate is small, slow convergence
	- If learning rate is large, may not converge
	- Try a range of learning rate in practice

### Features and Polynominal Regression
- Sometimes choosing different features yield a better model
- Polynominal regression
	- For example, suppose we want to fit a model:

<div align="center"><img src="http://latex.codecogs.com/gif.latex?h_{\theta}=\theta_{0}+\theta_{1}x_{1}+\theta_{2}x_{2}+\theta_{3}x_{3}=\theta_{0}+\theta_{1}(size)+\theta_{2}(size)^2+\theta_{3}(size)^3" /></div>
	- We can turn it to a linear regression by having:

<div align="center"><img src="http://latex.codecogs.com/gif.latex?x_{1}=(size)\hspace{5pt}x_{2}=(size)^2\hspace{5pt}x_{3}=(size)^3" /></div>
	- Features scaling becomes very important
	- By choosing different features you can get different model 

### Normal Equation
- Method for solving theta analytically
- Intuition:

<div align="center"><img src="http://latex.codecogs.com/gif.latex?J(\theta)%20=%20\sum\limits_{i=1}^m%20\frac{1}{2m}(h_{\theta}(x^{i})%20-%20y^{i})^2" /></div>
<div align="center"><img src="http://latex.codecogs.com/gif.latex?\frac{\partial}{\partial%20\theta_{j}}J_{\theta}=\ldots=0" /></div>
<div align="center"><img src="http://latex.codecogs.com/gif.latex?\text{Solve%20for}\hspace{10pt}\theta_{0},\theta_{1},\ldots,\theta_{n}" /></div>
- Optimal Solution

<div align="center"><img src="http://latex.codecogs.com/gif.latex?\theta=(X^\intercal%20X)^{-1}X^\intercal%20y" /></div>
- No need to do feature scaling
- Comparison with gradient descent
	- Gradient descent
		- Need to choose learning rate
		- Need many iterations
		- Works well even when n (number of features) is large
	- Normal equation
		- No need to choose learning rate
		- Don't need iterations
		- Need to compute (X'X)^-1 - O(n^3)
		- Slow if n is large
		- Doesn't work for some problems

### Normal Equation and Noninvertibility
- What if X'X is non-invertible
	- May have redundant features
	- Too many features
		- Delete some features or use regularization
- pinv (pseudo inverse) in Octave will calculate correctly even X is not invertible
