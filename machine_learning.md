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

<div align="center"><img src="http://latex.codecogs.com/gif.latex?\theta_{j}:=\theta_{j}-\alpha\frac{1}{m}\sum\limits_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})x^{(i)}_{j}" /></div>
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

## Week 3
### Logistic Regression
- Classification problmes
	- Classify into "0" (negative class) or "1" (positive class)
	- Or multi class problem (classify into "0", "1", "2", ...)
- Can use linear regression
	- Threshold classifier: > 0.5, predict "1", otherwise predict "0"
	- Not a good idea
		- Often it is right beacause you are lucky (e.g., specific training data)
		- Output value can be > 1 or < 0 while we want output to be 0 or 1

### Hypothesis Representation
- Want prediction to be between 0 and 1
- Use logistic (sigmoid) function

<div align="center"><img src="http://latex.codecogs.com/gif.latex?h_{\theta}(x)=\frac{1}{1%20+%20\mathrm{e}^{-{\theta}^{\intercal}x}}" /></div>
- Output can be interpreted as the probability that y = 1 given x, parameterized by theta

### Desicion Boundary
- Predict "1" if output >= 0.5, else predict "0"
- If the underlying regression outputs >= 0, the output will be >= 0.5
- Fit a line that separated the region where the hypothesis predicts 1 and that predicts 0
	- The line is desicion booundary
- Nonlinear decision boundaries
	- Fit a polynominal desicion boundary

### Cost Function
- Take the cost function from linear regression

<div align="center"><img src="http://latex.codecogs.com/gif.latex?J(\theta)=\frac{1}{m}\sum\limits_{i=1}^{m}Cost(h_{\theta}(x^{i}),y^{i})" /></div>
<div align="center"><img src="http://latex.codecogs.com/gif.latex?Cost(h_{\theta}(x),y)=\frac{1}{2}(h_{\theta}(x)-y)^2" /></div>
- This will give us a non-convex function (lots of local optima), because hypothesis uses logistic function
- Cost function for logistic regression

<div align="center"><img src="http://latex.codecogs.com/gif.latex?Cost(h_{\theta}(x),y)=\begin{cases}-\log(h_{\theta}(x))&\mbox{if%20}y=1\\%20-\log(1-h_{\theta}(x))&\mbox{if%20}y=0\end{cases}" /></div>
- Capture the intuition that if y = h, cost is 0. If h = 0 and y = 1, the cost is infinity and vice versa.

### Simplified Cost Function and Gradient Descent
- A simpler cost function

<div align="center"><img src="http://latex.codecogs.com/gif.latex?Cost(h_{\theta}(x),y)=-y\log(h_{\theta}(x))+(1-y)\log(1-h_{\theta}(x)})" /></div>
<div align="center"><img src="http://latex.codecogs.com/gif.latex?J(\theta)=-\frac{1}{m}[\sum\limits_{i=1}^{m}y^{(i)}\log%20h_{\theta}(x^{(i)})+(1-y^{(i)})\log(1-h_{\theta}(x^{(i)}})]" /></div>

- Can be derived from statistics using the principle of maximum liklihood
- Convex function

- To fit the parameters, minimize the cost function to get a set of parameters
	- Use gradient descent, the algorithm looks idential to linear regression
	- Feature scaling also applies to logistic regression

### Advanced Optimization
- Optimization algorithms
	- Conjugate gradient
	- BFGS
	- L-BFGS
- These algorithms
	- No need to pick up a learning rate
	- Often much faster to converge
	- More complex

### Multiclass Classification: One-vs-all
- One-vs-all
	- Suppose we have three classes
	- Turn the training data into three separate binary classification problems
	- Train three classifiers
	- Pick the class i that maximizes h

### The Problem of Overfitting
- Underfit (high bias)
	- Algorithm not fitting the data well
- Overfit (high variance)
	- If we have too many features, the learned hypothesis may fit the training set very well, but fail to generalize to new examples
- Addressing overfitting
	- Reduce number of features
		- Manually select features
		- Model selection to automatically select features
		- It also throws away useful information\
	- Regularization
		- Keep all features, but reduce magnitude/values of parameter
		- Works well if we have lots of features and each of them contributes a bit

### Cost Function
- If we have small values of parameters
	- We have a simpler hypothesis
	- Less prone to overfitting
- Take cost function and shrink all parameters

<div align="center"><img src="http://latex.codecogs.com/gif.latex?J(\theta)=-\frac{1}{2m}[\sum\limits_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2+\lambda\sum\limits_{j=1}^{n}\theta_{j}^2]" /></div>
- By convention, not penalizing theta0
- Lambda is a regularization parameter
	- Control the tradeoff between fitting the training set well and keeping parameters small
	- Too large lambda will penalize parameters too much, thus causing underfitting

### Regularized Linear Regression
- Gradient descent
	- Regualarization term shrinks theta a little for each iteration
	- For j = 0, the update rule doesn't change
	- For j = 1, 2, 3, ...

<div align="center"><img src="http://latex.codecogs.com/gif.latex?\theta_{j}:=\theta_{j}-\alpha[\frac{1}{m}\sum\limits_{i=1}^{m}(h_{\theta}(x^{(x)})-y^{(i)})x_{j}^{(i)}+\frac{\lambda}{m}\theta_{j}]" /></div>
- Normal Equation
	- Non-invertibility
		- Suppose number of training data <= number of features. If lambda > 0, the matrix will be invertable

<div align="center"><img src="http://latex.codecogs.com/gif.latex?\theta=(X^\intercal%20X+\lambda%20\begin{bmatrix}0&&&\\&1&&\\&&\ddots&\\&&&1\end{bmatrix})^{-1}X^\intercal%20y" /></div>

### Regularized Logistic Regression
- Add the regularization term to the cost function
	- Same update rule for linear regression

<div align="center"><img src="http://latex.codecogs.com/gif.latex?\mbox{Regularization%20term:%20}\frac{1}{2m}\sum\limits_{j=1}^{n}\lambda\theta_{j}" /></div>

## Week 4: Neural Networks: Representation
### Non-linear Hypothesis
- Many features
	- if include quadratic features, there are too many features and maybe overfitting
	- if not, not enough features to fit the data set
- Simple logistic regression with quadratic features added in is not a good way to learn complex hypothesis

### Neurons and Brain
- Origin: mimic brain
- Popularity: 80s, early 90s, diminished later
- Resurgence: more powerful computers

### Model Representation
- Neuron unit: logistic unit
	- Feed in some inputs, the neuron does some computation and outputs
	- A sigmoid(logistic) activation function
- Network: neuron unit wired together
	- Layer 1: input layer
	- Layer 2: hidden layer, can have many
	- Layer 3: output layer

<div align="center"><img src="https://dl.dropboxusercontent.com/u/55685931/neural%20network%20representation.png" /></div>
<div align="center"><img src="http://latex.codecogs.com/gif.latex?a_{i}^{(j)}=\mbox{%22activation%22%20of%20unit%20}i\mbox{%20in%20layer%20}j" /></div>
<div align="center"><img src="http://latex.codecogs.com/gif.latex?\Theta^{(j)}=\mbox{matrix%20of%20weights%20controlling%20function%20mapping%20from%20layer%20j%20to%20layer%20j%20+%201}" /></div>
<div align="center"><img src="http://latex.codecogs.com/gif.latex?a_{1}^{(2)}=g(\Theta_{10}^{(1)}x_{0}+\Theta_{11}^{(1)}x_{1}+\Theta_{12}^{(1)}x_{2}+\Theta_{13}^{(1)}x_{3})" /></div>
<div align="center"><img src="http://latex.codecogs.com/gif.latex?a_{2}^{(2)}=g(\Theta_{20}^{(1)}x_{0}+\Theta_{21}^{(1)}x_{1}+\Theta_{22}^{(1)}x_{2}+\Theta_{23}^{(1)}x_{3})" /></div>
<div align="center"><img src="http://latex.codecogs.com/gif.latex?a_{3}^{(2)}=g(\Theta_{30}^{(1)}x_{0}+\Theta_{31}^{(1)}x_{1}+\Theta_{32}^{(1)}x_{2}+\Theta_{33}^{(1)}x_{3})" /></div>
<div align="center"><img src="http://latex.codecogs.com/gif.latex?h_{\Theta}(x)=a_{1}^{(3)}=g(\Theta_{10}^{(2)}a_{0}^{(2)}+\Theta_{11}^{(2)}a_{1}^{(2)}+\Theta_{12}^{(2)}a_{2}^{(2)}+\Theta_{13}^{(2)}a_{3}^{(2)})" /></div>

- Forward propagation

<div align="center"><img src="http://latex.codecogs.com/gif.latex?z^{(2)}=\Theta^{(1)}a^{(1)}\\a^{(2)}=g(z^{(2)})\\a^{(2)}_{0}=1\\z^{(3)}=\Theta^{(2)}a^{(2)}\\h_{\Theta}(x)=a^{(3)}=g(z^{(3)})" /></div>
- Neural networks learn its own features
	- Like logistic regression
	- Use the computed features *a* insteand of the original feature *x*
	- Hidden layer computed more complex features
- Multi-class Classification
	- Suppose we have four classes
	- Have four output units
	- Want [1 0 0 0] for class 1, [0 1 0 0] for class 2, and so on
	- For training set, represent as [1 0 0 0] and so on

## Week 5 - Neural Networks: Learning
### Cost Function

<div align="center"><img src="http://latex.codecogs.com/gif.latex?J(\Theta)=-\frac{1}{m}[\sum\limits_{i=1}^{m}\sum\limits_{k=1}^{K}y_{k}^{(i)}\log%20(h_{\Theta}(x^{(i)}))_{k}+(1-y_{k}^{(i)})\log(1-(h_{\Theta}(x^{(i)}}))_{k}]+\frac{\lambda}{2m}\sum\limits_{j=l}^{L-1}\sum\limits_{i=1}^{s_{l}}\sum\limits_{j=1}^{s_{l+1}}(\Theta_{ji}^{l})^{2}" /></div>

### Backpropagation Algorithm
- Want to minimize cost function
- Need to compute cost function and gradient
- Intuition: calculate error of node j in layer l

<div align="center"><img src="http://latex.codecogs.com/gif.latex?\delta_{j}^{(4)}=a_{j}^{(4)}-y_{j}" /></div>

<div align="center"><img src="http://latex.codecogs.com/gif.latex?\delta_{j}^{(3)}=(\Theta^{(3)})^{\intercal}\delta^{(4)}.*g'(z^{(3)})" /></div>

<div align="center"><img src="http://latex.codecogs.com/gif.latex?\delta_{j}^{(2)}=(\Theta^{(2)})^{\intercal}\delta^{(3)}.*g'(z^{(2)})" /></div>

<div align="center"><img src="http://latex.codecogs.com/gif.latex?g'(z^{(i)})=a^{(i)}.*(1-a^{(i)})" /></div>

- Backpropagation algorithm

<div align="center"><img src="http://latex.codecogs.com/gif.latex?\mbox{Set%20}\Delta_{ij}^{(l)}=0" /></div>

```
For traning examples 1 to m
	Set a(1) = x(i)
	Perform forward propagation to compute a(l) for l = 2 to L
	Using y(i) to compute delta(L) = a(L) - y(i)
	Compute delta(L - 1) to delta(2)
```

<div style="padding-left: 6%"><img src="http://latex.codecogs.com/gif.latex?\Delta_{ij}^{(l)}:=\Delta_{ij}^{(l)}+a_{j}^{(l)}\delta_{i}^{(l+1)}" /></div>

<div align="center"><img src="http://latex.codecogs.com/gif.latex?D_{ij}^{(l)}:=\frac{1}{m}\Delta_{ij}^{(l)}+\lambda\Theta_{ij}^{(l)}\mbox{%20if%20}j\neq%200" /></div>

<div align="center"><img src="http://latex.codecogs.com/gif.latex?D_{ij}^{(l)}:=\frac{1}{m}\Delta_{ij}^{(l)}\mbox{%20if%20}j\eq%200" /></div>

- Formally delta is partial derivative of cost over *z*
- delta of *(i - 1)* is calculated by deltas from *i* weighted by the parameters

### Gradient Checking
- Calculate a approximate value of the derivative and compare with gradient

```matlab
for 1 = 1:n
	thetaPlus = theta;
	thetaPlus(i) = thetaPlus(i) + EPSILON;
	thetaMinus = theta;
	thetaMinus(i) = thetaPlus(i) - EPSILON;
	gradApprox(i) = (J(thetaPlus) - J(thetaMinus)) / (2 * EPSILON);
end;
% Compare with gradient
```

- Turn off gradient checking once the code is verified to be correct

### Random Initialization
- Initial value of 0 does not work for neural networks
	- All activiation ouputs are the same, the errors will also be same
	- Gradient will be the same
	- After each update, parameters corresponding to inputs going into each group of two hidden units are identical
- Initial to be value between -epsilon and epsilon

### Putting it together
- Pick a network architecture
	- Number of input units: dimension of input *x*
	- Number of output units: number of classes
	- Reasonable default: 1 hidden layer, or if >1 hiddent layer, use same number of hidden units in each layer (usually the more the better)
- Training network
	- Random initialization
	- Implement forward propagation
	- Implement cost function
	- Implement back propagation to get derivatives
	- Use gradient checking, then disable it
	- Use gradient descent or other advanced algorithm to minimize cost function
		- Cost function is non-convext, so will probably stuck in local minimium
