# Regression-Methods
Comparing The Effectiveness of Two Regression Methods:
Stochastic Gradient Descent vs. Multi Layer Perception Neural Networks

In this project, I challenged myself to learn about machine learning algorithms and their implementation. 
I self-taught everything I needed to know for this machine learning project through Coursera, as well as brushing up on my python skills throughout. 
This is my first time programming anything machine learning related. 

The following code uses regression models and the California Housing Dataset from Scikit Learn 
to compare the effectiveness of the stochastic gradient descent method and the neural networks method. 
The time to train each method is provided as well as their individual residual score. 
A graphic was created to visualize the difference in the methods.


Initially, my trained models were producing very inaccurate results, and sometimes even predicting housing costs to be in the negatives. After tinkering with my code a bit more
I found that I needed to implement a scaler as well as a quantile transformer (to fit the data to a normal distribution) to produce the most accurate results (residuals <1).

