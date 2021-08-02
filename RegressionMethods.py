from sklearn.linear_model import SGDRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.neural_network import MLPRegressor
from time import time
import matplotlib.pyplot as plt


# California Dataset Preprocessing
X_cali, Y_cali = fetch_california_housing(return_X_y = True)

# Extract 15% of the data for testing and use the remaining data for training
X_train, X_test, Y_train, Y_test = train_test_split(X_cali, Y_cali, test_size=0.15)


# Training the Model Using Gradient Descent Method
print("Training the Model Using Gradient Descent...")
tic = time()
SGD = make_pipeline(StandardScaler(), QuantileTransformer(), SGDRegressor(max_iter=30000))
SGD.fit(X_train,Y_train)

print(f"done in {time() - tic:.3f}s")

score = SGD.score(X_train, Y_train)
print("R-squared:", score)
print()


# Using the Trained SGD Model to Predict Housing Cost
Y_pred = SGD.predict(X_test)


# Training the Model Using Neural Networks Method
print("Training the Model Using Neural Networks...")
MLP = make_pipeline(StandardScaler(), QuantileTransformer(), MLPRegressor(hidden_layer_sizes=(50, 50),
                                                                          learning_rate='adaptive',
                                                                          learning_rate_init=0.02,))
MLP.fit(X_train, Y_train)

print(f"done in {time() - tic:.3f}s")

score2 = MLP.score(X_train, Y_train)
print("R-squared:", score2)

# Using Trained MLP Model to Predict Housing Costs
Y_MLPred = MLP.predict(X_test)


# Visualization of Predictions Compared to Actual
xaxis = range(len(Y_pred))

fig, [ax1, ax2] = plt.subplots(2, 1)
fig.suptitle("California Housing Cost Prediction vs. Actual")
ax1.plot(xaxis,Y_test, label = "Actual")
ax1.plot(xaxis,Y_pred, label = "SGD Prediction", color='r')

ax1.legend(loc='upper right', fontsize='small',markerscale=0.6, fancybox=True)
ax1.grid(True)

ax2.plot(xaxis,Y_test, label = "Actual")
ax2.plot(xaxis, Y_MLPred, label="MLP Prediction")
ax2.legend(loc='upper right', fontsize='small',markerscale=0.6, fancybox=True)
ax2.set_xlabel("California Block Example No.")
ax2.set_ylabel("Average House Value in 100,000's", loc='bottom')
ax2.grid(True)
plt.show()