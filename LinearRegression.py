import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)
# plt.plot(x,y,"b.")
plt.axis([0, 2, 0, 15])
# plt.show()

# Normal equation --> theta_OLS
X_design = np.c_[np.ones((100, 1)), x]
theta_OLS = np.linalg.inv(X_design.T.dot(X_design)).dot(X_design.T).dot(y)

print(X_design)
print(theta_OLS)

# get the plpt
X_test = np.array([[0], [2]])
X_test_design = np.c_[np.ones((2, 1)), X_test]
y_predict = X_test_design.dot(theta_OLS)
print(y_predict)

# plt.plot(X_test, y_predict, "r--")
plt.axis([0, 2, 0, 15])
# plt.show()

# use sklearn directly
lin_reg = LinearRegression()
lin_reg.fit(x, y)
print(lin_reg.coef_)
print(lin_reg.intercept_)

# Polynomial Regression

## prepare data
count = 100
x = 6 * np.random.rand(count, 1) - 3
y = 0.5 * x ** 2 + x + np.random.rand(count, 1)

# plt.plot(x, y, "b.")
plt.axis([-3, 3, -5, 10])
# plt.show()

poly_features = PolynomialFeatures(degree=2, include_bias=False)
# transform the input value
x_poly = poly_features.fit_transform(x)
# fit the data and get the formula
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)
print(lin_reg.coef_)
print(lin_reg.intercept_)
# get fitted line
fitted_x = np.linspace(-3, 3, 100).reshape(100, 1)
print("--------------fitted_x--------------")
print(fitted_x)
fitted_x_poly = poly_features.transform(fitted_x)
y_predict = lin_reg.predict(fitted_x_poly)
# plt.plot(x, y, "b.")
plt.axis([-3, 3, -5, 10])
# plt.plot(fitted_x, y_predict, "r--",label="predication")
# plt.legend()
plt.show()
'''
Overfitting
'''

# Polynomial Regression comparision on Degree
for style, width, degree in (("g--", 1, 1), ("b--", 5, 2), ("r--", 3, 10)):
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    std = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_reg = Pipeline([('poly_features', poly_features), ('StandardScaler', std), ('lin_reg', lin_reg)])
    polynomial_reg.fit(x, y)
    y_predict = polynomial_reg.predict(fitted_x)
    plt.plot(fitted_x, y_predict, style, label=str(degree), linewidth=width)
plt.plot(x, y, ".b")
plt.axis([-3, 3, -5, 10])
plt.legend()
plt.show()

# count of the sample

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def plot_sampleCount_RMSE_curve(regression, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    train_errors, test_errors = [], []
    for i in range(1, len(x_train)):
        regression.fit(x_train[:i], y_train[:i])
        y_train_predict = regression.predict(x_train[:i])
        y_test_predict = regression.predict(x_test)
        train_errors.append(np.sqrt(mean_squared_error(y_train[:i], y_train_predict)))
        test_errors.append(np.sqrt(mean_squared_error(y_test, y_test_predict)))
    plt.plot(train_errors, 'r--', linewidth=2, label="train_error")
    plt.plot(test_errors, 'b--', linewidth=2, label="test_errors")
    plt.xlabel = "training sample count"
    plt.ylabel = "RMSE"
    plt.legend()


'''
as the sample count grows, (the train error - the train error) reduces, the likelihood of overfitting reduces 
'''
lin_reg = LinearRegression()
plot_sampleCount_RMSE_curve(lin_reg, x, y)
plt.axis([0, 100, 0, 5])
plt.show()
