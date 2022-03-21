import numpy as np
import matplotlib.pyplot as plt

x = 2*np.random.rand(100,1)
y = 4 + 3*x + np.random.randn(100,1)
plt.plot(x,y,"b.")
plt.axis([0,2,0,15])
# plt.show()

#normal equation --> theta_OLS
X_design = np.c_[np.ones((100, 1)), x]
theta_OLS = np.linalg.inv(X_design.T.dot(X_design)).dot(X_design.T).dot(y)

print(X_design)
print(theta_OLS)

#get the plpt
X_test = np.array([[0], [2]])
X_test_design = np.c_[np.ones((2, 1)), X_test]
y_predict = X_test_design.dot(theta_OLS)
print(y_predict)

plt.plot(X_test, y_predict, "r--")
plt.axis([0,2,0,15])
plt.show()





















