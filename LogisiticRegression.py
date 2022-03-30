'''
Iris dataset
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# load dataset
iris = load_iris()
print(iris.DESCR)
keys = iris.keys()
print(keys)
print(iris["data"][:, 3:])
print(iris["target"])

# prepare input and output
x = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int0)

# logistic regression
log_res = LogisticRegression()
log_res.fit(x, y)

# predicate probali
x_predicate = np.linspace(0, 3, 1000).reshape(-1, 1)
y_predicate = log_res.predict_proba(x_predicate)
plt.figure(figsize=(10, 5))
decision_boundary = x_predicate[y_predicate[:, 1] >= 0.5][0]
plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:")
plt.plot(x_predicate, y_predicate[:, 1], "g--", "Iris")
plt.plot(x_predicate, y_predicate[:, 0], "b--", "not Iris")
plt.show()
