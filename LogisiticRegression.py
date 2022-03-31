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
# plt.figure(figsize=(10, 5))
decision_boundary = x_predicate[y_predicate[:, 1] >= 0.5][0]
# plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:")
# plt.plot(x_predicate, y_predicate[:, 1], "g--", "Iris")
# plt.plot(x_predicate, y_predicate[:, 0], "b--", "not Iris")
# plt.arrow(decision_boundary, 0.02, -1, 0, head_width=0.1, head_length=0.1, fc="b", ec="b")
# plt.arrow(decision_boundary, 1, 1, 0, head_width=0.1, head_length=0.1, fc="g", ec="g")
# plt.text(decision_boundary + 0.02, 0.15, "Decision Boundary", fontsize=16, color="k", ha="center")
# plt.xlabel("peta width(cm)", fontsize=10)
# plt.ylabel("prob", fontsize=10)
# plt.show()

'''
Decision Boundary
'''
x = iris["data"][:, (2, 3)]
y = (iris["target"] == 2).astype(np.int0)

log_res = LogisticRegression(C=10000)
log_res.fit(x, y)

# Construct axis chessboard
x1, x2 = np.meshgrid(np.linspace(2, 9, 500).reshape(-1, 1), np.linspace(0.8, 2.7, 200).reshape(-1, 1))
x_test = np.c_[x1.ravel(), x2.ravel()]
y_proba = log_res.predict_proba(x_test)
plt.figure(figsize=(10, 4))
plt.plot(x[y == 0, 0], x[y == 0, 1], "bs")
plt.plot(x[y == 1, 0], x[y == 1, 1], "g^")
z = y_proba[:, 1].reshape(x1.shape)
contour = plt.contour(x1, x2, z)
plt.clabel(contour, inline=1)
plt.axis([2.9, 7, 0.8, 2.7])
plt.text(3.5, 1.75, "not vir", fontsize=16)
plt.text(6.5, 1.5, "vir", fontsize=16)
plt.show()
