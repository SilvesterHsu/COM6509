from matplotlib import pyplot as plt
import numpy as np
a = np.random.normal(0,2,(150,2))*2
b = np.random.normal(5,2,(150,2))*2+5
y = np.hstack((np.zeros(150),np.ones(150)))
x = np.vstack((a,b))
plt.scatter(x[:,0], x[:,1],c = y,s=50, cmap='RdBu')

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x, y)

x[0:1]
model.predict(x[0:200])
x_test = np.random.rand(1000,2)*40-10
y_pre = model.predict(x_test)
plt.scatter(x_test[:,0], x_test[:,1],c = y_pre,s=50, cmap='RdBu', alpha=0.1)
