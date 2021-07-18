import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
from numpy.linalg import eig
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import mnist

# mnist.init()
image_dimension = 28
number_of_dimension = 14
number_of_classes = 10

x_train, t_train, x_test, t_test = mnist.load()

x_train = (x_train > 0) * 1
x_test = (x_test > 0) * 1

# Train
x_train_features = np.zeros([len(x_train), 2 * image_dimension])
for i in range(0, len(x_train)):
    image = np.reshape(x_train[i], [image_dimension, image_dimension])
    x_train_features[i, 0:image_dimension] = np.sum(image, axis=0)
    x_train_features[i, image_dimension:2 * image_dimension] = np.sum(image, axis=1)

Sw = np.zeros([2 * image_dimension, 2 * image_dimension])
Sb = np.zeros([2 * image_dimension, 2 * image_dimension])
p = np.zeros(number_of_classes)

m = np.mean(x_train_features, axis=0)
for i in range(0, number_of_classes):
    index = t_train == i
    data = x_train_features[index]
    p[i] = len(data) / len(x_train_features)
    mi = np.mean(data, axis=0)
    covMatrix = np.cov(data.T)
    Sw = Sw + p[i] * covMatrix
    Sb = Sb + p[i] * np.matmul((m - mi), np.transpose((mi - m)))

S = inv(Sw) * Sb
eigenValues, eigenVectors = eig(S)

idx = eigenValues.argsort()[::-1]
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:, idx]
transformationMatrix = np.transpose(np.real(eigenVectors[0:number_of_dimension, :]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(0, number_of_classes):
    index = t_train == i
    data = x_train_features[index]
    data = np.matmul(data, transformationMatrix)
    ax.scatter(data[0:100, 0], data[0:100, 1], data[0:100, 2])
plt.show()

m = np.zeros([number_of_dimension, number_of_classes])
covMat = np.zeros([number_of_dimension, number_of_dimension, number_of_classes])
for i in range(0, number_of_classes):
    index = t_train == i
    data = x_train_features[index]
    data = np.matmul(data, transformationMatrix)
    m[:, i] = np.mean(data, axis=0)
    covMat[:, :, i] = np.cov(data.T)

# Test
x_test_features = np.zeros([len(x_test), 2 * image_dimension])
for i in range(0, len(x_test)):
    image = np.reshape(x_test[i], [image_dimension, image_dimension])
    x_test_features[i, 0:image_dimension] = np.sum(image, axis=0)
    x_test_features[i, image_dimension:2 * image_dimension] = np.sum(image, axis=1)

acc = 0
correct_class = []
detect_class = []
for i in range(0, number_of_classes):
    index = t_test == i
    data = x_test_features[index]
    data = np.matmul(data, transformationMatrix)
    answer = np.zeros(len(data))
    for j in range(0, len(data)):
        correct_class.append(i)
        q = np.zeros(number_of_classes)
        for k in range(0, number_of_classes):
            point = data[j]
            q[k] = p[i]*1/((2 * np.pi)**(number_of_dimension / 2) * (det(covMat[:, :, k])) ** 0.5)*np.exp(-0.5 * np.matmul(np.matmul(point - m[:, k], inv(covMat[:, :, k])), np.transpose(point - m[:, k])))

        answer[j] = np.argmax(q)
        detect_class.append(answer[j])

conf = confusion_matrix(correct_class, detect_class)

acc = np.sum(np.diag(conf)) / np.sum(conf) * 100
print('Accuracy: ' + repr(np.round(acc, 2)) + '%')

sns.heatmap(conf/np.sum(conf, axis=1), annot=True, fmt='.2%', cmap='Blues')
plt.xlabel('Predicted class')
plt.ylabel('Correct class')
plt.show()