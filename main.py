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


def getFeatures(data):
    features = np.zeros([len(data), 2 * image_dimension])
    for i in range(0, len(data)):
        image = np.reshape(data[i], [image_dimension, image_dimension])
        features[i, 0:image_dimension] = np.sum(image, axis=0)
        features[i, image_dimension:2 * image_dimension] = np.sum(image, axis=1)
    return features


def getTransformationMatrix(features):
    Sw = np.zeros([2 * image_dimension, 2 * image_dimension])
    Sb = np.zeros([2 * image_dimension, 2 * image_dimension])
    p = np.zeros(number_of_classes)

    m = np.mean(features, axis=0)
    for i in range(0, number_of_classes):
        index = t_train == i
        data = features[index]
        p[i] = len(data) / len(features)
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
    return p, transformationMatrix


def visualiseData(features, target, N):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(0, number_of_classes):
        index = target == i
        data = features[index]
        data = np.matmul(data, transformationMatrix)
        ax.scatter(data[0:N, 0], data[0:N, 1], data[0:N, 2])
    plt.show()


def getClassParameters(features, target):
    m = np.zeros([number_of_dimension, number_of_classes])
    covMat = np.zeros([number_of_dimension, number_of_dimension, number_of_classes])
    for i in range(0, number_of_classes):
        index = t_train == i
        data = x_train_features[index]
        data = np.matmul(data, transformationMatrix)
        m[:, i] = np.mean(data, axis=0)
        covMat[:, :, i] = np.cov(data.T)
    return m, covMat


def getClassificationResult(features, transformationMatrix, p, plotConfustion):
    correct_class = []
    detect_class = []
    for i in range(0, len(features)):
        data = features[i]
        data = np.matmul(data, transformationMatrix)
        correct_class.append(t_test[i])
        q = np.zeros(number_of_classes)
        for k in range(0, number_of_classes):
            q[k] = p[k] * 1 / ((2 * np.pi) ** (number_of_dimension / 2) * (det(covMat[:, :, k])) ** 0.5) * np.exp(
                -0.5 * np.matmul(
                    np.matmul(data - m[:, k], inv(covMat[:, :, k])), np.transpose(data - m[:, k])))
        answer = np.argmax(q)
        detect_class.append(answer)

    if plotConfustion:
        conf = confusion_matrix(correct_class, detect_class)

        acc = np.sum(np.diag(conf)) / np.sum(conf) * 100
        print('Accuracy: ' + repr(np.round(acc, 2)) + '%')

        sns.heatmap(conf / np.sum(conf, axis=1), annot=True, fmt='.2%', cmap='Blues')
        plt.xlabel('Predicted class')
        plt.ylabel('Correct class')
        plt.show()

    return detect_class


def displayClassificationResult(data, detect_class, images_per_row_col):
    image = np.zeros([images_per_row_col * image_dimension, images_per_row_col * image_dimension])
    output = np.zeros([images_per_row_col, images_per_row_col])
    for i in range(0, len(data)):
        img = np.reshape(data[i], [image_dimension, image_dimension])
        indexX = int((i % images_per_row_col ** 2) / images_per_row_col)
        indexY = i % images_per_row_col ** 2 - indexX * images_per_row_col
        image[indexX * image_dimension: (indexX + 1) * image_dimension, indexY * image_dimension: (indexY + 1) *
                                                                                                  image_dimension] = img
        if i % images_per_row_col ** 2 == 0 and i > 0:
            out = detect_class[i - images_per_row_col ** 2: i]
            output = np.reshape(out, [images_per_row_col, images_per_row_col])
            plt.figure(int(i / images_per_row_col ** 2))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.subplot(1, 2, 2)
            sns.heatmap(output, annot=True, fmt="d", cmap='Blues')
            image = np.zeros([images_per_row_col * image_dimension, images_per_row_col * image_dimension])
            output = np.zeros([images_per_row_col, images_per_row_col])

    plt.show()


# Load and binarization
x_train, t_train, x_test, t_test = mnist.load()
x_train = (x_train > 0) * 1
x_test = (x_test > 0) * 1

# Train
x_train_features = getFeatures(x_train)
p, transformationMatrix = getTransformationMatrix(x_train_features)

visualiseData(x_train_features, t_train, 100)
m, covMat = getClassParameters(x_train_features, t_train)

# Test
x_test_features = getFeatures(x_test)
detect_class = getClassificationResult(x_test_features, transformationMatrix, p, True)
displayClassificationResult(x_test, detect_class, 25)
