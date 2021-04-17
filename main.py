import numpy as np
import matplotlib.pyplot as plt


# A function to plot images
def show_image(img):
    image = img.reshape((28, 28))
    plt.imshow(image, 'gray')


# The sigmoid function
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


# A function for feedforward calculation
def feedforward(x, w, a, b):
    a[0] = sigmoid(np.add(np.matmul(w[0], x), b[0]))
    for i in range(1, 3):
        a[i] = sigmoid(np.add(np.matmul(w[i], a[i - 1]), b[i]))
    return a


# Reading The Train Set
train_images_file = open('train-images.idx3-ubyte', 'rb')
train_images_file.seek(4)
num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
train_images_file.seek(16)

train_labels_file = open('train-labels.idx1-ubyte', 'rb')
train_labels_file.seek(8)

train_set = []
for n in range(num_of_train_images):
    image = np.zeros((784, 1))
    for i in range(784):
        image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256

    label_value = int.from_bytes(train_labels_file.read(1), 'big')
    label = np.zeros((10, 1))
    label[label_value, 0] = 1

    train_set.append((image, label))

# Reading The Test Set
test_images_file = open('t10k-images.idx3-ubyte', 'rb')
test_images_file.seek(4)

test_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
test_labels_file.seek(8)

num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
test_images_file.seek(16)

test_set = []
for n in range(num_of_test_images):
    image = np.zeros((784, 1))
    for i in range(784):
        image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256

    label_value = int.from_bytes(test_labels_file.read(1), 'big')
    label = np.zeros((10, 1))
    label[label_value, 0] = 1

    test_set.append((image, label))

# # Plotting an image
# show_image(train_set[0][0])
# plt.show()
# # Print label of image
# print(np.argmax(train_set[0][1]))

# Initialize weights with standard normal distribution
w = [np.random.standard_normal(size=(16, 784)), np.random.standard_normal(size=(16, 16)),
     np.random.standard_normal(size=(10, 16))]

# Initialize bias with zero matrix
b = [np.zeros((16, 1)), np.zeros((16, 1)), np.zeros((10, 1))]

# Initialize activations with zero matrix
a = [np.zeros((16, 1)), np.zeros((16, 1)), np.zeros((10, 1))]


hit = 0
for i in range(100):
    a = feedforward(train_set[i][0], w, a, b)
    result = np.argmax(a[2])
    label = np.argmax(train_set[i][1])
    # print(result, label)
    if result == label:
        hit += 1

print("Accuracy = ", hit/100)









