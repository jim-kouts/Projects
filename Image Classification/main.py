import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models



(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255.0, testing_images / 255.0

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

'Display first 16 images from the training set'
# for i in range(16):
#     plt.subplot(4, 4, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(training_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[training_labels[i][0]])
# plt.show()



training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:20000]
training_labels = training_labels[:20000]
# evauation_images = testing_images[10000:25000]
# evaluation_labels = testing_labels[10000:25000]


'Build the CNN model and save it'
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3,3), activation= 'relu', input_shape= (32, 32, 3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation= 'relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation= 'relu'))
# model.add(layers.Flatten())

# model.add(layers.Dense(64, activation= 'relu'))
# model.add(layers.Dense(10, activation= 'softmax'))

# model.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy', metrics= ['accuracy'])
# model.fit(training_images, training_labels, epochs=10, validation_data= (testing_images, testing_labels))

# loss, accuracy = model.evaluate(testing_images, testing_labels)
# print(f'Test accuracy: {accuracy}')
# print(f'Test loss: {loss}')

# model.save('image_classifier.h5')


model = models.load_model('image_classifier.h5')

'Make predictions'
predictions = model.predict(testing_images)
print(f'Predictions shape: {predictions.shape}')

max_index = np.argmax(predictions, axis=1)
print('max_index shape: ', max_index.shape)

for i in range(20):
    print(f'predicted label: {class_names[max_index[i]]}, actual label: {class_names[testing_labels[i][0]]}')


'Display first 20 images from the testing set with predicted labels'

plt.figure(figsize=(12,12))
for i in range(20):
    plt.subplot(5, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(testing_images[i], cmap=plt.cm.binary)
    plt.xlabel(f'Predicted: {class_names[max_index[i]]}\nActual: {class_names[testing_labels[i][0]]}' 
        , fontsize=9, labelpad=8, ha='center')


# adjust spacing between subplots
plt.subplots_adjust(hspace=0.6, wspace=0.3)
plt.show()
