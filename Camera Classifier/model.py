from sklearn.svm import LinearSVC
import numpy as np
import cv2 as cv
import PIL


class Model:

    def __init__(self):
        self.model = LinearSVC()

    
    def train_model(self, counters):

        img_list = np.array([])
        class_list = np.array([])

        for i in range(1, counters[0]): # Iterate through images in class 1
            img = cv.imread(f'1/frame{i}.jpg')[:,:,0]
            img.reshape(150*113)

            img_list = np.append(img_list, [img]) # Add new image to list, shape (1, 150*113)
            class_list = np.append(class_list, 1) # Add corresponding class

        for i in range(1, counters[1]): # Iterate through images in class 2
            img = cv.imread(f'2/frame{i}.jpg')[:,:,0]
            img.reshape(150*113)

            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 2)

        img_list = img_list.reshape(counters[0] - 1 + counters[1] - 1, 150*113) # Reshape to 2D array for model training
        self.model.fit(img_list, class_list)

        print('Model succesefullly trained!')


    def predict(self, frame):
        
        frame = frame[1] # Get the image from the (ret, frame) tuple
        cv.imwrite('frame.jpg', cv.cvtColor(frame, cv.COLOR_RGB2GRAY))

        img = PIL.Image.open('frame.jpg')
        img.thumbnail((150,150), PIL.Image.Resampling.LANCZOS)
        img.save('frame.jpg') 

        img = cv.imread('frame.jpg')[:,:,0]
        img = img.reshape(150*113)


        prediction = self.model.predict([img])

        return prediction[0]

