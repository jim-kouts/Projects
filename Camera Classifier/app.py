import tkinter as tk
from tkinter import simpledialog # Importing simpledialog for input dialogs
import cv2 as cv
import os
import PIL.Image, PIL.ImageTk
import camera
import model

class App:
    def __init__(self, window= tk.Tk(), window_title='Camera Classifier'):
        self.window = window
        self.window.title(window_title)

        self.counters = [1,1] # Counters for naming saved images
        
        self.model = model.Model()

        self.auto_predict = False
        self.camera = camera.Camera() # Initialize camera object

        self.init_gui()

        self.delay = 15 # milliseconds

        self.update()

        self.window.attributes('-topmost', True)   # position window on top of others
        self.window.mainloop()

    
    def init_gui(self):
        self.canvas = tk.Canvas(self.window, width= self.camera.width, height= self.camera.height) # Use camera dimensions
        self.canvas.pack() # Pack the canvas to make it visible

        self.btn_toggleauto = tk.Button(self.window, text='Auto Prediction', width=50, command=self.auto_predict_toggle)
        self.btn_toggleauto.pack(anchor=tk.CENTER, expand=True)

        self.classname_one = simpledialog.askstring('Classname one', 'Enter the name of the first class:', parent=self.window) # Prompt for first class name , parent works to center the dialog
        self.classname_two = simpledialog.askstring('Classname two', 'Enter the name of the second class:', parent=self.window) # Prompt for second class name , parent works to center the dialog

        self.btn_class_one = tk.Button(self.window, text= self.classname_one, width=50, command= lambda: self.save_for_class(1))
        self.btn_class_one.pack(anchor=tk.CENTER, expand=True)

        self.btn_class_two = tk.Button(self.window, text= self.classname_two, width=50, command= lambda: self.save_for_class(2))
        self.btn_class_two.pack(anchor=tk.CENTER, expand=True)

        self.btn_train = tk.Button(self.window, text= 'Train model', width=50, command= lambda: self.model.train_model(self.counters))
        self.btn_train.pack(anchor=tk.CENTER, expand=True)

        self.btn_predict = tk.Button(self.window, text= 'Predict', width=50, command= lambda: self.predict())
        self.btn_predict.pack(anchor=tk.CENTER, expand=True)

        self.btn_reset = tk.Button(self.window, text= 'Reset', width=50, command= lambda: self.reset())
        self.btn_reset.pack(anchor=tk.CENTER, expand=True)

        self.class_label = tk.Label(self.window, text='CLASS')
        self.class_label.config(font=('Arial', 20))
        self.class_label.pack(anchor=tk.CENTER, expand= True)



    def auto_predict_toggle(self):
        self.auto_predict = not self.auto_predict
        # if self.auto_predict:
        #     self.btn_auto_predict.config(text='Auto Predict: ON') 
        # else:
        #     self.btn_auto_predict.config(text='Auto Predict: OFF')


    def save_for_class(self, class_num):
        ret, frame = self.camera.get_frame()

        if not os.path.exists('1'):
            os.mkdir('1')
        if not os.path.exists('2'):
            os.mkdir('2')

        img_path = f'{class_num}/frame{self.counters[class_num-1]}.jpg'
        cv.imwrite(img_path, cv.cvtColor(frame, cv.COLOR_RGB2GRAY)) # Save as grayscale
        img = PIL.Image.open(img_path) # Reopen to ensure proper format
        img.thumbnail((150,150), PIL.Image.Resampling.LANCZOS) # Create thumbnail, LANCZOS for better quality
        img.save(img_path)

        self.counters[class_num - 1] += 1
    

    def reset(self):
        for directory in ['1','2']:
            if os.path.exists(directory):
                for file in os.listdir(directory): # Iterate through files in directory
                    file_path = os.path.join(directory, file) # Get full file path

                    if os.path.isfile(file_path): # Check if it's a file
                        os.unlink(file_path) # Delete the file
        
        self.counters = [1,1]
        self.model = model.Model()
        self.class_label.config(text='CLASS')


    def update(self):
        if self.auto_predict:
            self.predict()
            pass

        ret, frame = self.camera.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame)) # Convert frame to PhotoImage, frame is an array
            self.canvas.create_image(0,0, image=self.photo, anchor=tk.NW) # Draw the image on the canvas, NW anchor to position at top-left corner
        
        self.window.after(self.delay, self.update) # Schedule the next update after delay milliseconds


    
    def predict(self):
        frame = self.camera.get_frame()
        prediction = self.model.predict(frame)

        if prediction == 1:
            self.class_label.config(text= self.classname_one)
            return self.classname_one
        if prediction == 2:
            self.class_label.config(text= self.classname_two)
            return self.classname_two
        else:
            self.class_label.config(text='(no prediction)')
        return None
