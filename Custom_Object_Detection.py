from ultralytics import YOLO
import cv2
import os

os.chdir("D:\\Work\\Python 3\\Workspace\\My Programs\\Pool Shot Predictor");

class YOLO_Custom_Object_Detection():
    def __init__(self):
        self.class_names = None;
        self.model = None;
        pass;

    def train_model(self, model_file = None, datapath_file = None, epochs_val = 1):
        if (model_file is not None and datapath_file is not None):
            if (not model_file.endswith(".yaml") and not model_file.endswith(".yml")):
                raise Exception("Invalid Model File Type: Model File must have an extension of .yaml or .yml");
            if (not datapath_file.endswith(".yaml") and not datapath_file.endswith(".yml")):
                raise Exception("Invalid Datapath File Type: Datapath File must have an extension of .yaml or .yml");
            else:
                model = YOLO(model_file);
                train_results = model.train(data = datapath_file,
                                            epochs = epochs_val);
        else:
            if (model_file is None):
                raise ValueError("String Type Expected: Found NoneType for Model File name");
            if (datapath_file is None):
                raise ValueError("String Type Expected: Found NoneType for Datapath File name");

    def initialize_model(self, trained_model_path = None, class_details = None):
        if (trained_model_path is None):
            raise ValueError("String Type Expected: Found NoneType for Trained Model Path");
        if (class_details is None):
            raise ValueError("'dict' type Expected: Found NoneType for Class Details");
        else:
            self.class_names = class_details;
            self.model = YOLO(trained_model_path);

    def detect(self, image = None, confidence_threshold = 0.5, draw = True):
        if (image is None):
            raise ValueError("numpy.ndarray Expected: Found NoneType for Image");
        else:
            results = self.model(image)[0];
            if (draw):
                for result in results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = result;

                    if score > confidence_threshold:
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2);
                        cv2.putText(image, self.class_names[int(class_id)].upper(), (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA);
            return results;
                
                
                
        
