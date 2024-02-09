from ultralytics import YOLO


# load a pretrained yolov8n model 
model = YOLO("yolov8n.pt") 


# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
 
# Run inference on the source 
results = model(source='traffic.mp4',show=True, conf = 0.4, save = True)  # generator of result objects 

results = model(source=0,show=True, conf = 0.4, save = True)  # generator of result objects 