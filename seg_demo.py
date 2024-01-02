from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-seg.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model.predict('./static/uploads/food.jpg',save=True)  # return a list of Results objects

# # Process results list
# for result in results:
#     #boxes = result.boxes  # Boxes object for bbox outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     #keypoints = result.keypoints  # Keypoints object for pose outputs
#     #probs = result.probs  # Probs object for classification outputs
#     # print('boxes:',boxes)
#     print('type(masks):',type(masks))
#     print('masks:',masks.numpy)
#
#     # print('keypoints:',keypoints)
#     # print('probs:',probs)