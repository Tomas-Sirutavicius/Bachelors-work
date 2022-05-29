from torchvision.models import detection
import numpy as np
import torch
import cv2


def initialize_resnet(backbone):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = {3: "car", 8: "truck"}
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    models = {
        "frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
    }
    model = models[backbone](pretrained=True, progress=True, pretrained_backbone=True)
    return device, classes, colors, model


def calculate_detections(image, detections):
    detection_centers = []
    detection_areas = []
    detection_data = []
    xy_staring_coordinates = []
    xy_ending_coordinates = []

    for i in range(0, len(detections["boxes"])):
        confidence = detections["scores"][i].detach().cpu().numpy().astype("float")
        if confidence > 0.50:
            idx = int(detections["labels"][i])
            if not (idx == 3 or idx == 8):
                continue
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")

            xy_staring_coordinates.append([startX, startY])
            xy_ending_coordinates.append([endX, endY])
            centerX = (endX - startX) // 2 + startX
            centerY = (endY - startY) // 2 + startY
            area = (endX - startX) * (endY - startY)
            detection_centers.append([centerX, centerY])
            detection_data.append(confidence)
            detection_areas.append(area)

    if len(detection_areas) != 0:
        index = detection_areas.index(max(detection_areas))
        label = "{}: {:.2f}%".format("car", detections["scores"][index] * 100)
        y = xy_staring_coordinates[index][1] - 15 if xy_staring_coordinates[index][1] - 15 > 15 else xy_staring_coordinates[index][1] + 15

        image_with_b_box = cv2.rectangle(image, (xy_staring_coordinates[index][0], xy_staring_coordinates[index][1]),(xy_ending_coordinates[index][0], xy_ending_coordinates[index][1]), (0, 0, 255), 2)
        image_with_label = cv2.putText(image_with_b_box, label, (xy_staring_coordinates[index][0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return image_with_label, [detection_centers[index], detection_areas[index]], detection_data[index]
    else:
        return image, [[0, 0], 0], 0
