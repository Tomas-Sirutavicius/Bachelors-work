from YOLOUtilities import *
from tello.droneUtilities import *
import cv2
from imutils.video import FPS

# set the device
(DEVICE, CLASSES, COLORS, model) = initialize_yolo()

width = 960
height = 720
pid = [0.5, 0.5, 0.7]
yaw_p_error = 0
y_p_error = 0
fw_p_error = 0

startCounter = 0  # 0 for flight

drone = initialize_drone()

print("[INFO] starting video stream...")
fps = FPS().start()
detection_data = []
b_box_data = []

while True:
    frame = get_drone_video_frame(drone, width, height)
    if frame is None:
        break

    ## Flight
    if startCounter == 0:
        drone.takeoff()
        startCounter = 1

    results = model(frame)
    (b_box_image, detection_params, confidence) = calculate_detections(frame, results)
    detection_data.append(confidence)
    b_box_data.append(detection_params[1])

    cv2.imshow('YOLO drone footage', b_box_image)

    (yaw_p_error, y_p_error, fw_p_error) = drone_track_object(drone, detection_params, width, height, pid, yaw_p_error, y_p_error, fw_p_error)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        drone.streamoff()
        fps.stop()
        drone.land()
        drone.end()
        break
    fps.update()

(avg_detection_confidence, empty_frames, box_deviation) = get_quality_calculations(detection_data, b_box_data)
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("[DATA] average detection confidence: {:.2f}".format(avg_detection_confidence))
print("[DATA] empty frame percentage: {:.2f}".format(empty_frames))
print("[DATA] box deviation percentage: {:.2f}".format(box_deviation))
cv2.destroyAllWindows()
drone.end()
