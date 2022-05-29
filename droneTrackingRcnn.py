from imutils.video import FPS
from tello.droneUtilities import *
from FRCNNUtilities import *

width = 960
height = 720
pid = [0.5, 0.5, 0.7]
yaw_p_error = 0
y_p_error = 0
fw_p_error = 0

startCounter = 0  # 0 for flight

drone = initialize_drone()

# load the model
(DEVICE, CLASSES, COLORS, model) = initialize_resnet("frcnn-resnet")

#  set it to evaluation mode
model.to(DEVICE)
model.eval()

print("[INFO] starting video stream...")
fps = FPS().start()
detection_data = []
b_box_data = []

# loop over the frames from the video stream
while True:
    frame_read = drone.get_frame_read()
    if frame_read.frame is None:
        break

    ## Flight
    if startCounter == 0:
        drone.takeoff()
        startCounter = 1

    myFrame = frame_read.frame
    frame = cv2.resize(myFrame, (width, height))

    orig = frame.copy()
    # convert the frame from BGR to RGB channel ordering and change

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2, 0, 1))

    # add a batch dimension, scale the raw pixel intensities to the
    # range [0, 1], and convert the frame to a floating point tensor
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0
    frame = torch.FloatTensor(frame)
    frame = frame.to(DEVICE)
    detections = model(frame)[0]

    (b_box_image, detection_params, confidence) = calculate_detections(orig, detections)
    detection_data.append(confidence)
    b_box_data.append(detection_params[1])

    (yaw_p_error, y_p_error, fw_p_error) = drone_track_object(drone, detection_params, width, height, pid, yaw_p_error, y_p_error, fw_p_error)

    # show the output frame
    cv2.imshow("FRCNN", b_box_image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        drone.streamoff()
        fps.stop()
        drone.land()
        drone.end()
        break
    # update the FPS counter
    fps.update()

(avg_detection_confidence, empty_frames, box_deviation) = get_quality_calculations(detection_data, b_box_data)
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("[DATA] average detection confidence: {:.2f}".format(avg_detection_confidence))
print("[DATA] empty frame percentage: {:.2f}".format(empty_frames))
print("[DATA] box deviation percentage: {:.2f}".format(box_deviation))
cv2.destroyAllWindows()
drone.end()
