from djitellopy import Tello
import cv2
import numpy as np


def initialize_drone():
    drone = Tello()
    drone.connect()
    drone.for_back_velocity = 0
    drone.left_right_velocity = 0
    drone.up_down_velocity = 0
    drone.yaw_velocity = 0
    drone.speed = 0
    print(drone.get_battery())
    drone.streamoff()
    drone.streamon()
    return drone


def get_drone_video_frame(drone, width, height):
    frame = drone.get_frame_read()
    frame = frame.frame
    frame_image = cv2.resize(frame, (width, height))
    return frame_image


def drone_track_object(drone, detection_params, width, height, pid, yaw_p_error, y_p_error, fw_p_error):
    # yaw tracking
    yaw_error = detection_params[0][0] / 3 - width / 3 // 2
    yaw_speed = pid[0] * yaw_error + pid[1] * (yaw_error - yaw_p_error)
    yaw_speed = int(np.clip(yaw_speed, -100, 100))

    # Y axis tracking
    y_error = detection_params[0][1] / 3 - height / 3 // 2
    y_speed = pid[0] / 2 * y_error + pid[1] / 2 * (y_error - y_p_error)
    y_speed = y_speed * (-1)
    y_speed = int(np.clip(y_speed, -30, 30))

    # forward / backward tracking
    max_size = (width * height) // 7
    threshold = (width * height) // 40
    fw_error = detection_params[1] - max_size
    if fw_error > threshold:
        # Move back
        move = True
        fw_error = fw_error / ((width * height) // 3) * 45
    elif (threshold * (-1)) > fw_error > max_size * (-1):
        # Move forward
        move = True
        fw_error = (fw_error / max_size) * 40
    else:
        # Don't move
        move = False
    if move:
        fw_speed = pid[2] * fw_error + pid[2] * 0.6 * (fw_error - fw_p_error)
        fw_speed = int(np.clip(fw_speed, -50, 20))
        fw_speed = fw_speed * (-1)
    else:
        fw_speed = 0

    # print("X axis: " + str(yaw_speed) + "  |   Y axis: " + str(y_speed) + "  |   Forward: " + str(fw_speed) + "  Move: " + str(move))

    if detection_params[0][0] != 0:
        drone.yaw_velocity = yaw_speed
        drone.up_down_velocity = y_speed
        drone.for_back_velocity = fw_speed
    else:
        drone.for_back_velocity = 0
        drone.left_right_velocity = 0
        drone.up_down_velocity = 0
        drone.yaw_velocity = 0
        yaw_error = 0

    if drone.send_rc_control:
        drone.send_rc_control(drone.left_right_velocity,
                              drone.for_back_velocity,
                              drone.up_down_velocity,
                              drone.yaw_velocity)
    return yaw_error, y_error, fw_error


def get_quality_calculations(detection_data, b_box_data):
    frame_count = len(detection_data)
    avg_detection_confidence = sum(detection_data) / frame_count * 100
    empty_frames = detection_data.count(0) / frame_count * 100

    filtered_b_box_data = [i for i in b_box_data if i != 0]
    if len(filtered_b_box_data) == 0:
        filtered_b_box_data.append(1)
    biggest_box = max(filtered_b_box_data)
    smallest_box = min(filtered_b_box_data)
    average_box = sum(filtered_b_box_data) / len(filtered_b_box_data)
    box_deviation = (((biggest_box / average_box) - 1) + (1 - (smallest_box / average_box))) / 2 * 100
    return avg_detection_confidence, empty_frames, box_deviation
