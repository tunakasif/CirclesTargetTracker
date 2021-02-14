from typing import Any, Optional, Tuple
from enum import Enum, auto
from math import hypot
import cv2


class DetectorColor:
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    PURPLE = (255, 0, 255)
    CYAN = (255, 255, 0)


class TargetType(Enum):
    TARGET = auto()
    PEER = auto()


def detect_pentagons(contours: list) -> list:
    pentagon_contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(
            contour, 0.07 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if area > 50 and (4 < len(approx) < 7) and cv2.isContourConvex(approx):
            pentagon_contour_list.append(contour)
    pentagon_contour_list = sorted(pentagon_contour_list, key=cv2.contourArea)
    return pentagon_contour_list


def detect_circles(contours: list) -> list:
    circle_contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if area > 30 and len(approx) > 10 and cv2.isContourConvex(approx):
            circle_contour_list.append(contour)
    return circle_contour_list


def check_target_proximity(target_contours: list, count: int):
    center_x_values, center_y_values = [], []
    for contour in target_contours[:count]:
        M = cv2.moments(contour)
        center_x_values.append(int(M['m10'] / M['m00']))
        center_y_values.append(int(M['m01'] / M['m00']))

    max_x_proximity = max(center_x_values) - min(center_x_values)
    max_y_proximity = max(center_y_values) - min(center_y_values)
    max_proximity = hypot(max_x_proximity, max_y_proximity)
    return max_proximity


def get_target_bounding_rect(target_contours: list, target_count: int,
                             min_prox: int, max_prox: int) -> list:
    proximity = None
    if len(target_contours) > 0:
        proximity = check_target_proximity(target_contours, target_count)
    if len(target_contours) >= target_count:
        if min_prox <= proximity <= max_prox:
            big_rect = list(cv2.boundingRect(target_contours[0]))
            max_achieved_x = big_rect[0] + big_rect[2]
            max_achieved_y = big_rect[1] + big_rect[3]
            for c in target_contours:
                rect = cv2.boundingRect(c)
                max_achieved_x = max(max_achieved_x, rect[0] + rect[2])
                max_achieved_y = max(max_achieved_y, rect[1] + rect[3])
                big_rect[0] = min(big_rect[0], rect[0])
                big_rect[1] = min(big_rect[1], rect[1])
            big_rect[2] = max_achieved_x - big_rect[0]
            big_rect[3] = max_achieved_y - big_rect[1]
            return big_rect, proximity
    return None, proximity


def sentinel_mode(frame, target_count=3, target_scale=0.3,
                  min_proximity=30, max_proximity=300):
    # obtain edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 1.5)
    edges = cv2.Canny(blur, 100, 200)

    # obtain necessary contours from detected edges
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circle_contour_list = detect_circles(contours)
    pentagon_contour_list = detect_pentagons(contours)

    # obtain bounding rectangle for target
    # if target is not in the sight, obtain bounding rectangle for a peer
    circle_target_rect, circle_prox = get_target_bounding_rect(
        circle_contour_list, target_count, min_proximity, max_proximity)
    pentagon_target_rect, pentagon_prox = get_target_bounding_rect(
        pentagon_contour_list, target_count, min_proximity, max_proximity)

    if circle_target_rect:
        target_rect = circle_target_rect
        target_type = TargetType.TARGET
    elif pentagon_target_rect:
        target_rect = pentagon_target_rect
        target_type = TargetType.PEER
    else:
        target_rect = None
        target_type = None

    # draw contours of the detected shapes
    cv2.drawContours(frame, circle_contour_list,  -1, (0, 0, 255), 2)
    cv2.drawContours(frame, pentagon_contour_list,  -1, (0, 255, 0), 2)

    # obtain target rect for tracking
    if target_rect:
        frame_width_limit = int(frame.shape[1] * target_scale)
        frame_height_limit = int(frame.shape[0] * target_scale)

        # set a width limit to target_rect, and shift the target to center
        if target_rect[2] > frame_width_limit:
            target_rect[0] = target_rect[0] + \
                (target_rect[2] - frame_width_limit) // 2
            target_rect[2] = frame_width_limit

        # set a width limit to target_rect, and shift the target to center
        if target_rect[3] > frame_height_limit:
            target_rect[1] = target_rect[1] + \
                (target_rect[3] - frame_height_limit) // 2
            target_rect[3] = frame_height_limit

        cv2.rectangle(frame, (target_rect[0], target_rect[1]),
                      (target_rect[0] + target_rect[2],
                       target_rect[1] + target_rect[3]), (255, 0, 255), 2)
        return target_rect, target_type

    return None, None


def drawTrackingBox(frame, bbox, target_type: TargetType):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    target_color = DetectorColor.PURPLE if target_type == TargetType.TARGET else DetectorColor.CYAN
    cv2.putText(frame, 'Status:', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, DetectorColor.PURPLE, 2)
    cv2.putText(frame, 'Tracking', (100, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, DetectorColor.GREEN, 2)
    cv2.putText(frame, f'{target_type.name}', (200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, target_color, 2)
    cv2.rectangle(frame, (x, y), ((x + w), (y + h)), target_color, 3, 3)


def tracker_mode(frame, tracker, frame_size,
                 target_type: TargetType = TargetType.TARGET) -> Tuple[Optional[object],
                                                                       Optional[float],
                                                                       Optional[float]]:
    success, target = tracker.update(frame)
    if success:
        drawTrackingBox(frame, target, target_type)
        bounding_box = target.index.__self__
        center_location_x = bounding_box[0] + bounding_box[2] // 2
        center_location_y = bounding_box[1] + bounding_box[3] // 2

        quantized_x = center_location_x / (frame_size[0] / 2) - 1
        quantized_y = center_location_y / (frame_size[1] / 2) - 1
        quantized_x = min(1, max(-1, quantized_x))
        quantized_y = min(1, max(-1, quantized_y))
        return target, quantized_x, quantized_y
    else:
        return None, None, None


class Detector:

    def __init__(self, no: int, frame_size: Optional[Tuple[int, int]] = None, fps: int = 30):
        self._cap = cv2.VideoCapture(no)
        self._tracker = cv2.TrackerKCF_create()
        self._frame_size = frame_size
        if frame_size:
            self._scale = True
        else:
            self._scale = False
            _, frame = self._cap.read()
            self._frame_size = (frame.shape[0], frame.shape[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.cap.release()

    @property
    def scale(self):
        return self._scale

    @property
    def cap(self):
        return self._cap

    @property
    def frame_size(self):
        return self._frame_size

    @property
    def tracker(self):
        return self._tracker

    @tracker.setter
    def tracker(self, tracker):
        self._tracker = tracker

    def detect(self, target: Optional[Any] = None,
               target_type: Optional[Any] = None) -> Tuple[Any, Any, Optional[TargetType],
                                                           Optional[float], Optional[float]]:
        _, frame = self.cap.read()
        if self.scale:
            frame = cv2.resize(frame, self.frame_size, cv2.INTER_LINEAR)
        center_x, center_y = None, None
        if target:
            target, center_x, center_y = tracker_mode(
                frame, self.tracker, self.frame_size, target_type)
        else:
            target, target_type = sentinel_mode(frame)
            if target:
                self.tracker = cv2.TrackerKCF_create()
                self.tracker.init(frame, tuple(target))
        return frame, target, target_type, center_x, center_y
