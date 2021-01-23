from typing import Any, Optional, Tuple
import cv2


def detect_circles(contours: list) -> list:
    circle_contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if area > 30 and len(approx) > 10 and cv2.isContourConvex(approx):
            circle_contour_list.append(contour)
    return circle_contour_list


def get_target_bounding_rect(target_contours: list, target_count: int) -> list:
    if len(target_contours) >= target_count:
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
        return big_rect


def sentinel_mode(frame, headless=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 1.5)
    edges = cv2.Canny(blur, 100, 200)

    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circle_contour_list = detect_circles(contours)
    target_rect = get_target_bounding_rect(circle_contour_list, 3)

    cv2.drawContours(frame, circle_contour_list,  -1, (0, 0, 255), 2)
    if target_rect:
        cv2.rectangle(frame, (target_rect[0], target_rect[1]),
                      (target_rect[0] + target_rect[2],
                       target_rect[1] + target_rect[3]), (255, 0, 255), 2)
        return target_rect

    return None


def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 3)
    cv2.putText(img, 'Tracking', (100, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def tracker_mode(frame, tracker, frame_size) -> Tuple[Optional[object], Optional[float], Optional[float]]:
    success, target = tracker.update(frame)
    if success:
        drawBox(frame, target)
        bounding_box = target.index.__self__
        center_location_x = bounding_box[0] + bounding_box[2] // 2
        center_location_y = bounding_box[1] + bounding_box[3] // 2

        quantized_x = center_location_x / (frame_size[0] / 2) - 1
        quantized_y = center_location_y / (frame_size[1] / 2) - 1
        quantized_x = min(1, max(-1, quantized_x))
        quantized_y = min(1, max(-1, quantized_y))

        cv2.putText(frame, 'Status:', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
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

    def detect(self, target: Optional[Any] = None) -> Tuple[Any, Any, Optional[float], Optional[float]]:
        _, frame = self.cap.read()
        if self.scale:
            frame = cv2.resize(frame, self.frame_size, cv2.INTER_LINEAR)
        center_x, center_y = None, None
        if target:
            target, center_x, center_y = tracker_mode(
                frame, self.tracker, self.frame_size)
        else:
            target = sentinel_mode(frame)
            if target:
                self.tracker = cv2.TrackerKCF_create()
                self.tracker.init(frame, tuple(target))
        return frame, target, center_x, center_y
