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


def sentinel_mode(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (9, 9), 1.5)
    ret3, th3 = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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

    cv2.imshow('edges', edges)
    return None


def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 3)
    cv2.putText(img, "Tracking", (100, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def tracker_mode(frame, tracker):
    success, target = tracker.update(frame)
    if success:
        drawBox(frame, target)
        cv2.putText(frame, 'Status:', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        return target
    else:
        return None


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    tracker = cv2.TrackerKCF_create()
    target = None

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 360), cv2.INTER_LINEAR)
        if target:
            target = tracker_mode(frame, tracker)
        else:
            target = sentinel_mode(frame)
            if target:
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, tuple(target))

        cv2.imshow('frame', frame)
        if cv2.waitKey(10) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
