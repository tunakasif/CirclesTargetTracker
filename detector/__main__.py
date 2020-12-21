from detector import Detector
import cv2

with Detector(0, None) as detector:
    target = None
    while True:
        frame, target, center_x, center_y = detector.detect(target)
        if center_x and center_y:
            cv2.putText(frame, f'Center: ({center_x:0.3f}, {center_y:0.3f})', (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        else:
            cv2.putText(frame, f'Center: None', (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('Detected', frame)
    cv2.destroyAllWindows()
