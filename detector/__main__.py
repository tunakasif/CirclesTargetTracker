from detector import Detector
import cv2

with Detector(0, (640, 360)) as detector:
    target = None
    while True:
        frame, target, center = detector.detect(target)
        if center:
            cv2.putText(frame, f'Center: {center:0.3f}', (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        else:
            cv2.putText(frame, f'Center: None', (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('Detected', frame)
    cv2.destroyAllWindows()
