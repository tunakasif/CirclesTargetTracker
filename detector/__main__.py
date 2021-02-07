from datetime import time
from detector import Detector, TargetType
import cv2
from imutils.video import FPS

with Detector(0, None) as detector:
    target, target_type = None, None
    while True:
        fps = FPS().start()
        frame, target, target_type, center_x, center_y = detector.detect(
            target, target_type)
        if center_x and center_y:
            cv2.putText(frame, f'Center: ({center_x:0.3f}, {center_y:0.3f})', (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        else:
            cv2.putText(frame, f'Center: None', (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        if cv2.waitKey(1) == 27:
            break
        fps.update()
        fps.stop()
        cv2.putText(frame, f'FPS: {fps.fps():0.0f}', (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.imshow('Detected', frame)
    cv2.destroyAllWindows()
