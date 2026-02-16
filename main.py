# Main application for Bank Vault Intrusion Detection System
# البرنامج الرئيسي للنظام
# يحتوي على كشف الحركه ,تنشيط التتبع , تنشيط الانذار عند اكتشاف انسان
# يستخدم تقنيات معالجة الصور والرؤية الحاسوبية
#يستدعي مودل كشف الانسان من ملف human_detection.py
#يستخدم مودل التتبع من ملف tracking.py
#يستخدم مودل الانذار من ملف alarm.py

import cv2
import numpy as np
import time

from tracking import ObjectTracker
from human_detection import HumanDetector
from alarm import Alarm

import os
import psutil


SNAPSHOT_DIR = "snapshots"# Directory to save snapshots of detected intruders
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
# Set process to high priority to reduce latency
#تعيين أولوية العملية إلى عالية لتقليل الكمون
def set_high_priority():
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)
    

def save_snapshot(frame, roi=None):
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{SNAPSHOT_DIR}/intruder_{timestamp}.jpg"

    if roi is not None and roi.size > 0:
        cv2.imwrite(filename, roi)
    else:
        cv2.imwrite(filename, frame)

    print(f"📸 Snapshot saved: {filename}")


def preprocess_frame(frame):
    """
    Convert frame to grayscale and apply Gaussian blur
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # blur = cv2.medianBlur(gray, 5)
    return blur


def detect_motion(background, current_frame):
    """
    Detect motion using frame differencing
    Returns binary mask of motion areas
    """
    diff = cv2.absdiff(background, current_frame)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    return thresh


def main():
    
    set_high_priority()
    # Open video file
    # cap = cv2.VideoCapture("videos/house_theft2.mp4", cv2.CAP_FFMPEG)
    cap = cv2.VideoCapture(0)

    background = None
    
    alarm_active = False
    hog_enabled = True
    is_human = False
    
    frame_count = 0
    DRAW_EVERY = 5
    last_bbox = None
    
    snapshot_taken = False

    
    alarm_start_time = 0
    ALARM_DURATION = 30  # seconds
    
    human_detector = HumanDetector()
    tracker = ObjectTracker()
    alarm = Alarm()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame not received, continuing...")
            time.sleep(0.3)
            continue

        # Preprocess frame
        processed = preprocess_frame(frame)
        # عداد الفريمات
        frame_count += 1
        
        motion_detected = False

        # Set background (first frame only)
        if background is None:
            background = processed.astype("float")
            continue
        
        cv2.accumulateWeighted(processed, background, 0.01)
        background_uint8 = cv2.convertScaleAbs(background)
        if not tracker.active:
    # motion detection هنا
     
        # Detect motion
            motion_mask = detect_motion(background_uint8, processed)

        # Find contours
            contours, _ = cv2.findContours(
                motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            motion_detected = False

            for contour in contours:
            
                MIN_AREA = frame.shape[0] * frame.shape[1] * 0.005  # 0.5% of frame area
                if cv2.contourArea(contour) < MIN_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(contour)

            # Check motion inside ROI


                motion_detected = True
                

        # Draw unified bounding box and alert text
        if motion_detected:
            x_min, y_min = x, y
            x_max, y_max = x + w, y + h
            
            roi_person = frame[y_min:y_max, x_min:x_max]
            
            if roi_person is None or roi_person.size == 0:
                is_human = False
            else:
                h, w = roi_person.shape[:2]
                if h < 60 or w < 40:
                    is_human = False
                else:
                    try:
                        h, w = roi_person.shape[:2]
                        scale = 512 / h
                        new_w = int(w * scale)
                        roi_person = cv2.resize(roi_person, (new_w, 512))
                        
                    except Exception as e:
                        print(f"Error resizing ROI: {e}")
                        is_human = False
                    try:
                        is_human = human_detector.detect(roi_person)
                    except Exception as e:
                        print(f"Error in human detection: {e}")
                        is_human = False

            if not hog_enabled:
                is_human = False
                
            if is_human and not tracker.active:
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                tracker.start(frame, bbox)

                alarm_active = True
                alarm_start_time = time.time()
                hog_enabled = False
                alarm.start()
                
                if not snapshot_taken:
                    save_snapshot(frame, roi_person)
                    snapshot_taken = True
                
            
    
            if is_human:
                cv2.rectangle(
                    frame,
                    (x_min, y_min),
                    (x_max, y_max),
                    (0, 0, 255),
                    2
                )
                cv2.putText(
                    frame,
                    "HUMAN DETECTED",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                
            # else:
            #     cv2.rectangle(
            #         frame,
            #         (x_min, y_min),
            #         (x_max, y_max),
            #         (0, 255, 0),
            #         2
            #     )
                
        if tracker.active:
            tracked_bbox = tracker.update(frame)
            
            if tracked_bbox is not None:
                
                if frame_count % DRAW_EVERY == 0:
                    last_bbox = tuple(map(int, tracked_bbox))
                    
                if last_bbox is not None:
                    x, y, w, h = last_bbox    
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 0, 255),
                        2
                    )
                    cv2.putText(
                        frame,
                        "TRACKING HUMAN",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
            else:
                tracker.stop()
                hog_enabled = True
                is_human = False
                snapshot_taken = False
                
        if alarm_active:
            if time.time() - alarm_start_time >= ALARM_DURATION:
                alarm_active = False
                hog_enabled = True
                is_human = False
                tracker.stop()
                alarm.stop()
                snapshot_taken = False


                  
        # Display results
        cv2.imshow("Original Frame", frame)
        # cv2.imshow("Motion Mask", motion_mask)

        key = cv2.waitKey(50) & 0xFF
        if key == 27:
            break
        
        if key == ord('q'):
            alarm_active = False
            hog_enabled = True
            is_human = False
            tracker.stop()
            alarm.stop()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
