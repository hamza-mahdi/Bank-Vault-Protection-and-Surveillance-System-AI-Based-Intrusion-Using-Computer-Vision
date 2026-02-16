# Object tracking module using CSRT
# ملف التتبع باستخدام CSRT
# يعتمد على مكتبة OpenCV
# يستخدم خوارزمية CSRT لتتبع الأجسام في الفيديو
# يمكن بدء التتبع على إطار معين مع صندوق تحديد
# يمكن تحديث موقع الصندوق في الإطارات التالية

import cv2


class ObjectTracker:
    def __init__(self):
        self.tracker = None
        self.active = False

    def start(self, frame, bbox):
        """
        Initialize tracker with first bounding box
        bbox: (x, y, w, h)
        """
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox)
        self.active = True

    def update(self, frame):
        """
        Update tracker position
        Returns updated bbox or None if tracking failed
        """
        if not self.active:
            return None

        success, bbox = self.tracker.update(frame)

        if success:
            return bbox
        else:
            self.active = False
            return None

    def stop(self):
        self.tracker = None
        self.active = False
