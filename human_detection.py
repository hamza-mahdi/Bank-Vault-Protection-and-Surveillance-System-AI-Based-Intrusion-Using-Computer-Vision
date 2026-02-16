# Human detection module using HOG + SVM
# ملف كشف الإنسان باستخدام HOG
# يعتمد على مكتبة OpenCV
# يستخدم كاشف الأشخاص المدرب مسبقًا في OpenCV
# يمكنه الكشف عن وجود إنسان في منطقة معينة من الصورة (ROI) لكن لم يتم استخدام الروي في المشروع بسبب المشاكل التي يسببها
# تم استخدام مودل جاهز HOG من OpenCV لكشف الأشخاص   
# يمكنه إعادة قيمة منطقية تشير إلى وجود إنسان أو عدم وجوده

import cv2


class HumanDetector:
    def __init__(self):
        """
        Initialize HOG person detector (pre-trained in OpenCV)
        """
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(
            cv2.HOGDescriptor_getDefaultPeopleDetector()
        )

    def detect(self, frame_roi):
        """
        Detect human in a given ROI
        :param frame_roi: cropped image (ROI)
        :return: True if human detected, False otherwise
        """

        if frame_roi is None or frame_roi.size == 0:
            return False

        # Resize for more stable detection (optional but recommended)
        roi_resized = cv2.resize(frame_roi, None, fx=1.0, fy=1.0)

        humans, _ = self.hog.detectMultiScale(
            roi_resized,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )

        if len(humans) > 0:
            return True

        return False
