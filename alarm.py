# Alarm module
# ملف التنبيه الصوتي
# يعتمد على مكتبة time و threading و platform
# يمكن تشغيل إنذار صوتي في خلفية البرنامج
# يمكن إيقاف الإنذار عند الحاجة
# يعمل الإنذار في حلقة منفصلة لتجنب تعطيل البرنامج الرئيسي
# يمكن تخصيص تردد ومدة الصوت في نظام ويندوز

import time
import threading
import platform

class Alarm:
    def __init__(self):
        self.active = False
        self.thread = None

    def _alarm_loop(self):
        while self.active:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(1000, 500)  # frequency, duration
            else:
                print("\a", end="", flush=True)  # simple beep
            time.sleep(0.5)

    def start(self):
        if self.active:
            return
        self.active = True
        self.thread = threading.Thread(target=self._alarm_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.active = False
