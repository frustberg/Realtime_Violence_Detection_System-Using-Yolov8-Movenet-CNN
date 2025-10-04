import pygame, threading, time
import os
from playsound import playsound

alert_sound = "alarm_sound.mp3"
def init_audio():
    try:
        pygame.mixer.init()
    except Exception as e:
        print(f"Audio init error: {e}")

def play_alarm_sound():
    try:
        # Construct path relative to project root
        alarm_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'alarm_sound.mp3')
        alarm_path = os.path.abspath(alarm_path)

        if os.path.exists(alarm_path):
            playsound(alarm_path)
        else:
            print(f"[Warning] Alarm sound not found at: {alarm_path}")
    except Exception as e:
        print(f"[Error] Unable to play alarm sound: {e}")


def play_alert():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(alert_sound)
        pygame.mixer.music.play(-1)
    except Exception as e:
        print(f"Error playing alert sound: {e}")
        print("\a")
def stop_alert():
    try:
        pygame.mixer.music.stop()
    except Exception:
        pass
alert_active = False
def update_alert_status(detected):
    global alert_active
    if detected and not alert_active:
        alert_active = True
        threading.Thread(target=play_alert, daemon=True).start()
    elif not detected and alert_active:
        alert_active = False
        stop_alert()
