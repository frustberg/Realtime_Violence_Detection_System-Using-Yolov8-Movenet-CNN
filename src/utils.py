import pygame, threading, time
alert_sound = "alarm_sound.mp3"
def init_audio():
    try:
        pygame.mixer.init()
    except Exception as e:
        print(f"Audio init error: {e}")
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
