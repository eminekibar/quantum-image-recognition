"""
GPIO Kontrolcusu — Raspberry Pi
---------------------------------
LED'ler, buton ve buzzer'i yonetir.
Projeyle entegrasyon: predict.py bu dosyayi import eder.

Devre baglantilari:
  LED 0  -> GPIO 4  (Fiziksel Pin 7)
  LED 1  -> GPIO 17 (Fiziksel Pin 11)
  LED 2  -> GPIO 27 (Fiziksel Pin 13)
  LED 3  -> GPIO 22 (Fiziksel Pin 15)
  LED 4  -> GPIO 5  (Fiziksel Pin 29)
  LED 5  -> GPIO 6  (Fiziksel Pin 31)
  LED 6  -> GPIO 13 (Fiziksel Pin 33)
  LED 7  -> GPIO 19 (Fiziksel Pin 35)
  LED 8  -> GPIO 26 (Fiziksel Pin 37)
  LED 9  -> GPIO 21 (Fiziksel Pin 40)
  BUTON  -> GPIO 12 (Fiziksel Pin 32) + GND
  BUZZER -> GPIO 16 (Fiziksel Pin 36) -> BC547 transistor -> Buzzer -> 5V
"""

import time

# RPi.GPIO sadece Raspberry Pi'de var; PC'de import hatasi vermemesi icin
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("UYARI: RPi.GPIO bulunamadi. GPIO ozellikleri devre disi.")

# ── PIN TANIMLARI (BCM numaralari) ────────────────────────────────────────
LED_PINS = {
    0: 4,    # Fiziksel Pin 7
    1: 17,   # Fiziksel Pin 11
    2: 27,   # Fiziksel Pin 13
    3: 22,   # Fiziksel Pin 15
    4: 5,    # Fiziksel Pin 29
    5: 6,    # Fiziksel Pin 31
    6: 13,   # Fiziksel Pin 33
    7: 19,   # Fiziksel Pin 35
    8: 26,   # Fiziksel Pin 37
    9: 21,   # Fiziksel Pin 40 (GPIO 21)
}
BUTTON_PIN = 12   # Fiziksel Pin 32
BUZZER_PIN = 16   # Fiziksel Pin 36


class GPIOController:
    """
    LED, buton ve buzzer'i yoneten sinif.
    GPIO mevcut degilse (PC'de) hicbir sey yapmaz — kod calismaya devam eder.
    """

    def __init__(self):
        self._available = GPIO_AVAILABLE
        if not self._available:
            return

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # LED pinleri: CIKIS, baslangicta kapali
        for pin in LED_PINS.values():
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

        # Buton: GIRIS, dahili pull-up direnci aktif (basmayinca HIGH, basinca LOW)
        GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        # Buzzer: CIKIS, baslangicta kapali
        GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)

        print("GPIO baslatildi.")

    # ── LED KONTROL ────────────────────────────────────────────────────────

    def show_digit(self, digit: int):
        """Tahmin edilen rakamin LED'ini yak, diger 9'unu sondur."""
        if not self._available:
            return
        for d, pin in LED_PINS.items():
            GPIO.output(pin, GPIO.HIGH if d == digit else GPIO.LOW)

    def all_off(self):
        """Tum LED'leri ve buzzeri sondur."""
        if not self._available:
            return
        for pin in LED_PINS.values():
            GPIO.output(pin, GPIO.LOW)
        GPIO.output(BUZZER_PIN, GPIO.LOW)

    def blink(self, digit: int, times: int = 3, interval: float = 0.2):
        """Belirli bir rakamin LED'ini yanip sondur (sonuc vurgulama)."""
        if not self._available:
            return
        pin = LED_PINS.get(digit)
        if pin is None:
            return
        for _ in range(times):
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(interval)
            GPIO.output(pin, GPIO.LOW)
            time.sleep(interval)
        GPIO.output(pin, GPIO.HIGH)  # son olarak yakik birak

    # ── BUZZER ────────────────────────────────────────────────────────────

    def beep(self, duration: float = 0.15):
        """Kisa bir bip sesi cikar."""
        if not self._available:
            return
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(BUZZER_PIN, GPIO.LOW)

    def double_beep(self):
        """Iki kisa bip: tahmin tamamlandi sinyali."""
        self.beep(0.1)
        time.sleep(0.1)
        self.beep(0.1)

    # ── BUTON ─────────────────────────────────────────────────────────────

    def button_pressed(self) -> bool:
        """Su an butona basilmis mi? (anlık kontrol)"""
        if not self._available:
            return False
        return GPIO.input(BUTTON_PIN) == GPIO.LOW

    def wait_for_button(self, timeout: float = None):
        """
        Butona basilana kadar bekle.
        timeout: maksimum bekleme suresi (saniye). None = sonsuza kadar bekle.
        Donus degeri: True (basıldı) | False (timeout)
        """
        if not self._available:
            input("Butona basin simulasyonu: Enter'a basin...")
            return True
        print("Tahmin icin BUTONA BASIN...")
        start = time.time()
        while not self.button_pressed():
            if timeout and (time.time() - start) > timeout:
                return False
            time.sleep(0.05)
        time.sleep(0.25)  # titreme onleme (debounce)
        return True

    # ── ANIMASYONLAR ──────────────────────────────────────────────────────

    def startup_animation(self):
        """Acilista LED'leri 0'dan 9'a kadar sirayla yak — sistem hazir sinyali."""
        if not self._available:
            return
        print("Acilis animasyonu...")
        for d in range(10):
            GPIO.output(LED_PINS[d], GPIO.HIGH)
            time.sleep(0.08)
        time.sleep(0.3)
        self.all_off()
        self.beep(0.3)  # tek uzun bip: hazir
        print("Sistem hazir.")

    def processing_animation(self, duration: float = 1.0):
        """Model tahmin yaparken LED'leri geri sayimla goster."""
        if not self._available:
            return
        steps = int(duration / 0.1)
        for i in range(steps):
            d = i % 10
            self.show_digit(d)
            time.sleep(0.1)
        self.all_off()

    # ── TEMIZLIK ──────────────────────────────────────────────────────────

    def cleanup(self):
        """Program kapatilirken GPIO pinlerini sifirla."""
        if not self._available:
            return
        self.all_off()
        GPIO.cleanup()
        print("GPIO temizlendi.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()
