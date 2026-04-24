"""
Import knihoven potřebných pro úlohu:
    - cv2: práce s kamerou a detekce objektů (OpenCV)
    - pydobot: připojení a ovládání robota Dobot Magician
    - numpy: numerické výpočty a práce s poli
    - time: časování a synchronizace operací
"""
import cv2
import numpy as np
from pydobot import Dobot
import time


# === Inicializace robota ===

# Komunikační port robota
port = "COM3"

# Vytvoření instance robota
bot = Dobot(port=port)


# === Definice pracovního prostoru ===

# Výchozí (domovská) pozice robota
home_position = {'x': 228, 'y': 0, 'z': 85.9213, 'r': -25}

# Omezení pracovního prostoru v ose X a Y
X_MIN, X_MAX = 50, 350
Y_MIN, Y_MAX = -150, 150

# Aktuální pozice robota (inicializace na home pozici)
current_x = home_position['x']
current_y = home_position['y']
current_z = home_position['z']


# === Časovací parametry ===

# Čas poslední detekce objektu
last_seen_time = time.time()

# Čas posledního odeslaného příkazu robotu
last_command_time = 0

# Minimální prodleva mezi příkazy (sekundy)
command_delay = 0.3

# Tolerance návratu do home pozice (není dále explicitně využita)
home_tolerance = 5


# === Definice ukládacích pozic (věží) ===

# Souřadnice jednotlivých věží podle barvy
towers = {
    "green":  {"x": 180, "y": 140, "z": -38, "count": 0},
    "yellow": {"x": 180, "y": 200, "z": -38, "count": 0},
    "red":    {"x": 220, "y": 200, "z": -38, "count": 0},
    "blue":   {"x": 220, "y": 140, "z": -38, "count": 0}
}

# Výška jednoho objektu (krychle) pro skládání do věže
cube_height = 25


# === Inicializace kamery ===

# Otevření výchozí kamery (index 0)
cap = cv2.VideoCapture(0)

# Vypnutí automatického vyvážení bílé (pro stabilnější detekci barev)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)


# === Definice barevných rozsahů v HSV prostoru ===

HSV_RANGES = {
    "red": [
        ([0, 120, 80], [10, 255, 255]),
        ([170, 120, 80], [180, 255, 255])
    ],
    "green":  [([50, 80, 80], [75, 255, 255])],
    "blue":   [([95, 80, 80], [125, 255, 255])],
    "yellow": [([20, 120, 120], [40, 255, 255])]
}


# === Parametry řízení pohybu ===

# Přepočet pixelů na milimetry
step_scale = 0.05

# Tolerance vystředění objektu (pixely)
center_tolerance_px = 4

# Střed obrazu kamery (kalibrovaný bod)
frame_center = (90, 230)

# Doba, po kterou musí být objekt ve středu, aby došlo k uchopení
center_hold_time = 1.0


def wait_for_slot():
    """
    Zajišťuje minimální časový odstup mezi příkazy odesílanými robotu.
    """
    global last_command_time

    now = time.time()
    diff = now - last_command_time

    if diff < command_delay:
        time.sleep(command_delay - diff)

    last_command_time = time.time()


def move_bot_safe(x, y, z, r):
    """
    Bezpečný pohyb robota na zadané souřadnice.
    V případě chyby se pokusí o opětovné připojení.
    """
    global bot

    try:
        wait_for_slot()
        bot.move_to(x, y, z, r, wait=True)

    except Exception as e:
        print("Chyba pohybu:", e)

        # Pokus o korektní ukončení spojení
        try:
            bot.close()
        except:
            pass

        time.sleep(1)

        # Pokus o opětovné připojení
        try:
            bot = Dobot(port=port)
            time.sleep(2)

            # Návrat do výchozí pozice po obnovení spojení
            wait_for_slot()
            bot.move_to(
                home_position['x'],
                home_position['y'],
                home_position['z'],
                home_position['r'],
                wait=True
            )

        except Exception as e2:
            print("Opětovné připojení selhalo:", e2)
            return  # Program pokračuje bez pádu


# === Inicializační pohyb do home pozice ===

move_bot_safe(
    home_position['x'],
    home_position['y'],
    home_position['z'],
    home_position['r']
)

# Čas začátku stabilního vystředění objektu
center_start_time = None


# === Hlavní řídicí smyčka ===

while True:
    try:
        # Načtení snímku z kamery
        ret, frame = cap.read()
        if not ret:
            continue

        # Převod do HSV barevného prostoru
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        detected = []

        # === Detekce objektů podle barvy ===
        for color_name, ranges in HSV_RANGES.items():

            # Inicializace masky
            mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)

            # Složení masky z více rozsahů (např. červená)
            for lower, upper in ranges:
                mask += cv2.inRange(
                    hsv_frame,
                    np.array(lower),
                    np.array(upper)
                )

            # Vyhledání kontur
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                # Filtrace malých objektů
                if cv2.contourArea(cnt) < 500:
                    continue

                # Aproximace tvaru
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

                # Filtrace podle počtu vrcholů (přibližně čtverec)
                if 4 <= len(approx) <= 6:
                    x, y, w, h = cv2.boundingRect(approx)

                    # Kontrola poměru stran (přibližně čtverec)
                    if 0.7 < w / float(h) < 1.3:
                        cx = x + w // 2
                        cy = y + h // 2

                        # Vykreslení detekce do obrazu
                        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                        detected.append((color_name, cx, cy))

        # === Zpracování detekovaných objektů ===
        if detected:
            last_seen_time = time.time()

            # Výběr objektu s nejvyšší hodnotou X (priorita zprava)
            color_name, cx, cy = max(detected, key=lambda d: d[1])

            # Odchylka od středu obrazu
            dx = cx - frame_center[0]
            dy = frame_center[1] - cy

            # Přepočet na milimetry
            dx_mm = dx * step_scale
            dy_mm = dy * step_scale

            # Pohyb robota, pokud je odchylka významná
            if abs(dx_mm) > 0.01 or abs(dy_mm) > 0.01:
                current_x += dx_mm
                current_y += dy_mm

                # Omezení do pracovního prostoru
                current_x = max(min(current_x, X_MAX), X_MIN)
                current_y = max(min(current_y, Y_MAX), Y_MIN)

                move_bot_safe(
                    current_x,
                    current_y,
                    current_z,
                    home_position['r']
                )

            # === Kontrola vystředění ===
            if abs(dx) <= center_tolerance_px and abs(dy) <= center_tolerance_px:
                if center_start_time is None:
                    center_start_time = time.time()
            else:
                center_start_time = None

            # === Uchopení objektu ===
            if center_start_time and (
                time.time() - center_start_time >= center_hold_time
            ):
                # Snížení k objektu
                move_bot_safe(current_x, current_y, -42, home_position['r'])

                # Aktivace přísavky
                bot.suck(True)
                time.sleep(0.5)

                # Zvednutí objektu
                move_bot_safe(current_x, current_y, current_z, home_position['r'])

                # Výběr cílové věže
                tower = towers[color_name]
                target_z = tower['z'] + tower['count'] * cube_height

                # Přesun nad věž
                move_bot_safe(
                    tower['x'],
                    tower['y'],
                    home_position['z'],
                    home_position['r']
                )

                # Složení objektu
                move_bot_safe(
                    tower['x'],
                    tower['y'],
                    target_z,
                    home_position['r']
                )

                # Deaktivace přísavky
                bot.suck(False)

                # Aktualizace počtu objektů ve věži
                tower['count'] += 1

                # Návrat do bezpečné výšky
                move_bot_safe(
                    tower['x'],
                    tower['y'],
                    home_position['z'],
                    home_position['r']
                )

                # Návrat do výchozí pozice
                move_bot_safe(
                    home_position['x'],
                    home_position['y'],
                    home_position['z'],
                    home_position['r']
                )

                # Reset aktuální pozice
                current_x = home_position['x']
                current_y = home_position['y']
                current_z = home_position['z']

                center_start_time = None

        # === Automatický návrat do home pozice při nečinnosti ===
        if time.time() - last_seen_time > 6:
            move_bot_safe(
                home_position['x'],
                home_position['y'],
                home_position['z'],
                home_position['r']
            )

            current_x = home_position['x']
            current_y = home_position['y']
            current_z = home_position['z']

            last_seen_time = time.time()

        # Zobrazení obrazu z kamery
        cv2.imshow("frame", frame)

        # Ukončení programu klávesou ESC
        if cv2.waitKey(1) == 27:
            break

    except Exception as loop_error:
        # Ošetření chyby hlavní smyčky bez ukončení programu
        print("Chyba hlavní smyčky:", loop_error)
        time.sleep(0.5)


# === Uvolnění prostředků ===

cap.release()
cv2.destroyAllWindows()

try:
    bot.close()
except:
    pass
