import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from collections import deque, Counter

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# ---------- Configuración ----------
BASE = Path(__file__).parent / "Imagenes"   # carpeta "Imagenes" junto a main.py
NOMBRES = ["NEUTRAL", "T", "ESTIRADOS", "MANOS_ARRIBA"]
# archivos: neutral.*, t.*, estirados.*, manos_arriba.*
TAM_IMG = (640, 480)
VENTANA = 6          # frames usados para estabilizar
MIN_VOTOS = 3        # frames iguales necesarios para cambiar de pose
VIS_MIN = 0.3        # visibilidad mínima de los puntos

# Umbrales en grados (ajústalos según lo que veas en pantalla)
NEUTRAL_MAX = 60                 # brazo pegado al cuerpo: ángulo menor a este valor
T_MIN, T_MAX = 60, 125           # brazos horizontales
ESTIRADOS_MIN = 140              # brazos hacia arriba
CODO_RECTO = 135                 # codo considerado estirado

# Manos arriba: codos flexionados y antebrazos casi verticales
CODO_FLEX_MIN, CODO_FLEX_MAX = 45, 130   # rango del ángulo del codo
ANTEBRAZO_MAX = 50                       # desviación máxima del antebrazo respecto a la vertical


# ---------- Utilidades ----------
def cargar_imagenes():
    imgs = {}
    for nombre in NOMBRES:
        coincidencias = list(BASE.glob(f"{nombre.lower()}.*"))
        if not coincidencias:
            raise FileNotFoundError(
                f"No hay imagen para '{nombre}' en {BASE}. "
                f"Archivos encontrados: {[p.name for p in BASE.iterdir()]}"
            )
        img = cv2.imread(str(coincidencias[0]))
        if img is None:
            raise FileNotFoundError(f"No se pudo leer: {coincidencias[0]}")
        imgs[nombre] = cv2.resize(img, TAM_IMG)
    return imgs


def punto(lm, i, w, h):
    return np.array([lm[i].x * w, lm[i].y * h])


def angulo_brazo(hombro, muneca):
    """Ángulo entre hombro->muñeca y la vertical hacia abajo
    (0° = abajo, 90° = horizontal, 180° = arriba)."""
    v = muneca - hombro
    abajo = np.array([0.0, 1.0])
    cos = np.dot(v, abajo) / (np.linalg.norm(v) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def angulo_codo(hombro, codo, muneca):
    a, b = hombro - codo, muneca - codo
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def angulo_antebrazo(codo, muneca):
    """Ángulo entre codo->muñeca y la vertical hacia arriba (0° = antebrazo vertical)."""
    v = muneca - codo
    arriba = np.array([0.0, -1.0])
    cos = np.dot(v, arriba) / (np.linalg.norm(v) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def clasificar(lm, w, h):
    """Devuelve (pose, angulo_brazo_izq, angulo_brazo_der)."""
    ids = [11, 12, 13, 14, 15, 16]
    if any(lm[i].visibility < VIS_MIN for i in ids):
        return None, None, None

    lados = []
    # Izquierdo (11, 13, 15) y derecho (12, 14, 16): hombro, codo, muñeca
    for hom, cod, mun in [(11, 13, 15), (12, 14, 16)]:
        H, C, M = punto(lm, hom, w, h), punto(lm, cod, w, h), punto(lm, mun, w, h)
        lados.append((angulo_brazo(H, M), angulo_codo(H, C, M),
                      angulo_antebrazo(C, M), M[1] < H[1]))  # último: muñeca más arriba que el hombro

    (ang_i, codo_i, ante_i, arriba_i), (ang_d, codo_d, ante_d, arriba_d) = lados
    ambos_rectos = codo_i > CODO_RECTO and codo_d > CODO_RECTO

    if ang_i < NEUTRAL_MAX and ang_d < NEUTRAL_MAX:
        return "NEUTRAL", ang_i, ang_d

    # Manos arriba: codos flexionados, antebrazos verticales y muñecas sobre los hombros
    flexionados = (CODO_FLEX_MIN < codo_i < CODO_FLEX_MAX and
                   CODO_FLEX_MIN < codo_d < CODO_FLEX_MAX)
    antebrazos_verticales = ante_i < ANTEBRAZO_MAX and ante_d < ANTEBRAZO_MAX
    if flexionados and antebrazos_verticales and arriba_i and arriba_d:
        return "MANOS_ARRIBA", ang_i, ang_d
    if (T_MIN < ang_i < T_MAX and T_MIN < ang_d < T_MAX) and ambos_rectos:
        return "T", ang_i, ang_d
    if ang_i > ESTIRADOS_MIN and ang_d > ESTIRADOS_MIN and ambos_rectos:
        return "ESTIRADOS", ang_i, ang_d
    return None, ang_i, ang_d


# ---------- Programa principal ----------
def main():
    imagenes = cargar_imagenes()
    historial = deque(maxlen=VENTANA)
    pose_actual = None          # al inicio no hay ninguna pose
    ventana_abierta = False

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    # Ventana redimensionable: la imagen se ajusta al tamaño de la ventana
    cv2.namedWindow("Camara", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Camara", 800, 600)  # tamaño inicial

    with mp_pose.Pose(model_complexity=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # efecto espejo
            h, w = frame.shape[:2]
            resultado = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            detectada, ang_i, ang_d = None, None, None
            if resultado.pose_landmarks:
                mp_draw.draw_landmarks(frame, resultado.pose_landmarks,
                                       mp_pose.POSE_CONNECTIONS)
                detectada, ang_i, ang_d = clasificar(
                    resultado.pose_landmarks.landmark, w, h)

            # Estabilización: la pose debe repetirse varios frames para cambiar.
            # Si no hay pose válida, se conserva la última (la foto se mantiene).
            historial.append(detectada)
            etiqueta, votos = Counter(historial).most_common(1)[0]
            if etiqueta is not None and votos >= MIN_VOTOS:
                pose_actual = etiqueta

            # Textos en la cámara
            texto = pose_actual if pose_actual else "NINGUNA"
            cv2.putText(frame, f"Pose: {texto}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if ang_i is not None:
                cv2.putText(frame, f"Brazo izq: {ang_i:.0f}  der: {ang_d:.0f}",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)
            cv2.imshow("Camara", frame)

            # La foto aparece con la primera pose detectada y se mantiene
            if pose_actual:
                if not ventana_abierta:
                    cv2.namedWindow("Imagen", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow("Imagen", *TAM_IMG)
                cv2.imshow("Imagen", imagenes[pose_actual])
                ventana_abierta = True

            # Salir con la tecla q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Salir si se cierra alguna ventana con la X
            if cv2.getWindowProperty("Camara", cv2.WND_PROP_VISIBLE) < 1:
                break
            if ventana_abierta and cv2.getWindowProperty("Imagen", cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()