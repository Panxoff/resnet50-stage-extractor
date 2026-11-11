#!/usr/bin/env python3
"""Extrae los cinco stages de ResNet50 v1, sin redimensionar la entrada.

Instalación (en un entorno compatible con TensorFlow):
    python -m pip install -r requirements.txt
Ejecución, desde cualquier directorio:
    python resnet50_featuresmap.py

Carpetas input/ y output/: junto a este script (colócalo en la raíz del proyecto).
Entrada: uno o varios PNG RGB de exactamente 512 × 512 píxeles.
Salida por imagen: cinco .npy (H,W,C, float32) y una lámina explicativa (visualizacion.png).
La primera ejecución requiere conexión para descargar los pesos de ImageNet.
Los archivos de una imagen se sobrescriben al volver a procesarla.

Notas de diseño:
- Los .npy son los datos reales (sin normalizar ni reescalar). La lámina es solo exploratoria.
- La arquitectura (capas, repeticiones y reducciones) sigue la Tabla 1 del paper; el
  preprocesamiento es el de Keras (BGR y media por canal de ImageNet, sin escalar).
- La forma de visualizar (mapa promedio, canales con mayor activación media, escala por
  cuadro) es una decisión de este script: no proviene del paper.

Fuentes:
He et al., Deep Residual Learning for Image Recognition, CVPR 2016:
https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
Listado de referencia de nombres y conexiones:
https://deeplearning.cms.waikato.ac.nz/user-guide/model-zoo/keras/KerasResNet50/
API oficial:
https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50

ÍNDICE DEL ARCHIVO
    0. Imports
    1. Configuración: rutas, stages a extraer y ajustes de visualización
    2. Lectura de la imagen de entrada
    3. Lámina de visualización (visualizacion.png)
        3.1 Constantes de diseño y paleta de color
        3.2 Conversión de mapas a imagen
        3.3 Utilidades de dibujo
        3.4 Secciones de la lámina
        3.5 Ensamblado de la lámina
    4. Flujo principal: modelo, extracción, validación y guardado
"""

# =========================================================================================
# 0. IMPORTS
# =========================================================================================
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# =========================================================================================
# 1. CONFIGURACIÓN
# =========================================================================================

# ---- Rutas -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"

# ---- Stages a extraer ------------------------------------------------------------------
# Salida del stem y de los cuatro stages residuales.
STAGES = (
    (1, "pool1_pool", (128, 128, 64)),
    (2, "conv2_block3_out", (128, 128, 256)),
    (3, "conv3_block4_out", (64, 64, 512)),
    (4, "conv4_block6_out", (32, 32, 1024)),
    (5, "conv5_block3_out", (16, 16, 2048)),
)
# Por stage: (nombre en el paper, qué ocurre hasta esa salida, reducción respecto a la entrada).
INFO = {
    1: ("stem", "conv 7×7 /2 + max pool /2", 4),
    2: ("conv2_x", "3 bloques residuales", 4),
    3: ("conv3_x", "4 bloques, el primero con /2", 8),
    4: ("conv4_x", "6 bloques, el primero con /2", 16),
    5: ("conv5_x", "3 bloques, el primero con /2", 32),
}

# ---- Ajustes de la visualización -------------------------------------------------------
TOP_K = 8            # canales mostrados por stage (8 caben en una fila; más agregan filas)
CELDA = 128          # lado (px) de la celda donde se dibuja cada mapa
ZOOM_ENTERO = True   # True: amplía cada mapa por un entero (vecino más cercano, sin interpolar)
                     #       para que todos midan CELDA px. False: escala 1:1 (1 valor = 1 px).
MAPA_COLOR = True    # True: paleta tipo "inferno". False: escala de grises.
# -----------------------------------------------------------------------------------------


# =========================================================================================
# 2. LECTURA DE LA IMAGEN DE ENTRADA
# =========================================================================================
def read_input(path):
    with Image.open(path) as img:
        if img.size != (512, 512):
            raise ValueError(f"se requiere 512 × 512; recibido {img.width} × {img.height}")
        if img.mode not in ("RGB", "RGBA"):
            raise ValueError(f"se requiere PNG RGB o RGBA; modo recibido: {img.mode}")
        if img.mode == "RGBA":
            # Política explícita: transparencia compuesta sobre blanco.
            bg = Image.new("RGBA", img.size, "white")
            img = Image.alpha_composite(bg, img)
        return img.convert("RGB")


# =========================================================================================
# 3. LÁMINA DE VISUALIZACIÓN
# #    Todo lo de esta sección solo dibuja; no modifica los datos de los .npy.
# =========================================================================================

# -----------------------------------------------------------------------------------------
# 3.1 Constantes de diseño y paleta de color
# -----------------------------------------------------------------------------------------
W, MARGEN, COLS = 1760, 40, 8
ANCHO = W - 2 * MARGEN
BG, TARJETA, BORDE, TINTA, SUAVE = "#f1f5f9", "#ffffff", "#d9e2ec", "#0f172a", "#64748b"
COLORES = ("#0e7490", "#2563eb", "#7c3aed", "#c026d3", "#e11d48")  # un color por stage

_ANCLAS = ("#000004", "#160b39", "#420a68", "#6a176e", "#932667", "#bc3754",
           "#dd513a", "#f37819", "#fca50a", "#f6d746", "#fcffa4")


def _crear_lut():
    rgb = np.array([[int(c[i:i+2], 16) for i in (1, 3, 5)] for c in _ANCLAS], dtype=np.float32)
    xs, t = np.linspace(0, 1, len(rgb)), np.linspace(0, 1, 256)
    return np.stack([np.interp(t, xs, rgb[:, k]) for k in range(3)], axis=1).round().astype(np.uint8)


LUT = _crear_lut()


# -----------------------------------------------------------------------------------------
# 3.2 Conversión de mapas a imagen
# -----------------------------------------------------------------------------------------
def gray_native(values):
    """Normaliza cada mapa con su propio mín/máx para visualizar; NO cambia los datos .npy."""
    low, high = float(values.min()), float(values.max())
    if high <= low:
        pixels = np.zeros(values.shape, dtype=np.uint8)
    else:
        pixels = np.rint((values-low)/(high-low)*255).clip(0, 255).astype(np.uint8)
    if MAPA_COLOR:
        return Image.fromarray(LUT[pixels], "RGB")
    return Image.fromarray(pixels).convert("RGB")


def ampliar(img):
    """Amplía por un factor entero (sin interpolar). Devuelve la imagen y el factor."""
    z = max(1, CELDA // img.height) if ZOOM_ENTERO else 1
    if z == 1:
        return img, 1
    return img.resize((img.width * z, img.height * z), Image.NEAREST), z


# -----------------------------------------------------------------------------------------
# 3.3 Utilidades de dibujo
# -----------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def font(size, bold=False):
    nombres = ("DejaVuSans-Bold.ttf", "arialbd.ttf") if bold else ("DejaVuSans.ttf", "arial.ttf")
    for name in nombres:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def texto(draw, xy, s, size=16, fill=TINTA, bold=False, centro=False):
    f = font(size, bold)
    x, y = xy
    if centro:
        x -= draw.textlength(s, font=f) / 2
    draw.text((x, y), s, fill=fill, font=f)


def parrafo(draw, xy, s, size, ancho, fill=TINTA, paso=None):
    """Texto con salto de línea automático. Devuelve la y siguiente."""
    f, paso = font(size), paso or int(size * 1.45)
    x, y = xy
    actual = ""
    for palabra in s.split():
        prueba = f"{actual} {palabra}".strip()
        if actual and draw.textlength(prueba, font=f) > ancho:
            draw.text((x, y), actual, fill=fill, font=f)
            y, actual = y + paso, palabra
        else:
            actual = prueba
    if actual:
        draw.text((x, y), actual, fill=fill, font=f)
        y += paso
    return y


def flecha(draw, x1, x2, y, color=SUAVE):
    draw.line((x1, y, x2 - 6, y), fill=color, width=3)
    draw.polygon([(x2, y), (x2 - 12, y - 7), (x2 - 12, y + 7)], fill=color)


def tarjeta(draw, x, y, w, h):
    draw.rounded_rectangle((x, y, x + w, y + h), 16, fill=TARJETA, outline=BORDE, width=2)


def titulo_seccion(draw, x, y, numero, s):
    draw.ellipse((x, y, x + 32, y + 32), fill=TINTA)
    texto(draw, (x + 16, y + 6), str(numero), 18, "#ffffff", True, centro=True)
    texto(draw, (x + 46, y + 2), s, 23, TINTA, True)


def pegar_mapa(canvas, draw, img, x, y):
    """Pega un mapa centrado en su celda de CELDA × CELDA. Devuelve el factor de zoom."""
    img, z = ampliar(img)
    ox, oy = x + (CELDA - img.width) // 2, y + (CELDA - img.height) // 2
    canvas.paste(img, (ox, oy))
    draw.rectangle((ox - 1, oy - 1, ox + img.width, oy + img.height), outline="#94a3b8")
    return z


# -----------------------------------------------------------------------------------------
# 3.4 Secciones de la lámina
# -----------------------------------------------------------------------------------------
def seccion_recorrido(canvas, draw, original, features, y):
    alto = 345
    tarjeta(draw, MARGEN, y, ANCHO, alto)
    titulo_seccion(draw, 64, y + 18, 1, "Recorrido de la imagen por la red (tamaños reales entre etapas)")

    # Entrada
    lado, tx, ty = 192, 64, y + 66
    miniatura = original.resize((lado, lado), Image.LANCZOS)
    canvas.paste(miniatura, (tx, ty))
    draw.rectangle((tx - 1, ty - 1, tx + lado, ty + lado), outline="#94a3b8")
    texto(draw, (tx + lado / 2, ty + lado + 8), "Entrada RGB", 16, TINTA, True, centro=True)
    texto(draw, (tx + lado / 2, ty + lado + 30), "512 × 512 (reducida)", 14, SUAVE, centro=True)

    base = ty + lado          # las pilas se alinean abajo para comparar tamaños
    y_flecha = base - 64
    pilas = []
    for i, ((stage, layer, _), arr) in enumerate(zip(STAGES, features)):
        h, w, c = arr.shape
        cx = 520 + 285 * i
        hojas = int(np.clip(round(np.log2(c / 32)), 1, 6))   # más canales, más hojas
        x0, y0, color = cx - w // 2, base - h, COLORES[i]
        for k in range(hojas, 0, -1):
            draw.rectangle((x0 + 5*k, y0 - 5*k, x0 + 5*k + w, y0 - 5*k + h),
                           fill="#ffffff", outline=color, width=2)
        canvas.paste(gray_native(arr.mean(axis=2)), (x0, y0))      # tamaño real, sin zoom
        draw.rectangle((x0 - 1, y0 - 1, x0 + w, y0 + h), outline=color, width=2)
        texto(draw, (cx, base + 8), f"STAGE {stage}", 17, color, True, centro=True)
        texto(draw, (cx, base + 31), f"{h} × {w} × {c}", 15, TINTA, centro=True)
        texto(draw, (cx, base + 53), f"{INFO[stage][0]} · ÷{INFO[stage][2]}", 14, SUAVE, centro=True)
        pilas.append((cx, w, hojas))

    # Flecha de entrada → stage 1
    x1, x2 = tx + lado + 14, pilas[0][0] - pilas[0][1] // 2 - 12
    flecha(draw, x1, x2, y_flecha)
    mid = (x1 + x2) / 2
    texto(draw, (mid, y_flecha - 66), "Preprocesamiento:", 13, TINTA, True, centro=True)
    texto(draw, (mid, y_flecha - 48), "RGB→BGR y resta de", 13, SUAVE, centro=True)
    texto(draw, (mid, y_flecha - 31), "media ImageNet", 13, SUAVE, centro=True)
    texto(draw, (mid, y_flecha + 12), "Stem: conv 7×7 /2", 13, SUAVE, centro=True)
    texto(draw, (mid, y_flecha + 29), "+ max pool 3×3 /2", 13, SUAVE, centro=True)

    # Flechas entre stages
    etiquetas = (("3 bloques", "igual tamaño"), ("4 bloques", "lado ÷ 2"),
                 ("6 bloques", "lado ÷ 2"), ("3 bloques", "lado ÷ 2"))
    for i in range(4):
        (cx_a, w_a, n_a), (cx_b, w_b, _) = pilas[i], pilas[i + 1]
        x1, x2 = cx_a + w_a // 2 + 5 * n_a + 10, cx_b - w_b // 2 - 12
        flecha(draw, x1, x2, y_flecha)
        mid = (x1 + x2) / 2
        texto(draw, (mid, y_flecha - 40), etiquetas[i][0], 13, TINTA, True, centro=True)
        texto(draw, (mid, y_flecha - 22), etiquetas[i][1], 13, SUAVE, centro=True)
    return alto


def seccion_lectura(canvas, draw, y):
    alto = 215
    tarjeta(draw, MARGEN, y, ANCHO, alto)
    titulo_seccion(draw, 64, y + 18, 2, "Cómo leer los mapas")
    col, ancho = 414, 380
    z_ej = max(1, CELDA // 16) if ZOOM_ENTERO else 1
    bloques = (
        ("Mapa promedio",
         "Promedio de todos los canales de la etapa. Resume en qué zonas de la imagen hay más "
         "actividad en conjunto."),
        (f"Canales con mayor activación media",
         f"De cada etapa se muestran {TOP_K} canales, ordenados de mayor a menor activación media "
         "(#1, #2, …). «c123» es el índice del canal y «media» su activación promedio."),
        ("Color",
         "Cada cuadro se normaliza por separado. El color solo se compara dentro de un mismo "
         "cuadro, no entre cuadros ni entre etapas."),
        ("Zoom",
         ("Los mapas se amplían por un número entero (sin interpolar) para verse del mismo "
          "tamaño. El factor aparece en cada etapa; el tamaño real está en la sección 1.")
         if ZOOM_ENTERO else
         "Los mapas se muestran a escala 1:1: un valor del mapa ocupa un píxel de la imagen."),
    )
    for i, (titulo, cuerpo) in enumerate(bloques):
        x = 64 + i * col
        texto(draw, (x, y + 66), titulo, 17, TINTA, True)
        yy = y + 96
        if i == 2:  # barra de color
            barra = Image.fromarray(np.tile(np.arange(256, dtype=np.uint8), (16, 1)))
            barra = Image.fromarray(LUT[np.array(barra)], "RGB") if MAPA_COLOR else barra.convert("RGB")
            canvas.paste(barra.resize((ancho - 60, 16)), (x, yy))
            draw.rectangle((x - 1, yy - 1, x + ancho - 60, yy + 16), outline="#94a3b8")
            texto(draw, (x, yy + 20), "mín del cuadro", 13, SUAVE)
            fin = x + ancho - 60
            texto(draw, (fin - draw.textlength("máx del cuadro", font=font(13)), yy + 20),
                  "máx del cuadro", 13, SUAVE)
            yy += 50
        parrafo(draw, (x, yy), cuerpo, 15, ancho, fill="#334155")
    return alto


def alto_stage():
    filas = (TOP_K + COLS - 1) // COLS
    return 62 + filas * (CELDA + 44) + 10


def tarjeta_stage(canvas, draw, y, idx, info, arr):
    stage, layer, _ = info
    h, w, c = arr.shape
    color, alto, x = COLORES[idx], alto_stage(), MARGEN
    tarjeta(draw, x, y, ANCHO, alto)

    # Columna izquierda: identidad de la etapa
    draw.rounded_rectangle((x + 24, y + 20, x + 160, y + 54), 17, fill=color)
    texto(draw, (x + 92, y + 26), f"STAGE {stage}", 18, "#ffffff", True, centro=True)
    texto(draw, (x + 24, y + 68), layer, 15, SUAVE)
    texto(draw, (x + 24, y + 94), f"{h} × {w} × {c}", 22, TINTA, True)
    operacion = "salida tras max pooling" if stage == 1 else "salida tras ReLU final"
    texto(draw, (x + 24, y + 130), operacion, 14, SUAVE)
    texto(draw, (x + 24, y + 152), f"{INFO[stage][1]}", 13, SUAVE)
    texto(draw, (x + 24, y + 172), f"{INFO[stage][0]} · lado ÷{INFO[stage][2]} de la entrada", 13, SUAVE)

    # Mapa promedio
    ax, ty = x + 320, y + 62
    texto(draw, (ax, y + 30), "Mapa promedio", 15, TINTA, True)
    z = pegar_mapa(canvas, draw, gray_native(arr.mean(axis=2)), ax, ty)
    texto(draw, (ax + CELDA / 2, ty + CELDA + 8), f"de {c} canales", 14, SUAVE, centro=True)
    zoom = "escala real 1:1" if z == 1 else f"ampliado ×{z}"
    texto(draw, (ax + CELDA / 2, ty + CELDA + 28), zoom, 12, SUAVE, centro=True)
    sep = ax + CELDA + 26
    draw.line((sep, y + 24, sep, y + alto - 24), fill=BORDE, width=2)

    # Canales con mayor activación media
    gx = x + 500
    texto(draw, (gx, y + 30), f"Canales con mayor activación media (de mayor a menor)", 15, TINTA, True)
    medias = arr.mean(axis=(0, 1))
    seleccion = np.argsort(-medias, kind="stable")[:TOP_K]
    for k, canal in enumerate(seleccion):
        fila, col = divmod(k, COLS)
        cx0 = gx + col * (CELDA + 12)
        cy0 = ty + fila * (CELDA + 44)
        pegar_mapa(canvas, draw, gray_native(arr[:, :, canal]), cx0, cy0)
        texto(draw, (cx0 + CELDA / 2, cy0 + CELDA + 6), f"#{k+1} · c{int(canal)}", 14, TINTA, True, centro=True)
        texto(draw, (cx0 + CELDA / 2, cy0 + CELDA + 25), f"media {medias[canal]:.2f}", 12, SUAVE, centro=True)


# -----------------------------------------------------------------------------------------
# 3.5 Ensamblado de la lámina (aquí se crea y se guarda visualizacion.png)
# -----------------------------------------------------------------------------------------
def visualize(original, features, destination):
    """Guarda una única lámina explicativa: recorrido, guía de lectura y resultado por etapa."""
    y_rec, h_rec = 124, 345
    y_lec = y_rec + h_rec + 20
    h_lec = 215
    y_res = y_lec + h_lec + 28
    y_cards = y_res + 48
    gap = 20
    total = y_cards + 5 * alto_stage() + 4 * gap + 78

    canvas = Image.new("RGB", (W, total), BG)
    draw = ImageDraw.Draw(canvas)
    texto(draw, (MARGEN, 26), "Mapas de características de ResNet50 v1", 36, TINTA, True)
    texto(draw, (MARGEN, 76), f"Imagen: {destination.name}  ·  una salida por etapa  ·  "
                              "pesos de ImageNet, red congelada (solo inferencia)", 18, SUAVE)

    seccion_recorrido(canvas, draw, original, features, y_rec)
    seccion_lectura(canvas, draw, y_lec)
    titulo_seccion(draw, MARGEN + 24, y_res, 3, "Resultado por etapa")
    y = y_cards
    for idx, (info, arr) in enumerate(zip(STAGES, features)):
        tarjeta_stage(canvas, draw, y, idx, info, arr)
        y += alto_stage() + gap

    texto(draw, (MARGEN, y + 4),
          "Los datos reales están en stage_1.npy … stage_5.npy (float32, H × W × C). Esta lámina es solo "
          "exploratoria: el promedio no es Grad-CAM ni indica importancia.", 14, SUAVE)
    texto(draw, (MARGEN, y + 28),
          "Fuentes: He et al., CVPR 2016 (Tabla 1); listado de capas de Waikato (KerasResNet50); "
          "API oficial tf.keras.applications.ResNet50.", 14, SUAVE)
    canvas.save(destination / "visualizacion.png")


# =========================================================================================
# 4. FLUJO PRINCIPAL
# =========================================================================================
def main():
    # --- Carpetas y lista de imágenes de entrada (orden alfabético) ---
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    paths = sorted(p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower()==".png")
    if not paths:
        print(f"Coloca tus PNG RGB de 512 × 512 en: {INPUT_DIR}")
        return 0
    # Evita colisiones de carpetas en sistemas que no distinguen mayúsculas.
    stems = [p.stem.casefold() for p in paths]
    if len(stems) != len(set(stems)):
        print("ERROR: hay imágenes con el mismo nombre base. Renómbralas para evitar sobrescrituras.", file=sys.stderr)
        return 1

    # --- Modelo: ResNet50 con pesos de ImageNet y extractor de los 5 stages ---
    import tensorflow as tf
    base = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=(512, 512, 3)
    )
    base.trainable = False
    extractor = tf.keras.Model(base.input, [base.get_layer(name).output for _, name, _ in STAGES])
    extractor.trainable = False
    print(f"TensorFlow {tf.__version__} | ResNet50 ImageNet | pesos congelados | inferencia")

    # --- Bucle por imagen: leer, preprocesar, extraer, validar, guardar ---
    failed = 0
    for path in paths:
        try:
            original = read_input(path)
            # RGB 0..255 float32; preprocess_input aplica BGR y resta medias ImageNet.
            batch = np.asarray(original, dtype=np.float32)[None, ...].copy()
            batch = tf.keras.applications.resnet50.preprocess_input(batch)
            tensors = extractor(batch, training=False)
            features = [np.asarray(t.numpy()[0], dtype=np.float32) for t in tensors]
            # Validación de forma y valores finitos
            for (_, name, expected), arr in zip(STAGES, features):
                if arr.shape != expected or not np.isfinite(arr).all():
                    raise ValueError(f"salida no válida para {name}: {arr.shape}")
            # Guardado de los .npy y de la lámina
            dest = OUTPUT_DIR / path.stem
            dest.mkdir(exist_ok=True)
            for (stage, _, _), arr in zip(STAGES, features):
                np.save(dest / f"stage_{stage}.npy", arr, allow_pickle=False)
            visualize(original, features, dest)
            print(f"OK: {path.name} → {dest}")
        except Exception as exc:
            failed += 1
            print(f"ERROR: {path.name}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())