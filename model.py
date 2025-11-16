# ============================================================
#                     model.py (VERSIÓN SIMPLE FILTRO)
# ============================================================

import os
import cv2
import uuid
import math
import glob
import shutil
import exifread
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from PIL import Image, ImageEnhance

from utils import (
    get, merge_panels, bbox_center_w_h,
    meters_per_deg, deg_offsets_from_m,
    pixel_to_meter_scale_from_cone,
    haversine_m
)

# ============================================================
# CONFIG
# ============================================================

CONE_REAL_WIDTH_M = 0.28
MERGE_RADIUS_M = 0.6
USE_FOV_FALLBACK_IF_NO_CONE = True

# directorio para imágenes filtradas
OUTPUT_FILTERED_BASE = os.path.join("data", "img_filtradas")

model = YOLO("Modelo_V2/runs/train/names3/weights/best.pt")


# ============================================================
# UTILIDADES BÁSICAS
# ============================================================

def generar_id():
    return f"PNL_{uuid.uuid4().hex[:8].upper()}"


# ============================================================
# FILTRADO SIMPLE PRE-BATCH (1 SÍ, 2 NO)
# ============================================================

def filtrar_imagenes_simple(input_dir, output_dir):
    """
    Toma 1 imagen, descarta las siguientes 2.
    Mantiene: 1, 4, 7, 10, ... en el orden original.
    """
    os.makedirs(output_dir, exist_ok=True)

    imagenes = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not imagenes:
        print(f"[Filtro simple] ⚠️ No se encontraron imágenes en {input_dir}")
        return []

    kept_paths = []
    counter = 0

    for fname in imagenes:
        if counter % 3 == 0:  # 1 sí, 2 no
            src = os.path.join(input_dir, fname)
            dst = os.path.join(output_dir, fname)
            shutil.copy2(src, dst)
            kept_paths.append(dst)
        counter += 1

    print(f"[Filtro simple] {len(kept_paths)} imágenes seleccionadas.")
    return kept_paths


# ============================================================
# GPS / EXIF
# ============================================================

def extract_gps_from_exif(image_path):
    with open(image_path, "rb") as f:
        tags = exifread.process_file(f, details=True)

    try:
        lat_ref = tags["GPS GPSLatitudeRef"].values
        lon_ref = tags["GPS GPSLongitudeRef"].values
        lat = tags["GPS GPSLatitude"].values
        lon = tags["GPS GPSLongitude"].values
        alt = tags["GPS GPSAltitude"].values[0]

        def conv(v):
            return float(v[0].num)/v[0].den + \
                   float(v[1].num)/v[1].den/60 + \
                   float(v[2].num)/v[2].den/3600

        lat = conv(lat)
        lon = conv(lon)
        alt_m = float(alt.num)/alt.den if alt else 3.0

        if lat_ref != "N":
            lat = -lat
        if lon_ref != "E":
            lon = -lon

        return lat, lon, alt_m
    except:
        return None, None, None


def leer_timestamp(image_path):
    try:
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False)
            if "EXIF DateTimeOriginal" in tags:
                t = str(tags["EXIF DateTimeOriginal"])
                return datetime.strptime(t, "%Y:%m:%d %H:%M:%S")
    except:
        pass
    return datetime.fromtimestamp(os.path.getmtime(image_path))


# ============================================================
# RUMBO (BEARING)
# ============================================================

def rumbo_dron(lat1, lon1, lat2, lon2):
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)

    y = math.sin(dlon) * math.cos(lat2r)
    x = math.cos(lat1r)*math.sin(lat2r) - math.sin(lat1r)*math.cos(lat2r)*math.cos(dlon)

    brng = math.degrees(math.atan2(y, x))
    return (brng + 360) % 360


# ============================================================
# PREPROCESAMIENTO DE IMAGEN
# ============================================================

def preprocesar_imagen(path, out_path="data/preprocessed.jpg", max_size=1600):
    img = Image.open(path)
    w, h = img.size

    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = ImageEnhance.Brightness(img).enhance(1.1)
    img = ImageEnhance.Sharpness(img).enhance(1.5)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, "JPEG", quality=90, optimize=True)

    return out_path


# ============================================================
# FOV (cuando no hay cono)
# ============================================================

def calcular_cobertura_en_metros(alt, sensor_mm=6.3, focal_mm=4.5, img_w=4000, img_h=3000):
    hfov = 2 * math.atan(sensor_mm / (2 * focal_mm))
    vfov = hfov * (img_h / img_w)

    w_m = 2 * alt * math.tan(hfov / 2)
    h_m = 2 * alt * math.tan(vfov / 2)
    return w_m, h_m


# ============================================================
# PROCESAR UNA SOLA IMAGEN (con rumbo + corrección roll)
# ============================================================

def procesar_imagen(image_path, string_id, debug=False, rumbo=None):

    # ------------------------------------
    # Preprocesamiento
    # ------------------------------------
    pre = preprocesar_imagen(image_path)
    img_pred = cv2.imread(pre)
    H, W = img_pred.shape[:2]

    # ------------------------------------
    # YOLO
    # ------------------------------------
    results = model.predict(pre, show=False, conf=0.5)

    if not (isinstance(results, list) and hasattr(results[0], "boxes")):
        print("[YOLO] Error leyendo cajas")
        return

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    clss = results[0].boxes.cls.cpu().numpy()

    preds = []
    for (x1, y1, x2, y2), c, cls_id in zip(boxes, confs, clss):
        preds.append({
            "x": (x1 + x2) / 2,
            "y": (y1 + y2) / 2,
            "width": x2 - x1,
            "height": y2 - y1,
            "confidence": float(c),
            "class": results[0].names[int(cls_id)]
        })

    # ------------------------------------
    # Filtrar bordes
    # ------------------------------------
    filtered = []
    for p in preds:
        x, y, w, h = p["x"], p["y"], p["width"], p["height"]
        x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2

        if x1 < 0.01 * W or y1 < 0.01 * H or x2 > 0.99 * W or y2 > 0.99 * H:
            continue
        filtered.append(p)

    # separar paneles y conos
    panels = []
    cones = []
    for p in filtered:
        cname = p["class"].lower()
        if "panel" in cname:
            panels.append(p)
        elif "cono" in cname:
            cones.append(p)

    # ------------------------------------
    # Dibujar detecciones en imagen
    # ------------------------------------
    for p in filtered:
        x = int(p["x"] - p["width"] / 2)
        y = int(p["y"] - p["height"] / 2)
        w = int(p["width"])
        h = int(p["height"])
        cv2.rectangle(img_pred, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # ------------------------------------
    # GPS
    # ------------------------------------
    lat, lon, alt = extract_gps_from_exif(image_path)
    if lat is None:
        print("[GPS] no encontrado")
        return

    alt_relat = 4 # incorporar AQUI info EFIX sobe altura relativa al suelo.

    # ------------------------------------
    # Escala por cono
    # ------------------------------------
    m_per_px = None
    if cones:
        widths = [bbox_center_w_h(c)[2] for c in cones]
        m_per_px = CONE_REAL_WIDTH_M / max(widths)

    # FOV fallback
    width_m_fov, height_m_fov = calcular_cobertura_en_metros(alt_relat, img_w=W, img_h=H)

    # ============================================================
    # ORIGEN: centro de la imagen
    # ============================================================

    cx0 = W/2
    cy0 = H/2
    # ============================================================
    # CÁLCULO GEOMÉTRICO DE TODOS LOS PANELES
    # ============================================================

    raw_dxdy = []  # puntos (dx, dy) antes del roll

    rumbo_rad = math.radians(rumbo) if rumbo is not None else None

    for p in panels:
        px, py, _, _ = bbox_center_w_h(p)

        # ------------------------------------
        # px → metros (origen en centro óptico)
        # ------------------------------------
        dx_px = px - cx0           # derecha positivo
        dy_px = cy0 - py           # arriba positivo  (coincide con ENU local)

        if m_per_px:
            dx_m = dx_px * m_per_px
            dy_m = dy_px * m_per_px

        else:
            # Normalizar a [-0.5, +0.5]
            xc = (px - cx0) / W
            yc = (cy0 - py) / H

            dx_m = xc * width_m_fov
            dy_m = yc * height_m_fov

        # ------------------------------------
        # Rotar según rumbo del dron
        # ------------------------------------
        if rumbo_rad is not None:
            r = rumbo_rad

            dx_rot = dx_m * math.cos(r) - dy_m * math.sin(r)
            dy_rot = dx_m * math.sin(r) + dy_m * math.cos(r)
        else:
            dx_rot, dy_rot = dx_m, dy_m

        raw_dxdy.append((dx_rot, dy_rot))

    # ============================================================
    # CORRECCIÓN DE ROLL (alinear fila dentro del frame)
    # ============================================================

    if len(raw_dxdy) >= 2:
        xs = [p[0] for p in raw_dxdy]
        ys = [p[1] for p in raw_dxdy]

        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)

        num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(len(xs)))
        den = sum((xs[i] - mean_x) ** 2 for i in range(len(xs)))
        a = num / den if den > 0 else 0.0  # pendiente

        theta = math.atan(a)

        corrected = []
        for dx, dy in raw_dxdy:
            dx_c = dx * math.cos(-theta) - dy * math.sin(-theta)
            dy_c = dx * math.sin(-theta) + dy * math.cos(-theta)
            corrected.append((dx_c, dy_c))
    else:
        corrected = raw_dxdy
        
    # ============================================================
    # CORRECCIÓN DE ORIENTACIÓN LATERAL (90°)
    # El dron avanza en rumbo, pero la cámara mira de lado
    # ============================================================

    if rumbo is not None:
        # La cámara apunta lateralmente:
        # Si los paneles están a la derecha del centro, la cámara mira al ESTE
        # Si están a la izquierda, mira al OESTE

        # Determinar orientación según posición promedio
        # (si dx promedio > 0 → paneles a la derecha → cámara mirando a la derecha)
        mean_dx = sum(dx for dx, dy in corrected) / len(corrected)

        if mean_dx > 0:
            # Cámara apuntando a la derecha → rotar +90°
            rot = math.radians(90)
        else:
            # Cámara apuntando a la izquierda → rotar -90°
            rot = math.radians(-90)

        corrected_rot = []
        for dx, dy in corrected:
            dx_r = dx * math.cos(rot) - dy * math.sin(rot)
            dy_r = dx * math.sin(rot) + dy * math.cos(rot)
            corrected_rot.append((dx_r, dy_r))
    else:
        corrected_rot = corrected[:]   # asegurar que son sólo tuplas (dx, dy)

    # ============================================================
    # CONVERTIR A LAT/LON Y GUARDAR
    # ============================================================

    # ============================================================
    # CONVERTIR A LAT/LON Y GUARDAR
    # ============================================================

    registros = []
    for (dx_c, dy_c), p in zip(corrected_rot, panels):
        # dy_c = metros hacia el norte, dx_c = metros hacia el este
        dlat, dlon = deg_offsets_from_m(lat, dy_c, dx_c)
        registros.append({
            "id_panel": generar_id(),
            "lat": lat + dlat,
            "lon": lon + dlon,
            "conf": round(p["confidence"], 3),
            "image": os.path.basename(image_path)
        })


    # --------------------------------------
    # Guardar resultados
    # --------------------------------------
    folder = os.path.join("data", "resultados", string_id)
    os.makedirs(folder, exist_ok=True)

    cv2.imwrite(os.path.join(folder, "PRED_" + os.path.basename(image_path)), img_pred)

    excel_path = os.path.join(folder, f"resultados_{string_id}.xlsx")

    if registros:
        df_new = pd.DataFrame(registros)
        df_new["images"] = df_new["image"]

        if not os.path.exists(excel_path):
            df_new.to_excel(excel_path, index=False)
        else:
            df_prev = pd.read_excel(excel_path)
            if "images" not in df_prev.columns:
                df_prev["images"] = df_prev["image"]

            merged = df_prev.copy()

            for _, nw in df_new.iterrows():
                if len(merged) == 0:
                    merged = pd.concat([merged, pd.DataFrame([nw])], ignore_index=True)
                    continue

                d = merged.apply(
                    lambda r: haversine_m(r["lat"], r["lon"], nw["lat"], nw["lon"]),
                    axis=1
                )
                idx = d.idxmin()

                if d.loc[idx] <= MERGE_RADIUS_M:
                    row = merged.loc[idx].to_dict()
                    latm, lonm, confm, imgs = merge_panels(row, nw.to_dict())
                    merged.at[idx, "lat"] = latm
                    merged.at[idx, "lon"] = lonm
                    merged.at[idx, "conf"] = confm
                    merged.at[idx, "images"] = imgs
                else:
                    merged = pd.concat([merged, pd.DataFrame([nw])], ignore_index=True)

            merged.to_excel(excel_path, index=False)

    return folder


# ============================================================
# PROCESAR CARPETA COMPLETA (BATCH)
# ============================================================

def procesar_batch(folder, string_id):

    # 1) Filtrar por lógica simple 1 sí, 2 no
    OUTPUT_FILTERED = os.path.join(OUTPUT_FILTERED_BASE, string_id)
    print("\n[Filtro simple] Filtrando imágenes de la carpeta…")
    filtered_imgs = filtrar_imagenes_simple(
        input_dir=folder,
        output_dir=OUTPUT_FILTERED
    )

    if not filtered_imgs:
        print("[Filtro simple] No quedaron imágenes tras filtrado")
        return

    # 2) GPS + timestamps
    info = []
    for img in filtered_imgs:
        lat, lon, alt = extract_gps_from_exif(img)
        if lat is None:
            continue
        ts = leer_timestamp(img)
        info.append({"path": img, "lat": lat, "lon": lon, "alt": alt, "ts": ts})

    if not info:
        print("[Batch] Ninguna imagen con GPS válido.")
        return

    info.sort(key=lambda x: x["ts"])

    # 3) calcular rumbo por pares consecutivos (fix para 1 sola imagen)
    if len(info) > 1:
        for i in range(len(info)):
            if i == 0:
                info[i]["rumbo"] = rumbo_dron(
                    info[0]["lat"], info[0]["lon"],
                    info[1]["lat"], info[1]["lon"]
                )
            else:
                info[i]["rumbo"] = rumbo_dron(
                    info[i-1]["lat"], info[i-1]["lon"],
                    info[i]["lat"], info[i]["lon"]
                )
    else:
        info[0]["rumbo"] = 0.0  # rumbo arbitrario si solo hay una imagen

    # 4) procesar cada imagen
    for item in info:
        print(f"Procesando {os.path.basename(item['path'])}, rumbo={item['rumbo']}")
        procesar_imagen(
            image_path=item["path"],
            string_id=string_id,
            debug=False,
            rumbo=item["rumbo"]
        )

    print("\n=== Batch finalizado ===")
