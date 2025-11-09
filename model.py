# FondefSolarPanels/main.py
import os
import cv2
import uuid
import math
import exifread
import requests
import pandas as pd
from roboflow import Roboflow
from PIL import Image, ImageEnhance

# Inicializa Roboflow
rf = Roboflow(api_key="xKaCtPip3o6zzKkw57JJ")
project = rf.workspace().project("pdt_01-jpzwx")
model = project.version(1).model

def extract_gps_from_exif(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)
    try:
        lat_ref = tags["GPS GPSLatitudeRef"].values
        lon_ref = tags["GPS GPSLongitudeRef"].values
        lat = tags["GPS GPSLatitude"].values
        lon = tags["GPS GPSLongitude"].values
        alt = tags["GPS GPSAltitude"].values[0]

        def to_degrees(value):
            return float(value[0].num) / float(value[0].den) + \
                   float(value[1].num) / float(value[1].den) / 60 + \
                   float(value[2].num) / float(value[2].den) / 3600
        
        lat = to_degrees(lat)
        lon = to_degrees(lon)
        altitude = float(alt.num) / float(alt.den) if alt else 0.0

        if lat_ref != 'N': lat = -lat
        if lon_ref != 'E': lon = -lon
        
        return lat, lon, altitude
    
    except KeyError as e:
        print(f"[!] No se pudo extraer {e} desde {image_path}")
        return None, None, None

def generar_id():
    return f"PNL_{uuid.uuid4().hex[:8].upper()}"

def obtener_elevacion(lat, lon):
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"Elevation ground {response.json()['results'][0]['elevation']}")
            return response.json()['results'][0]['elevation'] # en metros
    except:
        pass
    return None

def calcular_cobertura_en_metros(alt_relat, sensor_mm=6.3, focal_mm=4.5, img_w=4000, img_h=3000):
    hfov_rad = 2 * math.atan(sensor_mm / (2 * focal_mm))
    vfov_rad = hfov_rad * (img_h / img_w)

    width_m = 2 * alt_relat * math.tan(hfov_rad / 2)
    height_m = 2 * alt_relat * math.tan(vfov_rad / 2)

    return width_m, height_m

def calcular_posicion_panel(lat, lon, img_w, img_h, bbox, width_m, height_m):
    deg_por_m = 0.00001

    x_center = bbox['x'] / img_w
    y_center = bbox['y'] / img_h

    offset_lat = (0.5 - y_center) * height_m * deg_por_m
    offset_lon = (x_center - 0.5) * width_m * deg_por_m

    return lat + offset_lat, lon + offset_lon

from PIL import Image, ImageEnhance

def preprocesar_imagen(image_path, output_path=None, max_size=1600, quality=90):
    """
    Preprocesa una imagen para análisis:
    - Convierte a escala de grises.
    - Mejora contraste, brillo y nitidez.
    - Redimensiona automáticamente si excede max_size.
    - Comprime para evitar errores HTTP 413 (imagen demasiado grande).
    """

    # Abrir imagen
    image = Image.open(image_path)

    # --- Reducción de tamaño si es muy grande ---
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        image = image.resize(new_size, Image.LANCZOS)

    # --- Conversión a escala de grises ---
    image = image.convert("L")  # 'L' = grayscale

    # --- Mejoras de imagen ---
    image = ImageEnhance.Contrast(image).enhance(1.5)
    image = ImageEnhance.Brightness(image).enhance(1.2)
    image = ImageEnhance.Sharpness(image).enhance(2.0)

    # --- Guardar versión procesada (si corresponde) ---
    if output_path:
        # Asegura que la carpeta destino exista
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Guarda en formato JPEG comprimido
        image.save(output_path, format="JPEG", quality=quality, optimize=True)

    return image


def get(pred, key):
    """Compatibilidad entre dict y objeto Prediction"""
    if isinstance(pred, dict):
        return pred.get(key)
    return getattr(pred, key, None)

def procesar_imagen(image_path, 
                    string_id,
                    debug=False
                    ):
    # --- Preprocesamiento ---
    preprocessed_path = "data/preprocessed.jpg"
    preprocesar_imagen(image_path, preprocessed_path)

    # --- Ejecutar predicción ---
    results = model.predict(image_path=preprocessed_path, confidence=40)

    # --- Leer imagen preprocesada (la usada por el modelo) ---
    img_pred = cv2.imread(preprocessed_path)
    H_pred, W_pred = img_pred.shape[:2]

   # --- Determinar estructura de resultados ---
    if hasattr(results, "json"):  # Roboflow API
        predictions = results.json().get("predictions", [])
    elif isinstance(results, list) and hasattr(results[0], "boxes"):  # YOLO local
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        predictions = []
        for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, classes):
            predictions.append({
                "x": (x1 + x2) / 2,
                "y": (y1 + y2) / 2,
                "width": x2 - x1,
                "height": y2 - y1,
                "confidence": float(conf),
                "class": results[0].names[int(cls)]
            })
    else:
        print("[!] Formato de resultados no reconocido.")
        return

    if debug:
        print("\n--- Estructura real de la primera predicción ---")
        import json
        print(json.dumps(predictions[0], indent=2))
        print("----------------------------------------------\n")
        print(f"Detecciones totales: {len(predictions)}")

    # --- Filtro: eliminar paneles cortados ---
    filtered_preds = []
    for pred in predictions:
        x = get(pred, "x")
        y = get(pred, "y")
        w = get(pred, "width")
        h = get(pred, "height")
        cls = get(pred, "class")
        conf = get(pred, "confidence")

        if None in [x, y, w, h, cls, conf]:
            continue

        x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2

        # margen relativo al tamaño de la imagen usada para predecir
        margin_x = W_pred * 0.01
        margin_y = H_pred * 0.01

        # aplicar filtro
        if x1 < margin_x or y1 < margin_y or x2 > W_pred - margin_x or y2 > H_pred - margin_y:
            continue

        filtered_preds.append(pred)

    if debug:
        print(f"Detecciones filtradas: {len(filtered_preds)}")

    # --- Dibujar sobre la imagen preprocesada ---
    for pred in filtered_preds:
        x = int(get(pred, "x") - get(pred, "width") / 2)
        y = int(get(pred, "y") - get(pred, "height") / 2)
        w = int(get(pred, "width"))
        h = int(get(pred, "height"))
        class_name = get(pred, "class")
        conf = get(pred, "confidence")

        cv2.rectangle(img_pred, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_pred, f"{class_name} {conf:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    # Crear carpeta de resultados por string
    result_folder = os.path.join("data", "resultados", string_id)
    os.makedirs(result_folder, exist_ok=True)

    # --- Guardar imagen anotada ---
    output_img_name = f"PRED_{os.path.basename(image_path)}"
    output_img_path = os.path.join(result_folder, output_img_name)
    cv2.imwrite(output_img_path, img_pred)

    # --- Extraer GPS ---
    lat, lon, alt = extract_gps_from_exif(image_path)
    if lat is None:
        print(f"[!] Sin datos GPS en {image_path}")
        return

    elev = obtener_elevacion(lat, lon)
    alt_relat = alt - elev if elev is not None else alt
    print(f"Altitud relativa: {alt_relat:.2f} m")

    # --- Calcular cobertura en metros ---
    width_m, height_m = calcular_cobertura_en_metros(
        alt_relat, sensor_mm=6.3, focal_mm=4.5, img_w=W_pred, img_h=H_pred)
    print(f"Cobertura estimada: {width_m:.2f} m x {height_m:.2f} m")

    # --- Guardar predicciones ---
    registros = []
    for pred in filtered_preds:
        if str(get(pred, "class")).lower() == "panel":
            adj_lat, adj_lon = calcular_posicion_panel(
                lat, lon, W_pred, H_pred, pred, width_m, height_m)
            registros.append({
                "id_panel": generar_id(),
                "lat": adj_lat,
                "lon": adj_lon,
                "conf": round(get(pred, "confidence"), 2),
                "image": os.path.basename(image_path)
            })

    # Archivo Excel global por string
    excel_path = os.path.join(result_folder, f"resultados_{string_id}.xlsx")

    if registros:
        df_out = pd.DataFrame(registros)

        if not os.path.exists(excel_path):
            df_out.to_excel(excel_path, index=False)
            print(f"[✔] Nuevo archivo Excel creado: {excel_path}")
        else:
            prev_df = pd.read_excel(excel_path)
            combined_df = pd.concat([prev_df, df_out], ignore_index=True)
            combined_df.to_excel(excel_path, index=False)
            print(f"[✔] Resultados agregados a: {excel_path}")
        print(f"[✔] {len(registros)} paneles guardados en {excel_path}")
    else:
        print(f"[!] No se detectaron paneles en {image_path}")
    
    return result_folder
# MAIN
# procesar_imagen('data/img_raw_rgb/DJI_20241115111127_0001_V.JPG')
