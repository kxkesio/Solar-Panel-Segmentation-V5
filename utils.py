import os
import math

# === CONFIGURACIÓN GEO/CONOS ===
# Ancho real del cono en metros
CONE_REAL_WIDTH_M = 0.28

def get(pred, key):
    """Compatibilidad entre dict y objeto Prediction"""
    if isinstance(pred, dict):
        return pred.get(key)
    return getattr(pred, key, None)

def meters_per_deg(lat_deg: float):
    """
    Devuelve (m_por_grado_lat, m_por_grado_lon) para un lat dado.
    """
    lat_rad = math.radians(lat_deg)
    m_per_deg_lat = 111_132.92 - 559.82 * math.cos(2*lat_rad) + 1.175 * math.cos(4*lat_rad)
    m_per_deg_lon = 111_412.84 * math.cos(lat_rad) - 93.5 * math.cos(3*lat_rad)
    return m_per_deg_lat, m_per_deg_lon

def deg_offsets_from_m(lat_deg: float, dy_m: float, dx_m: float):
    """
    Convierte desplazamientos en metros (dy hacia Norte(+), dx hacia Este(+))
    a offsets (dlat, dlon) en grados.
    """
    m_per_deg_lat, m_per_deg_lon = meters_per_deg(lat_deg)
    dlat = dy_m / m_per_deg_lat
    dlon = dx_m / m_per_deg_lon
    return dlat, dlon

def bbox_center_w_h(pred):
    x = get(pred, "x"); y = get(pred, "y")
    w = get(pred, "width"); h = get(pred, "height")
    return x, y, w, h

def nearest_item(target_xy, items_xy):
    """
    Retorna (idx, dist_pix) del item más cercano a 'target_xy' en píxeles.
    Si no hay items, retorna (None, None).
    """
    if not items_xy:
        return None, None
    tx, ty = target_xy
    best_i, best_d = None, None
    for i, (ix, iy) in enumerate(items_xy):
        d = math.hypot(ix - tx, iy - ty)
        if best_d is None or d < best_d:
            best_i, best_d = i, d
    return best_i, best_d

def pixel_to_meter_scale_from_cone(cone_bbox_w_px: float):
    """
    Estima escala m/px usando el ancho del cono detectado (aprox. al plano del suelo).
    """
    if cone_bbox_w_px <= 0:
        return None
    return CONE_REAL_WIDTH_M / cone_bbox_w_px

def haversine_m(lat1, lon1, lat2, lon2):
    # Distancia geodésica en metros
    R = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))

def merge_panels(existing_row, new_row):
    """
    Fusión simple: promedio de lat/lon (ponderado por conf), conf=max, agrega imagen a 'images'.
    """
    w1 = existing_row.get("conf", 0.5)
    w2 = new_row.get("conf", 0.5)
    lat = (existing_row["lat"]*w1 + new_row["lat"]*w2) / max(w1 + w2, 1e-9)
    lon = (existing_row["lon"]*w1 + new_row["lon"]*w2) / max(w1 + w2, 1e-9)
    conf = max(existing_row.get("conf", 0.0), new_row.get("conf", 0.0))
    imgs_old = existing_row.get("images", "")
    imgs_set = set([s for s in imgs_old.split("|") if s]) | {new_row.get("image", "")}
    images = "|".join(sorted([s for s in imgs_set if s]))
    return lat, lon, conf, images
