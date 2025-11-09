import os
import cv2
import shutil
from skimage.metrics import structural_similarity as ssim

# CONFIGURACIÓN
BASE_DIR = "data"
INPUT_DIR_RGB = os.path.join(BASE_DIR, "img_raw_rgb")
INPUT_DIR_IR = os.path.join(BASE_DIR, "img_raw_ir")
OUTPUT_DIR = os.path.join(BASE_DIR, "img_filtradas")
THRESHOLD_RGB = 0.05
THRESHOLD_IR = 0.37

def compare_images(img1, img2):
    img1 = cv2.resize(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), (400, 400))
    img2 = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), (400, 400))
    score, _ = ssim(img1, img2, full=True)
    return score

def filtrar_imagenes(input_dir, tipo):
    destino = os.path.join(OUTPUT_DIR, tipo)
    os.makedirs(destino, exist_ok=True)

    threshold = THRESHOLD_RGB if tipo == "RGB" else THRESHOLD_IR
    
    imagenes = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not imagenes:
        print(f"⚠️ No se encontraron imágenes en '{input_dir}'")
        return 0

    last_kept = None
    kept_count = 0

    for i, filename in enumerate(imagenes):
        path = os.path.join(input_dir, filename)
        image = cv2.imread(path)
        if image is None:
            continue

        if last_kept is None:
            shutil.copy2(path, os.path.join(destino, filename))
            last_kept = image
            kept_count += 1
            continue

        similarity = compare_images(image, last_kept)
        print(f"[{tipo}] {filename}: SSIM = {similarity:.4f}")

        if similarity < threshold:
            shutil.copy2(path, os.path.join(destino, filename))
            last_kept = image
            kept_count += 1

    print(f"✅ {tipo}: {kept_count} imágenes seleccionadas.")
    return kept_count

def limpiar_carpeta_salida(tipo):
    carpeta = os.path.join(OUTPUT_DIR, tipo)
    if os.path.exists(carpeta):
        for f in os.listdir(carpeta):
            ruta = os.path.join(carpeta, f)
            if os.path.isfile(ruta):
                os.remove(ruta)
            elif os.path.isdir(ruta):
                shutil.rmtree(ruta)
    else:
        os.makedirs(carpeta)

def main():
    print("🔘 ¿Qué tipo de imágenes deseas filtrar?")
    print("1 - RGB")
    print("2 - IR")
    opcion = input("Selecciona (1 o 2): ").strip()

    if opcion == "1":
        tipo = "RGB"
        input_dir = INPUT_DIR_RGB
    elif opcion == "2":
        tipo = "IR"
        input_dir = INPUT_DIR_IR
    else:
        print("❌ Opción inválida. Debes seleccionar 1 o 2.")
        return

    if not os.path.exists(input_dir):
        print(f"❌ La carpeta '{input_dir}' no existe.")
        return

    print(f"🧹 Limpiando carpeta de salida '{tipo}'...")
    limpiar_carpeta_salida(tipo)

    filtrar_imagenes(input_dir, tipo)

if __name__ == "__main__":
    main()
