import os
import shutil

INPUT_DIR = "data/img_raw"
RGB_DIR = "data/img_raw_rgb"
IR_DIR = "data/img_raw_ir"

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Carpeta '{INPUT_DIR}' no encontrada.")
        return

    os.makedirs(RGB_DIR, exist_ok=True)
    os.makedirs(IR_DIR, exist_ok=True)

    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    count_rgb, count_ir = 0, 0

    for img in images:
        src = os.path.join(INPUT_DIR, img)

        if "_V" in img or "_RGB" in img:
            dst = os.path.join(RGB_DIR, img)
            shutil.copy2(src, dst)
            count_rgb += 1
        elif "_T" in img or "_IR" in img:
            dst = os.path.join(IR_DIR, img)
            shutil.copy2(src, dst)
            count_ir += 1

    print(f"✅ {count_rgb} imágenes RGB copiadas a '{RGB_DIR}'")
    print(f"✅ {count_ir} imágenes IR copiadas a '{IR_DIR}'")

if __name__ == "__main__":
    main()
