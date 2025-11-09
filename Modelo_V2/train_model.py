from ultralytics import YOLO

# 1️⃣ Cargar modelo base (puede ser uno preentrenado o tuyo)
# Recomendado: usar un modelo base liviano

def main():

    model = YOLO('yolov8x.pt')  # n = nano, puedes usar s, m, l, x según tu GPU

    # 2️⃣ Entrenar
    results = model.train(
        data='data\data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        name='names',
        project='runs/train',
        device=0  # 0 para GPU, 'cpu' si no tienes GPU
    )

    # 3️⃣ (Opcional) Evaluar o exportar
    model.val()  # Evaluación en dataset de validación
    # model.export(format='onnx')  # Exporta a formato ONNX, si lo necesitas

if __name__ == "__main__":
    main()