# 🛰️ Solar Panel Segmentation V6

Sistema de detección y clasificación de paneles solares basado en **YOLOv8**, desarrollado para identificar distintas condiciones visuales en imágenes aéreas.

## ⚙️ Descripción

Este proyecto entrena y ejecuta un modelo de visión computacional capaz de clasificar **paneles solares** en distintas categorías:

- 🟩 Panel — panel solar completo y en buen estado.  
- 🟨 Panel_incompleto — panel parcialmente visible o dañado.  
- 🟧 Panel_impureza — panel con suciedad, reflejos o manchas.  
- 🔺 Cono_ref — cono de referencia visible en la toma.

El modelo fue entrenado localmente con una **NVIDIA RTX 4060**, utilizando **YOLOv8x** y un dataset personalizado.

## 📁 Estructura del proyecto

Modelo_V2/
│
├── train_model.py         # Script de entrenamiento  
├── test.py                # Script de pruebas y exportación  
├── runs/train/...         # Resultados y pesos entrenados  
│   └── weights/best.pt    # Modelo final (almacenado con Git LFS)  
├── yolov8x.pt             # Modelo base de YOLOv8 (LFS)  
│
├── app.py                 # Interfaz / dashboard PyQt  
├── imageFilter.py         # Filtros y preprocesamiento  
├── imageKindSorter.py     # Clasificación y organización de imágenes  
└── requirements.txt       # Dependencias del proyecto

## 🧠 Entrenamiento

python train_model.py

El archivo `data.yaml` dentro de `data/` define las rutas del dataset y las clases.

## 📦 Requisitos

- Python 3.11+
- PyTorch + CUDA
- Ultralytics YOLOv8
- OpenCV
- PyQt5

Instalar dependencias:
pip install -r requirements.txt

## 🚀 Resultados

- mAP50 ≈ 0.91  
- mAP50-95 ≈ 0.74  
- 100 epochs, convergencia estable sin overfitting.

## 👨‍💻 Autor

Enrique Leyton — Proyecto académico PDT  
Contacto: https://github.com/kxkesio

---

⚠️ Los modelos `.pt` son gestionados mediante **Git LFS** para evitar limitaciones de tamaño en GitHub.
