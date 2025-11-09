from ultralytics import YOLO

def main():
    model = YOLO('runs/train/solar4classes/weights/best.pt')
    model.export(format='onnx')

if __name__ == '__main__':
    main()
