import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QTableWidget,
    QTableWidgetItem, QFileDialog, QPushButton, QMessageBox, 
    QCheckBox, QLineEdit, QInputDialog
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from model import procesar_imagen, procesar_batch
import pandas as pd


class SolarDashboard(QWidget):
    
    ##  ________________________ FRONT __________________________  ##
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dashboard Solar Panels Fondef")
        self.setMinimumSize(800, 600)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Botón para crear un nuevo string
        self.string_button = QPushButton("Nuevo String")
        self.string_button.setFixedWidth(100)
        self.string_button.clicked.connect(self.crear_nuevo_string)
        self.layout.addWidget(self.string_button)

        # Etiqueta para mostrar el nombre del archivo
        self.image_name_label = QLabel("Imagen no seleccionada")
        self.layout.addWidget(self.image_name_label)
        
        # Etiqueta de Imagen
        self.image_label = QLabel("Selecciona una imagen")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter) 
        self.layout.addWidget(self.image_label)
        
        # Mostrar string actual
        self.string_label = QLabel("String actual: Ninguno")
        self.layout.addWidget(self.string_label)

        # Info metadatos
        self.meta_label = QLabel("")
        self.layout.addWidget(self.meta_label)

        # Tabla de predicciones
        self.pred_table = QTableWidget()
        self.layout.addWidget(self.pred_table)
        
        # Botón para seleccionar imagen
        self.select_button = QPushButton("Seleccionar imagen desde el sistema")
        self.select_button.setDisabled(True)
        self.select_button.clicked.connect(self.select_image)
        self.layout.addWidget(self.select_button)
        
        # Botón para procesar carpeta completa (batch)
        self.batch_button = QPushButton("Procesar carpeta completa")
        self.batch_button.setDisabled(True)  # se habilita cuando haya string
        self.batch_button.clicked.connect(self.select_batch_folder)
        self.layout.addWidget(self.batch_button)

        # Botón para procesar la imagen actual
        self.process_button = QPushButton("Procesar imagen seleccionada")
        self.process_button.clicked.connect(self.procesar_imagen_actual)
        self.process_button.setDisabled(True)
        self.layout.addWidget(self.process_button)

        # Inicialización
        self.string_id = None  # Inicialmente no hay string definido
        self.pred_df = None
        self.selected_image_path = None
        self.newString = False

    ##  ___________________________________ BACK _______________________________  ##

    def select_batch_folder(self):
        if self.string_id is None:
            QMessageBox.warning(self, "Error", "Primero debes crear un String.")
            return

        folder = QFileDialog.getExistingDirectory(
            self,
            "Seleccionar carpeta con imágenes contiguas"
        )

        if not folder:
            return

        QMessageBox.information(self, "Procesando", f"Procesando carpeta:\n{folder}")

        procesar_batch(folder, self.string_id)

        # Recargar Excel del string
        result_path = os.path.join("data", "resultados", self.string_id)
        excel_path = os.path.join(result_path, f"resultados_{self.string_id}.xlsx")

        if os.path.exists(excel_path):
            self.pred_df = pd.read_excel(excel_path)
            self.show_predictions(None)

        QMessageBox.information(self, "Completado", "Batch procesado exitosamente.")


    
    def crear_nuevo_string(self):
        nombre, ok = QInputDialog.getText(self, "Nuevo String", "Ingresa el nombre del nuevo string:")
        if ok and nombre.strip():
            self.string_id = nombre.strip()
            self.string_label.setText(f"String actual: {self.string_id}")
            self.select_button.setDisabled(False)
            self.batch_button.setDisabled(False)
            QMessageBox.information(self, "String creado", f"Ahora se trabaja en: {self.string_id}")
        else:
            QMessageBox.warning(self, "Error", "Nombre de string no válido.")
    
    def select_image(self):
        file_dialog = QFileDialog(self)
        image_path, _ = file_dialog.getOpenFileName(self, "Selecciona una imagen", "", "Imagenes (*.jpg *.jpeg *.png)")
        if image_path:
            self.selected_image_path = image_path
            self.image_name_label.setText(f"Imagen: {os.path.basename(image_path)}")
            self.load_image(image_path, show_prediction=False)
            self.process_button.setDisabled(False)

    def load_image(self, path, show_prediction):
        if not self.selected_image_path:
            return

        if show_prediction:
            if os.path.exists(path):
                pixmap = QPixmap(path).scaled(600, 400, Qt.AspectRatioMode.KeepAspectRatio)
            else:
                pixmap = QPixmap(self.selected_image_path).scaled(600, 400, Qt.AspectRatioMode.KeepAspectRatio)
        else:
            pixmap = QPixmap(self.selected_image_path).scaled(600, 400, Qt.AspectRatioMode.KeepAspectRatio)

        self.image_label.setPixmap(pixmap)
        self.meta_label.setText(f"Ruta: {self.selected_image_path}")

        # Mostrar predicciones si hay
        if self.pred_df is not None:
            self.show_predictions(os.path.basename(self.selected_image_path))

    def procesar_imagen_actual(self):
        if not self.selected_image_path:
            QMessageBox.warning(self, "Error", "No se ha seleccionado ninguna imagen.")
            return

        result_path = procesar_imagen(image_path=self.selected_image_path,  # Acá se corre el modelo
                        string_id=self.string_id, 
                        debug=False
                        )

        # Cargar automáticamente el CSV resultante
        if os.path.exists(result_path):
            try:
                self.pred_df = pd.read_excel(os.path.join(result_path, f"resultados_{self.string_id}.xlsx"))
                self.load_image(show_prediction=True, path=os.path.join(result_path, f"PRED_{os.path.splitext(os.path.basename(self.selected_image_path))[0]}.jpg"))
                self.show_predictions(os.path.basename(self.selected_image_path))
            except Exception as e:
                QMessageBox.warning(self, "Error", f"No se pudo cargar el archivo Excel:\n{e}")

    def show_predictions(self, image_name=None):
        if self.pred_df is None:
            self.pred_table.clear()
            self.pred_table.setRowCount(0)
            self.pred_table.setColumnCount(0)
            return

        df = self.pred_df
        self.pred_table.setColumnCount(len(df.columns))
        self.pred_table.setHorizontalHeaderLabels(df.columns.tolist())
        self.pred_table.setRowCount(len(df))

        for i, (_, row) in enumerate(df.iterrows()):
            for j, val in enumerate(row):
                self.pred_table.setItem(i, j, QTableWidgetItem(str(val)))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SolarDashboard()
    window.show()
    sys.exit(app.exec())
