import os
import gdown
import zipfile
from pathlib import Path

def setup_dataset():
    """Configura el dataset dentro del contenedor Docker."""
    # Rutas relativas al contenedor Docker
    base_dir = Path("/app/notebooks/Entrega2/simpsons_data")
    base_dir.mkdir(exist_ok=True, parents=True)
    
    zip_path = base_dir / "simpsons_dataset.zip"
    
    # Verificar si ya existe el dataset
    if os.path.exists(base_dir / "simpsons_dataset") and os.path.exists(base_dir / "kaggle_simpson_testset"):
        print("Dataset ya descargado y extraído.")
        return
        
    # Descargar desde Google Drive
    print("Descargando dataset (1.16GB)...")
    gdown.download("https://drive.google.com/uc?id=1DeToSr-V_BJn3FTRD40J2HvlB8BNP_La", str(zip_path), quiet=False)
    
    # Descomprimir
    print("Extrayendo archivos...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_dir)
    
    print("Dataset configurado con éxito.")
    
if __name__ == "__main__":
    setup_dataset()