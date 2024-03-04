import os

directorio_base = "dataset/train"

for directorio_actual, subdirectorios, archivos in os.walk(directorio_base):
    print(f"Directorio actual: {directorio_actual}")
    print(f"Subdirectorios: {subdirectorios}")
    print(f"Archivos: {archivos}")
    print("\n")