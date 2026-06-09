import sqlite3, os
db = sqlite3.connect('mlflow.db')
curr = os.getcwd()
win_path = "C:/Dev_Projects/TAREAS/5.MLOPs/mlops-fundamentals-homework"

# Corregimos la ubicación de los archivos reales (la tabla de experimentos)
db.execute("UPDATE runs SET artifact_uri = REPLACE(artifact_uri, ?, ?)", (win_path, curr))
# Corregimos el origen de las versiones del modelo
db.execute("UPDATE model_versions SET source = REPLACE(source, ?, ?)", (win_path, curr))

db.commit()
db.close()
print(f"✅ Base de datos adaptada a: {curr}")
