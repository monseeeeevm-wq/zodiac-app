import random
from questions import QUESTIONS, SIGNOS
from core import nueva_fila, guardar_fila, cargar_dataset, resumen_estadistico, reporte_a_texto
from clustering import entrenar, guardar_modelo, listar_modelos_guardados, cargar_modelo_por_nombre

random.seed(42)

# Simular 40 respuestas de cuestionario
for i in range(40):
    respuestas = {q["id"]: random.randint(0, 3) for q in QUESTIONS}
    fila = nueva_fila(f"Usuario{i}", random.randint(16, 45),
                      random.choice(["Masculino", "Femenino", "Prefiero no decir"]),
                      random.choice(SIGNOS), respuestas)
    guardar_fila(fila)

df = cargar_dataset()
print("Filas en dataset:", len(df))
print(df[["signo_predominante", "elemento_predominante"]].head())

reporte = resumen_estadistico(df)
print("\n--- Reporte estadístico (texto) ---")
print(reporte_a_texto(reporte)[:500])

for algo in ["kmeans", "jerarquico"]:
    for modo in ["signo", "elemento", "modalidad"]:
        modelo, scaler, resultado = entrenar(df, algoritmo=algo, modo=modo)
        print(f"\n[{algo} / {modo}] silhouette={resultado['silhouette_score']} pureza={resultado['pureza']}")
        model_path, meta_path, metadata = guardar_modelo(modelo, scaler, resultado)
        print("  Guardado:", metadata["nombre_modelo"])

print("\n--- Modelos guardados ---")
for m in listar_modelos_guardados():
    print(" -", m["nombre_modelo"], "| silhouette:", m["silhouette_score"], "| pureza:", m["pureza"])

# Probar carga de modelo por nombre
primero = listar_modelos_guardados()[0]
bundle, meta = cargar_modelo_por_nombre(primero["nombre_modelo"])
print("\nCarga de modelo por metadato OK:", bundle is not None, meta["nombre_modelo"])

print("\nTODO EL FLUJO PASÓ CORRECTAMENTE ✅")
