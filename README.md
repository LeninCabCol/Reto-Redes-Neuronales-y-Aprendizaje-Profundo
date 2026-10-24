# Reto Redes Neuronales y Aprendizaje Profundo

Clasificación de imágenes satelitales del dataset **EuroSAT (RGB)** y comparación entre una red feedforward (MLP) y una red convolucional (CNN).

## Enfoque

Se busca clasificar imágenes satelitales de 64×64 píxeles en 10 clases de uso o cobertura de suelo. La pregunta del proyecto es qué tanto ayuda la estructura convolucional frente a una red densa que trata la imagen como un vector plano.

### Base de datos

[EuroSAT (RGB)](https://www.kaggle.com/datasets/ryanholbrook/eurosat), tomada de Kaggle:

- 27,000 imágenes a color de 64×64×3.
- 10 clases: `AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River` y `SeaLake`.
- Partición estratificada 70 / 15 / 15 (entrenamiento / validación / prueba) con semilla 42.

### Qué se compara: MLP vs. CNN

| Modelo | Descripción |
|---|---|
| **MLP_R2** | Red feedforward 256-128 con ReLU y Dropout 0.5, `lr = 0.0001`. Es la mejor de las 4 corridas del modelo base y sirve de referencia. |
| **CNN_control** | CNN de 3 bloques (32, 64 y 128 filtros, 2 convoluciones 3×3 con BatchNorm por bloque, MaxPool) más una capa densa de 128 con Dropout 0.5. Sin aumento de datos. |
| **CNN_aumento** | Misma CNN, entrenada con aumento de datos (rotaciones de 90° y reflexiones). |

Las tres se entrenan con Adam, `lr = 0.0001`, 20 épocas, lotes de 64 y la misma semilla. El mejor checkpoint se elige por validación. El conjunto de prueba solo se usa al final, con el modelo ya elegido.

## Resultados

**Conjunto de prueba** (4,050 imágenes):

| Modelo | Exactitud | F1 macro | Parámetros |
|---|---|---|---|
| MLP_R2 | 45.2 % | 0.418 | 3,180,170 |
| CNN_aumento | **93.0 %** | **0.928** | 1,337,898 |

**Validación** (incluye la CNN sin aumento):

| Modelo | Exactitud val. | Exactitud entrenamiento (modo eval) | Brecha |
|---|---|---|---|
| MLP_R2 | 45.0 % | 45.0 % | 0.04 pp |
| CNN_control | 91.3 % | 97.4 % | 6.1 pp |
| CNN_aumento | 92.7 % | 94.2 % | 1.4 pp |

- La CNN supera al MLP por unos 48 puntos porcentuales en prueba, con menos de la mitad de los parámetros.
- El MLP base (R1–R4) quedó entre 25 % y 45 % de exactitud de validación. Bajar la tasa de aprendizaje a 0.0001 fue el cambio que más ayudó.
- El aumento de datos mejora la CNN en +1.5 pp de exactitud de validación y reduce el sobreajuste (la brecha pasa de 6.1 a 1.4 pp).
- Clases más difíciles para la CNN: `PermanentCrop` (se confunde con `HerbaceousVegetation`), `Highway` y `River`. El MLP casi no reconoce `Highway` (recall 0.05).

## Estructura del repositorio

```
.
├── .gitignore
├── README.md
├── requirements.txt # Dependencias de Python
├── data/            # Imágenes de EuroSAT (se genera al correr el notebook final o datos.ipynb, no se sube a git)
├── docs/            # Reporte final
├── notebooks/
│   ├── equipo_proyecto_final.ipynb    # Notebook final: descarga, MLP, CNN, aumento de datos y evaluación
│   ├── datos.ipynb                    # Descarga EuroSAT desde Kaggle y guarda los .jpg por clase
│   ├── redFFbaseline.ipynb            # MLP base y 4 experimentos de hiperparámetros (R1–R4)
│   └── equipo3_proyecto_avance4.ipynb # MLP base + CNN, aumento de datos y evaluación final en prueba
└── resultados/
    ├── *.csv          # Métricas, historiales por época, reportes por clase y matrices de confusión
    ├── figures/       # Curvas de entrenamiento, comparaciones y matrices de confusión (.png)
    └── modelos/       # Pesos de los mejores checkpoints (.pt)
```

### Archivos principales de `resultados/`

- `resumen_redFFbase.csv`: comparación de las corridas R1–R4 del MLP.
- `comparacion_validacion_a4.csv`: MLP vs. CNN sin y con aumento, en validación.
- `comparacion_prueba_a4.csv`: MLP vs. CNN elegida, en prueba.
- `efecto_aumento.csv`: diferencia entre CNN con y sin aumento.
- `reporte_*.csv` y `matriz_confusion_*.csv`: métricas por clase y matrices de confusión.
- `historial_*.csv`: pérdida y exactitud por época.

## Cómo usar el repositorio

1. Clonar el repositorio y entrar a la carpeta:

   ```bash
   git clone https://github.com/LeninCabCol/Reto-Redes-Neuronales-y-Aprendizaje-Profundo.git
   cd Reto-Redes-Neuronales-y-Aprendizaje-Profundo
   ```

2. Instalar las dependencias (Python 3.11, como en el entorno `torch-env` donde se corrió el código):

   ```bash
   pip install -r requirements.txt
   ```

   Si se quiere usar GPU, instalar `torch` con la versión de CUDA que corresponda desde [pytorch.org](https://pytorch.org/get-started/locally/).

3. Descargar la base de datos. La primera sección de `notebooks/equipo_proyecto_final.ipynb` (o `notebooks/datos.ipynb`) la descarga. Crea `data/EuroSAT/<clase>/*.jpg` y requiere acceso a Kaggle a través de `kagglehub`. Si las imágenes ya existen, no se vuelven a descargar.

4. Ejecutar los modelos. Lo más simple es correr `notebooks/equipo_proyecto_final.ipynb` completo (*Restart & Run All*), que incluye todo lo siguiente y genera además `resultados/figures/ejemplos_clases.png`. Por separado:
   - `notebooks/redFFbaseline.ipynb`: entrena el MLP base y los experimentos R1–R4.
   - `notebooks/equipo3_proyecto_avance4.ipynb`: reproduce la comparación completa MLP vs. CNN y guarda resultados, figuras y modelos en `resultados/`.

   Los notebooks detectan la raíz del proyecto, así que pueden correrse desde la raíz o desde `notebooks/`. Usan GPU si hay una disponible y la semilla fija es 42.

## Reporte

El reporte final se encuentra en la carpeta [`docs/`](docs/).
