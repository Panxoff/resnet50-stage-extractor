# Extracción de características con ResNet50

Este proyecto utiliza ResNet50 v1 preentrenada en ImageNet para extraer cinco salidas intermedias de imágenes PNG de 512 × 512 píxeles.

El modelo funciona como un extractor de características: sus pesos permanecen congelados y no se realiza entrenamiento ni clasificación.

## Instalación

Se necesita Python compatible con TensorFlow y pip.

Desde PowerShell, ubicado en la carpeta del proyecto, crea un entorno virtual:

```powershell
python -m venv .venv
```

Instala las dependencias utilizando directamente el Python del entorno:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Si ya tienes creado el entorno `.venv`, puedes omitir su creación.

La primera ejecución con imágenes requiere conexión a Internet para descargar los pesos de ImageNet.

## Preparación de las imágenes

Coloca las imágenes dentro de una carpeta llamada `input`, junto al script.

Las imágenes deben cumplir estos requisitos:

- Formato PNG.
- Dimensiones exactas de 512 × 512 píxeles.
- Modo RGB o RGBA.

El script no redimensiona las imágenes antes de introducirlas en la red. Si tienen transparencia, se componen sobre fondo blanco y se convierten a RGB.

No se procesan subcarpetas de `input`.

## Ejecución

Ejecuta desde PowerShell:

```powershell
.\.venv\Scripts\python.exe resnet50_featuresmap.py
```

Si el entorno virtual ya está activado:

```powershell
python resnet50_featuresmap.py
```

Las carpetas `input` y `output` se ubican respecto al archivo del script. Si no existen, se crean automáticamente.

## Metodología

La imagen se convierte a un arreglo `float32`, conservando inicialmente sus valores en el rango 0–255.

Después se aplica el preprocesamiento oficial de ResNet50 en Keras:

- Conversión de RGB a BGR.
- Resta de las medias de ImageNet por canal.
- Sin división por 255.

Se utiliza ResNet50 con pesos de ImageNet y `include_top=False`, eliminando la parte clasificadora.

La extracción se ejecuta con pesos congelados y `training=False`, manteniendo las capas de normalización en modo inferencia.

## Salidas extraídas

Se extraen cinco puntos de la red: el bloque inicial o stem y los cuatro grupos residuales.

| Archivo | Capa | Dimensiones: alto × ancho × canales |
|---|---|---|
| `stage_1.npy` | `pool1_pool` | 128 × 128 × 64 |
| `stage_2.npy` | `conv2_block3_out` | 128 × 128 × 256 |
| `stage_3.npy` | `conv3_block4_out` | 64 × 64 × 512 |
| `stage_4.npy` | `conv4_block6_out` | 32 × 32 × 1024 |
| `stage_5.npy` | `conv5_block3_out` | 16 × 16 × 2048 |

Las dimensiones corresponden a una entrada de 512 × 512 píxeles.

## Archivos generados

Para una imagen llamada `ejemplo.png`, se crea la carpeta `output/ejemplo`, que contiene:

- `stage_1.npy`
- `stage_2.npy`
- `stage_3.npy`
- `stage_4.npy`
- `stage_5.npy`
- `visualizacion.png`

Los archivos `.npy` conservan todos los canales de cada salida, en formato `float32`, sin normalización adicional para guardarlos. Antes de guardarlos se comprueban sus dimensiones y la ausencia de valores infinitos o NaN.

La imagen `visualizacion.png` muestra:

- El recorrido de la imagen y las dimensiones de las salidas.
- Un mapa promedio de los canales de cada salida.
- Los ocho canales con mayor activación media de cada salida.

Procesar nuevamente una imagen con el mismo nombre sobrescribe sus resultados.

## Interpretación de la visualización

La lámina tiene un propósito exploratorio:

- Cada cuadro se normaliza con su propio mínimo y máximo. Los colores no permiten comparar magnitudes entre cuadros o imágenes.
- Los mapas constantes se dibujan con el color mínimo, incluso cuando su valor es positivo.
- Una mayor activación media no demuestra mayor importancia.
- Los ocho canales seleccionados pueden cambiar entre imágenes. Para comparar un mismo canal, hay que revisar su identificador.
- El mapa promedio mezcla información de distintos canales; no es Grad-CAM ni explica una predicción.

Estas operaciones visuales no modifican los archivos `.npy`.

La selección de canales, el promedio y la representación visual son decisiones del script. El uso de imágenes de 512 × 512 no reproduce el protocolo experimental completo del artículo original.

## Dependencias

- TensorFlow
- NumPy
- Pillow

El archivo `requirements.txt` no fija versiones. Para registrar las versiones de un entorno utilizado en un experimento:

```powershell
.\.venv\Scripts\python.exe -m pip freeze
```

## Archivos excluidos de Git

El repositorio permite únicamente:

- `resnet50_featuresmap.py`
- `README.md`
- `requirements.txt`
- `.gitignore`

Los datasets, imágenes, resultados, entorno virtual y otros scripts permanecen locales.

## Fuentes

- [He et al. — Deep Residual Learning for Image Recognition, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
- [Documentación oficial de TensorFlow — ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50)
- [Waikato — KerasResNet50: nombres de capas y conexiones](https://deeplearning.cms.waikato.ac.nz/user-guide/model-zoo/keras/KerasResNet50/)