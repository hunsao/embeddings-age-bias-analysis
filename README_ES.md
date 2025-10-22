# Análisis de Embeddings y Sesgos de Edad en Modelos de Generación de Imágenes

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Versión en Español** | **[English Version](README.md)**

## 📋 Descripción

Este proyecto investiga los sesgos de edad en modelos de generación de imágenes basados en difusión, específicamente **Stable Diffusion XL**. A través del análisis de embeddings multimodales, se evalúa cómo diferentes grupos de edad (jóvenes, mediana edad, mayores) son representados en el espacio latente de distintos modelos de aprendizaje profundo.

El estudio utiliza más de **500 actividades cotidianas** extraídas de encuestas de uso del tiempo para generar cuadruplets de prompts que comparan representaciones neutrales con grupos de edad específicos.

## 🎯 Objetivos

- Evaluar la similaridad de embeddings entre prompts neutrales y específicos por edad
- Analizar sesgos de representación en modelos de generación de imágenes
- Comparar múltiples espacios de embeddings: CLIP, DINO, ResNet, VAE
- Visualizar la distribución de embeddings mediante técnicas de reducción dimensional (t-SNE, UMAP)

## 🏗️ Arquitectura del Proyecto

### Modelos Utilizados

1. **Stable Diffusion XL** - Generación de imágenes
2. **CLIP (ViT-B/32)** - Embeddings de texto e imagen
3. **DINO (ViT-s16)** - Embeddings de imagen autosupervisados
4. **ResNet-50** - Embeddings de imagen supervisados
5. **VAE (Stable Diffusion)** - Representaciones latentes pre-imagen

### Métricas de Evaluación

- **Cosine Similarity** - Para embeddings de CLIP, DINO, ResNet y VAE
- **Split-Product** - Similitud basada en parches de DINO
- **t-SNE & UMAP** - Visualización de espacios de embeddings

## 📁 Estructura del Repositorio

```
embeddings-age-bias-analysis/
│
├── SCRIPTS/
│   ├── embeddings_image_generation.ipynb  # Notebook principal de generación y evaluación
│   └── embeddings_plots.ipynb             # Visualizaciones interactivas (t-SNE, UMAP)
│
├── PLOTS/
│   ├── INTERACTIVE_UMAP/                  # Visualizaciones UMAP interactivas
│   ├── TSNE/                              # Gráficos t-SNE
│   └── UMAP/                              # Gráficos UMAP
│
├── README.md                         # Versión en inglés
├── README_ES.md                      # Este archivo (Español)
├── requirements.txt                  # Dependencias del proyecto
└── .gitignore                        # Configuración de Git ignore
```

## 🚀 Instalación

### Requisitos Previos

- Python 3.8 o superior
- CUDA compatible GPU (recomendado: 16GB+ VRAM)
- 50GB+ de espacio en disco

### Instalación de Dependencias

```bash
# Clonar el repositorio
git clone https://github.com/hunsao/embeddings-age-bias-analysis.git
cd embeddings-age-bias-analysis

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate safetensors
pip install clip-by-openai scikit-image scikit-learn
pip install umap-learn matplotlib pandas pillow
pip install plotly ipywidgets
```

## 💻 Uso

### 1. Generación de Imágenes y Evaluación (`SCRIPTS/embeddings_image_generation.ipynb`)

Este notebook realiza:

- **Generación de cuadruplets de prompts** para 500+ actividades
- **Generación de imágenes** usando Stable Diffusion XL con scheduler Euler Ancestral
- **Extracción de embeddings** de múltiples modelos (CLIP, DINO, ResNet, VAE)
- **Cálculo de métricas de similitud** entre embeddings
- **Exportación de resultados** a CSV

**Estructura de Cuadruplets:**

```python
neutral  = "An ultra realistic portrait photo of a person [actividad]"
young    = "An ultra realistic portrait photo of a 25 years-old young person [actividad]"
middle   = "An ultra realistic portrait photo of a 45 year-old middle-aged person [actividad]"
older    = "An ultra realistic portrait photo of a 75 years-old older person [actividad]"
```

**Ejemplo de ejecución:**

```python
# El notebook ejecuta automáticamente la función main()
results_df = main()
results_df.to_csv("quadruplets_results_v3_041425.csv", index=False)
```

### 2. Visualización de Embeddings (`SCRIPTS/embeddings_plots.ipynb`)

Este notebook ofrece:

- **Visualizaciones t-SNE** de embeddings de imagen y texto
- **Visualizaciones UMAP** con interactividad por grupo de edad
- **Gráficos interactivos con Plotly** filtrables por actividad
- **Análisis de similitud coseno** entre modalidades

**Visualizaciones Generadas:**

- t-SNE de Image Embeddings (CLIP)
- t-SNE de Text Embeddings (CLIP)
- UMAP interactivo por actividad
- Gráficos destacados por grupo de edad

## 📊 Resultados

### Métricas Principales

El análisis genera una tabla comparativa de métricas promedio:

| Espacio de Embedding | Métrica | Neutral vs Young | Neutral vs Middle | Neutral vs Older |
|---------------------|---------|------------------|-------------------|------------------|
| CLIP (prompts)      | Cosine  | 0.XXX           | 0.XXX             | 0.XXX            |
| CLIP (imágenes)     | Cosine  | 0.XXX           | 0.XXX             | 0.XXX            |
| UNET-VAE (latentes) | Cosine  | 0.XXX           | 0.XXX             | 0.XXX            |
| ResNet-50           | Cosine  | 0.XXX           | 0.XXX             | 0.XXX            |
| DINO-s16            | Cosine  | 0.XXX           | 0.XXX             | 0.XXX            |
| DINO-s16            | Split   | 0.XXX           | 0.XXX             | 0.XXX            |

### Insights Clave

- Las métricas permiten identificar si existen sesgos sistemáticos en la representación de grupos de edad
- Los embeddings de CLIP capturan similitudes semánticas entre prompts
- Los embeddings de imagen (DINO, ResNet) revelan diferencias visuales
- El espacio latente VAE muestra cómo la información se codifica antes de la generación

## 🛠️ Configuración Técnica

### Parámetros de Generación (Stable Diffusion XL)

```python
num_inference_steps = 20
guidance_scale = 7.5
height = 768
width = 768
scheduler = "euler_ancestral"
negative_prompt = "painting, cartoon, anime, render, artwork, 3d render, ..."
```

### Optimizaciones de Memoria

```python
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_attention_slicing("auto")
pipe.enable_sequential_cpu_offload()
```

## 📚 Actividades Evaluadas

El proyecto evalúa 500+ actividades categorizadas en:

- **Actividades de sueño y descanso** (sleeping, napping, relaxing...)
- **Cuidado personal** (shower, shaving, doctor visits...)
- **Trabajo y educación** (programmer, teacher, studying...)
- **Tareas domésticas** (cooking, cleaning, laundry...)
- **Compras y servicios** (shopping, banking, errands...)
- **Cuidado de otros** (childcare, eldercare, helping...)
- **Actividades sociales** (parties, visiting, phone calls...)
- **Ocio y entretenimiento** (reading, TV, movies, concerts...)
- **Ejercicio y deportes** (running, swimming, team sports...)
- **Actividades artísticas** (painting, music, crafts...)

Ver archivo completo de actividades en el notebook `SCRIPTS/embeddings_image_generation.ipynb`.

## 🔬 Metodología

### Pipeline de Evaluación

1. **Generación de prompts** → Cuadruplets sistemáticos por actividad
2. **Generación de imágenes** → Stable Diffusion XL con seed fijo (123)
3. **Extracción de embeddings** → Múltiples modelos en paralelo
4. **Cálculo de métricas** → Similitud coseno y split-product
5. **Almacenamiento** → Caché de imágenes para reutilización
6. **Análisis estadístico** → Agregación de resultados por grupo

### Reproducibilidad

- Seed fijo: `torch.manual_seed(123)`
- Generador CUDA con seed: `torch.Generator(device="cuda").manual_seed(123)`
- Cache de imágenes para evitar regeneración

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👤 Autor

**David C.**
- GitHub: [@hunsao](https://github.com/hunsao)
- Ubicación: Barcelona
- Bio: Mainly polsci

## 🙏 Agradecimientos

- [Stability AI](https://stability.ai/) - Stable Diffusion XL
- [OpenAI](https://openai.com/) - CLIP
- [Meta AI](https://ai.meta.com/) - DINO
- [Microsoft](https://www.microsoft.com/) - ResNet
- [Hugging Face](https://huggingface.co/) - Transformers & Diffusers

## 📞 Contacto

Si tienes preguntas o sugerencias, por favor abre un issue en el repositorio.

---

**Nota:** Este proyecto es con fines de investigación académica sobre sesgos en modelos de IA. Los resultados no deben interpretarse como evidencia definitiva sin un análisis estadístico riguroso adicional.
