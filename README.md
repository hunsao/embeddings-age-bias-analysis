# Age Bias Analysis in Image Generation Models through Embeddings

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **[Versión en Español](README_ES.md)** | **English Version**

## 📋 Description

This project investigates age bias in diffusion-based image generation models, specifically **Stable Diffusion XL**. Through multimodal embedding analysis, it evaluates how different age groups (young, middle-aged, older) are represented in the latent space of various deep learning models.

The study uses over **500 daily activities** extracted from time-use surveys to generate prompt quadruplets that compare neutral representations with age-specific groups.

## 🎯 Objectives

- Evaluate embedding similarity between age-neutral and age-specific prompts
- Analyze representation biases in image generation models
- Compare multiple embedding spaces: CLIP, DINO, ResNet, VAE
- Visualize embedding distributions using dimensionality reduction techniques (t-SNE, UMAP)

## 🏗️ Project Architecture

### Models Used

1. **Stable Diffusion XL** - Image generation
2. **CLIP (ViT-B/32)** - Text and image embeddings
3. **DINO (ViT-s16)** - Self-supervised image embeddings
4. **ResNet-50** - Supervised image embeddings
5. **VAE (Stable Diffusion)** - Pre-image latent representations

### Evaluation Metrics

- **Cosine Similarity** - For CLIP, DINO, ResNet, and VAE embeddings
- **Split-Product** - Patch-based similarity using DINO
- **t-SNE & UMAP** - Embedding space visualization

## 📁 Repository Structure

```
embeddings-age-bias-analysis/
│
├── Embeddings_140425.ipynb          # Main generation and evaluation notebook
├── embeddings_plots.ipynb           # Interactive visualizations (t-SNE, UMAP)
├── README.md                         # This file (English)
├── README_ES.md                      # Spanish version
├── requirements.txt                  # Project dependencies
└── generated_images/                 # Generated images directory (not included)
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended: 16GB+ VRAM)
- 50GB+ disk space

### Installing Dependencies

```bash
# Clone the repository
git clone https://github.com/hunsao/embeddings-age-bias-analysis.git
cd embeddings-age-bias-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate safetensors
pip install clip-by-openai scikit-image scikit-learn
pip install umap-learn matplotlib pandas pillow
pip install plotly ipywidgets
```

## 💻 Usage

### 1. Image Generation and Evaluation (`Embeddings_140425.ipynb`)

This notebook performs:

- **Prompt quadruplet generation** for 500+ activities
- **Image generation** using Stable Diffusion XL with Euler Ancestral scheduler
- **Embedding extraction** from multiple models (CLIP, DINO, ResNet, VAE)
- **Similarity metric calculation** between embeddings
- **Results export** to CSV

**Quadruplet Structure:**

```python
neutral  = "An ultra realistic portrait photo of a person [activity]"
young    = "An ultra realistic portrait photo of a 25 years-old young person [activity]"
middle   = "An ultra realistic portrait photo of a 45 year-old middle-aged person [activity]"
older    = "An ultra realistic portrait photo of a 75 years-old older person [activity]"
```

**Execution Example:**

```python
# The notebook automatically executes the main() function
results_df = main()
results_df.to_csv("quadruplets_results_v3_041425.csv", index=False)
```

### 2. Embedding Visualization (`embeddings_plots.ipynb`)

This notebook offers:

- **t-SNE visualizations** of image and text embeddings
- **UMAP visualizations** with age group interactivity
- **Interactive Plotly charts** filterable by activity
- **Cosine similarity analysis** between modalities

**Generated Visualizations:**

- t-SNE of Image Embeddings (CLIP)
- t-SNE of Text Embeddings (CLIP)
- Interactive UMAP by activity
- Age group highlighted plots

## 📊 Results

### Main Metrics

The analysis generates a comparative table of average metrics:

| Embedding Space     | Metric  | Neutral vs Young | Neutral vs Middle | Neutral vs Older |
|---------------------|---------|------------------|-------------------|------------------|
| CLIP (prompts)      | Cosine  | 0.XXX           | 0.XXX             | 0.XXX            |
| CLIP (images)       | Cosine  | 0.XXX           | 0.XXX             | 0.XXX            |
| UNET-VAE (latents)  | Cosine  | 0.XXX           | 0.XXX             | 0.XXX            |
| ResNet-50           | Cosine  | 0.XXX           | 0.XXX             | 0.XXX            |
| DINO-s16            | Cosine  | 0.XXX           | 0.XXX             | 0.XXX            |
| DINO-s16            | Split   | 0.XXX           | 0.XXX             | 0.XXX            |

### Key Insights

- Metrics allow identification of systematic biases in age group representation
- CLIP embeddings capture semantic similarities between prompts
- Image embeddings (DINO, ResNet) reveal visual differences
- VAE latent space shows how information is encoded before generation

## 🛠️ Technical Configuration

### Generation Parameters (Stable Diffusion XL)

```python
num_inference_steps = 20
guidance_scale = 7.5
height = 768
width = 768
scheduler = "euler_ancestral"
negative_prompt = "painting, cartoon, anime, render, artwork, 3d render, ..."
```

### Memory Optimizations

```python
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_attention_slicing("auto")
pipe.enable_sequential_cpu_offload()
```

## 📚 Evaluated Activities

The project evaluates 500+ activities categorized in:

- **Sleep and rest activities** (sleeping, napping, relaxing...)
- **Personal care** (shower, shaving, doctor visits...)
- **Work and education** (programmer, teacher, studying...)
- **Household tasks** (cooking, cleaning, laundry...)
- **Shopping and services** (shopping, banking, errands...)
- **Care for others** (childcare, eldercare, helping...)
- **Social activities** (parties, visiting, phone calls...)
- **Leisure and entertainment** (reading, TV, movies, concerts...)
- **Exercise and sports** (running, swimming, team sports...)
- **Artistic activities** (painting, music, crafts...)

See complete activity list in the `Embeddings_140425.ipynb` notebook.

## 🔬 Methodology

### Evaluation Pipeline

1. **Prompt generation** → Systematic quadruplets per activity
2. **Image generation** → Stable Diffusion XL with fixed seed (123)
3. **Embedding extraction** → Multiple models in parallel
4. **Metric calculation** → Cosine similarity and split-product
5. **Storage** → Image caching for reuse
6. **Statistical analysis** → Result aggregation by group

### Reproducibility

- Fixed seed: `torch.manual_seed(123)`
- CUDA generator with seed: `torch.Generator(device="cuda").manual_seed(123)`
- Image cache to avoid regeneration

## 🤝 Contributions

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is under the MIT License. See the `LICENSE` file for more details.

## 👤 Author

**David C.**
- GitHub: [@hunsao](https://github.com/hunsao)
- Location: Barcelona
- Bio: Mainly polsci

## 🙏 Acknowledgments

- [Stability AI](https://stability.ai/) - Stable Diffusion XL
- [OpenAI](https://openai.com/) - CLIP
- [Meta AI](https://ai.meta.com/) - DINO
- [Microsoft](https://www.microsoft.com/) - ResNet
- [Hugging Face](https://huggingface.co/) - Transformers & Diffusers

## 📞 Contact

If you have questions or suggestions, please open an issue in the repository.

---

**Note:** This project is for academic research purposes on AI model biases. Results should not be interpreted as definitive evidence without rigorous statistical analysis.
