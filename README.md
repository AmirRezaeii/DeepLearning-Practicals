### DL Practical Assignments
This repository contains all practical homework assignments from the **Deep Learning course** at Sharif University of Technology. Each notebook is a complete, self-contained implementation of a core deep learning concept, from building a neural network with nothing but NumPy, all the way to fine-tuning large language models and implementing diffusion-based image editing.

All notebooks are tested on **Google Colab** (free GPU runtime) and are written in Python with PyTorch.

---

## Notebook Descriptions

### PyTorch Basics — `PyTorch_Basics.ipynb` + `pytorch_basic.py`

The starting point of the course. This notebook walks through the most essential PyTorch operations that are used in every subsequent assignment. Topics include creating and manipulating tensors, slice indexing and assignment, fancy/integer indexing, boolean masking, broadcasting, reshaping and permuting dimensions, reduction operations, and moving tensors between CPU and GPU. The companion file `pytorch_basic.py` implements helper functions (used as a graded submission) for tasks like `create_sample_tensor`, `normalize_columns`, `batched_matrix_multiply` (both with loop and `torch.bmm`), and GPU matrix multiplication via `.cuda()`. A solid grasp of these primitives is assumed in every later notebook.

---

### Neural Network from Scratch — `NN_Scratch.ipynb`

A complete **Multi-Layer Perceptron framework built with NumPy only** — no PyTorch, no Keras. The assignment proceeds modularly: you first implement individual layer classes (linear, ReLU, sigmoid), then a loss function (MSE, cross-entropy), and finally wire them together into a trainable MLP. The backward pass is implemented by hand using the chain rule, with numerical gradient checking to verify correctness. The trained model is evaluated on the **California Housing** regression dataset. The goal is to deeply understand what frameworks like PyTorch are doing under the hood before using them as a black box.

---

### Optimization Algorithms — `Optimizations.ipynb`

An exploration of the optimization landscape of deep learning. Rather than training a real model, this notebook uses **synthetic 2D and 3D test functions** (with saddle points, ravines, and flat regions) to visually compare how different optimizers navigate them. Algorithms covered include vanilla SGD, SGD with Momentum, RMSProp, and Adam. The notebook encourages you to experiment with different learning rates and hyperparameters and observe the effect on convergence, making the behavior of each algorithm intuitive before applying them to real networks.

---

### RNN & LSTM — `RNNImplementation.ipynb`

Implements and compares **Vanilla RNN and LSTM** models for sentiment classification on the **IMDB movie reviews** dataset (50,000 reviews, binary positive/negative labels). The notebook covers the full pipeline: text tokenization with NLTK, vocabulary construction, stopword removal, custom `Dataset` and `DataLoader` classes, and training loop with early stopping. Both the RNN and LSTM are implemented from scratch using PyTorch's `nn.Module`, and their performance (accuracy, precision, recall, F1) is compared side-by-side with a `classification_report`. The notebook also includes analytical questions about gradient flow and the vanishing gradient problem.

---

### Emoji Classification — `EmojiClassification.ipynb`

A CNN-based **multi-class image classifier** for emoji images. The interesting twist here is that the dataset is **procedurally generated**: each training image is synthesized on the fly using PIL, with randomized backgrounds, Gaussian noise, color jitter, rotations, and other augmentations, creating a virtually unlimited and perfectly labeled dataset. The notebook implements a custom `Dataset` class for this generator, builds a CNN classifier, trains it, and evaluates with a confusion matrix and `classification_report`. A "hard mode" variant with more aggressive augmentation is also available for extra challenge.

---

### CAPTCHA Segmentation — `CAPTCHASegmantation.ipynb`

A complete pipeline for **segmenting and recognizing digits in synthetic CAPTCHA images**. The CAPTCHAs are generated with overlapping 5-digit sequences, sinusoidal distractor lines, random rotations, and Gaussian noise — making naive approaches fail. The task requires building a model that can both localize (segment) individual digits and classify them. The notebook uses a CNN-based architecture with a custom data generator (`synthesize_captcha_image`) and fixed hyperparameters for grading consistency. This is a practical introduction to the difference between classification and segmentation tasks.

---

### Variational Autoencoder — `VAE.ipynb`

A full **VAE implementation from scratch** trained on 64×64 face images from the **CelebA-Dialog** dataset. The notebook covers the theory behind the Evidence Lower Bound (ELBO), the reparameterization trick, and KL divergence regularization, then translates it into a working encoder-decoder architecture. After training for 30 epochs on 15,000 images, the notebook explores the learned latent space through interpolation between faces, latent space visualization with UMAP, random sampling from the prior, and reconstruction quality measurement via PSNR and LPIPS. A rigorous implementation that goes well beyond a toy example.

---

### VQ-VAE-2 — `VQVAE.ipynb`

A **Hierarchical Vector-Quantized VAE** (VQ-VAE-2) trained on the **Omniglot** dataset — 1,623 handwritten character classes from 50 world alphabets. The model learns a **two-level discrete latent representation** of each character: a top-level latent captures global topology (the skeleton/identity of the letter), while a bottom-level latent captures fine stroke details (pen pressure, thickness). The notebook introduces vector quantization, the straight-through estimator for backprop through discrete variables, commitment loss, and codebook collapse prevention. Output includes reconstructions, codebook utilization statistics, and visual inspection of what each level of the hierarchy has learned.

---

### DDPM — `DDPM.ipynb`

A complete implementation of a **Denoising Diffusion Probabilistic Model** following the original Ho et al. (2020) paper. The notebook starts from first principles: the forward noising process with a cosine/linear noise schedule, the reverse denoising process parameterized by a U-Net, and the simplified training objective. Also implements **Classifier-Free Guidance (CFG)** for conditional image generation. The notebook walks through all the mathematical details — including the derivation of the ELBO for diffusion models and the connection between DDPM and score matching — before the actual implementation, making it suitable both as a learning resource and a working codebase. Runs on Colab and Kaggle free runtimes.

---

### Diffusion-Based Image Editing — `Diffusion_Based_Image_Editing.ipynb`

Rather than training a diffusion model from scratch, this notebook **dissects a pretrained latent diffusion model** (Stable Diffusion architecture via `diffusers`) and rebuilds its sampling loop by hand. The goal is to understand the internals well enough to perform controlled image editing. Topics covered include the VAE latent compression step and PSNR reconstruction quality, the forward diffusion process in latent space, DDIM deterministic sampling, SDEdit (noise-and-denoise for style transfer), inpainting by masking the latent, and CLIP-conditioned generation. All experiments run on a free Colab T4 GPU using open-weight models — no paid API required.

---

### SimCLR — `SimCLR.ipynb`

A step-by-step implementation of **SimCLR** (*A Simple Framework for Contrastive Learning of Visual Representations*, Chen et al. 2020). The framework learns visual representations without any labels by pulling together two augmented views of the same image in embedding space while pushing apart views from different images. The notebook covers: designing the augmentation pipeline (random crop, color jitter, Gaussian blur, grayscale), building the two-view dataset, the ResNet backbone + MLP projection head, and the **NT-Xent (InfoNCE) contrastive loss**. After self-supervised pretraining, a linear probe is trained on top of the frozen representations to evaluate their quality — comparing favorably against a randomly initialized baseline.

---

### CLIP — `CLIP.ipynb`

An in-depth exploration of **CLIP (Contrastive Language-Image Pretraining)** using the **RESISC45** remote-sensing image dataset. The notebook progresses from zero-shot classification (using CLIP out of the box with no training) to deep analysis of the embedding space. Experiments include: prompt sensitivity analysis (how different text templates affect classification accuracy), embedding visualization, image retrieval using CLIP embeddings as features, and **fine-tuning CLIP with LoRA** adapters for improved domain-specific performance. The notebook also explores the mathematical formulation — cosine similarity in the shared embedding space with a learned temperature parameter — and shows how CLIP's representations can be repurposed for generation tasks.

---

### PEFT / LoRA — `PEFT.ipynb`

A practical comparison of **full fine-tuning vs. parameter-efficient fine-tuning** for large language models. The base model is **Qwen2.5-0.5B-Instruct**, fine-tuned using **QLoRA** (quantized LoRA with 4-bit precision via `bitsandbytes`). Two strategies are compared: (1) a specialized LoRA adapter trained on a single dataset, and (2) a general adapter trained on a mixture of datasets. The frozen base model serves as a zero-shot baseline. The notebook covers the theory behind LoRA (low-rank decomposition of weight updates, rank selection, alpha scaling), the practical setup with Hugging Face `peft`, and the evaluation of each strategy on held-out data. Highlights the dramatic difference between full fine-tuning cost and LoRA adapter size.

---

### RAG — `RAG.ipynb`

A from-scratch implementation of a **Retrieval-Augmented Generation** pipeline. The mathematical formulation is introduced first: given a query x, the model retrieves the top-k relevant chunks z from a knowledge base and conditions generation on them. In practice, the notebook builds: a document chunker, a dense retriever using **sentence-transformers** embeddings, a **FAISS** vector index for approximate nearest-neighbor search, and a generator using **GPT-2 / FLAN-T5**. The dataset is a subset of **SQuAD**. The pipeline is then extended into a conversational interface using **LangChain** with memory, allowing multi-turn Q&A grounded in the retrieved context. Demonstrates concretely why RAG outperforms vanilla generation on knowledge-intensive tasks.

---

### LLM Inference Techniques — `inference_techniques.ipynb`

An empirical study of **how much inference-time strategies can improve a small LLM** — without any training. The model is **Qwen3-0.6B**, evaluated on a 64-example subset of **GSM8K** (grade-school math word problems). Six strategies are implemented and compared: (1) direct answer with `/no_think`, (2) chain-of-thought with `/no_think`, (3) chain-of-thought with `/think` (activating the model's internal reasoning mode), (4) **self-consistency** with N=5 majority-voted samples, (5) **few-shot prompting** with k=1 example, and (6) **self-refinement** (the model critiques and corrects its own answer). Results are compared quantitatively, showing where each strategy helps and where a 0.6B model hits its ceiling regardless of prompting.

---

## Topics at a Glance

| Area | Notebooks |
|---|---|
| **PyTorch & Foundations** | PyTorch Basics, NN from Scratch, Optimizations |
| **CNNs & Computer Vision** | Emoji Classification, CAPTCHA Segmentation |
| **Sequential Models** | RNN Implementation |
| **Generative Models** | VAE, VQ-VAE-2, DDPM, Diffusion Image Editing |
| **Self-Supervised & Contrastive** | SimCLR |
| **Vision-Language** | CLIP |
| **Large Language Models** | PEFT/LoRA, RAG, Inference Techniques |

---

## Requirements

Core dependencies used across notebooks:

```
torch >= 2.0
torchvision
numpy
matplotlib
scikit-learn
transformers
datasets
peft
diffusers
sentence-transformers
faiss-cpu
langchain
tqdm
Pillow
```

Individual notebooks may install additional packages via `pip` in their setup cells.
