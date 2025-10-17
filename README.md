# 🧬 Cancer Pathology Image Classification using Multimodal LLMs

A research project combining **Vision Transformers (ViTs)** and **Large Language Models (LLMs)** to improve cancer classification from pathology images.  
The goal is to build not only accurate but also **explainable and interpretable** AI systems for real clinical use.

---

## 📌 Project Overview

Traditional deep learning methods for cancer classification often lack transparency, which limits their clinical adoption.  
This project integrates image processing and language reasoning using **Multimodal Large Language Models (MLLMs)** to make medical AI more transparent, interpretable, and trustworthy.

---

## 🧠 Problem & Motivation

- Existing ViT/CNN models give accurate predictions but **no explanation** of why.
- Doctors and clinicians need to **understand the reasoning** behind an AI decision.
- Multimodal LLMs can combine image + text reasoning to explain diagnostic decisions.

---

## ⚙️ Methods & Architecture

### 🧩 Key Components:

1. **Prompting-Based Reasoning**
   - Used **GPT-4o** and **LLaVA-1.5-7B** with:
     - 🔹 Zero-shot & Few-shot inference
     - 🔹 Chain-of-Thought (CoT) prompting for step-by-step explanation
     - 🔹 Tree-of-Thought (ToT) prompting for reasoning over multiple paths

2. **Caption-Guided Instruction Tuning**
   - Trained medical captioning models using:
     - **BLIP-2** + **GPT-2** on domain-specific datasets
   - Applied **LoRA-based PEFT (Parameter Efficient Fine-Tuning)** for efficiency
   - Used caption-guided fine-tuning to align vision inputs with clinical text outputs

3. **Retrieval-Augmented Generation (RAG)**
   - Built a dual-channel RAG pipeline:
     - **CLIP ViT-B/32** for visual similarity
     - **Sentence Transformers** for semantic similarity
     - Retrieval fused via **FAISS** to find most relevant image-caption pairs

---

## 📚 Datasets

| Dataset               | Description                                             |
|----------------------|---------------------------------------------------------|
| PatchCamelyon (PCam) | Binary cancerous/non-cancerous image classification     |
| PatchGastricADC22    | Gastric cancer slides with expert-written captions      |
| BIOMEDICA            | Large corpus of pathology images with descriptive text  |

---

## 🛠 Tech Stack

- **Python**, **PyTorch**, **CUDA**
- **HuggingFace Transformers**, **LoRA (PEFT)**
- **CLIP**, **BLIP-2**, **GPT-2**, **LLaVA-1.5**
- **FAISS** for similarity search
- **Sentence-Transformers**, **OpenAI GPT-4o**

---

## 📊 Performance Summary

| Model Setup                  | Accuracy | Recall  | Notes                               |
|-----------------------------|----------|---------|-------------------------------------|
| ViT + LoRA                  | 87%      | 87.5%   | Vision-only, strong baseline        |
| LLaVA-1.5 (few-shot)        | ~75%     | ~77%    | Adds explainability                 |
| BLIP-2 + GPT-2 (captioning) | —        | —       | High-quality domain-specific text   |
| RAG + CoT Prompting         | ~85%     | **88.6%** | Best trade-off of performance + explainability |

---

## 🧪 What Makes This Project Different

- **Caption-Guided Instruction Tuning**: Training on real medical captions—not generic image descriptions.
- **Dual-Channel RAG**: Combining both visual and semantic retrieval strategies for better context.
- **Explainable Prompts**: CoT and ToT provide step-wise reasoning behind predictions.
- **Balanced Design**: Tuned for both accuracy and trust in clinical settings.

---

## 🚀 Setup & Run (Prototype)

> ⚠️ This is a research project. For full model inference, ensure you have GPU support and relevant weights/datasets.


# Clone repo
git clone https://github.com/your-org/cancer-mllm.git
cd cancer-mllm

# Install dependencies
pip install -r requirements.txt

# Run caption generation
python generate_captions.py --model blip2 --dataset gastric

# Run ViT + LoRA classification
python train_classifier.py --model vit --lora

# Run LLaVA reasoning with CoT
python run_llava.py --prompt_strategy cot

# Run RAG inference
python rag_retriever.py --image sample.png
