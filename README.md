# Protein Function Prediction via Multimodal LLM

A research-focused project to design a novel multimodal architecture that predicts protein functions from amino acid sequences by combining ESM-2 protein embeddings with Phi-2 language model capabilities. This project bridges protein structure understanding with natural language generation, inspired by vision-language pretraining architectures like BLIP.

## 🎯 Project Overview

This system generates natural language descriptions of protein functions directly from protein sequences, achieving strong semantic understanding with a BERTScore F1 of 0.84. The model learns to translate between protein and text modalities using a multi-objective training approach.

**Key Innovation**: Adapting vision-language architectures to the protein-text domain, treating protein embeddings as "visual" features and function descriptions as captions.

## 🏗️ Architecture

```
Protein Sequence → ESM-2 (frozen) → MLP Projector → Phi-2 + LoRA → Function Description
                    1280-dim          2560-dim         2.7B params
```

### Core Components

1. **Protein Encoder**: ESM-2 (Meta AI, 650M parameters)
   - Frozen backbone preserves pretrained protein knowledge
   - Generates 1280-dimensional global embeddings via mean pooling

2. **Projection Network**: Trainable MLP (8.4M parameters)
   - Maps protein embeddings to language model space
   - Architecture: Linear(1280→2048) → GELU → Linear(2048→2560)

3. **Language Model**: Phi-2 (Microsoft Research, 2.7B parameters)
   - LoRA fine-tuning (r=16, α=32) on attention layers
   - Efficient parameter-efficient training (~1% trainable parameters)

### Training Objective

Multi-task loss combining three components:

```python
Total Loss = LM Loss + 1.0 × Alignment Loss + 0.5 × Contrastive Loss
```

- **Language Modeling Loss**: Standard next-token prediction
- **Alignment Loss**: Cosine similarity between protein and text representations
- **Contrastive Loss**: CLIP-style symmetric cross-entropy (temperature=0.07)

## 📊 Dataset

- **Source**: UniProt Swiss-Prot (high-quality reviewed entries)
- **Size**: 10,000 human proteins
- **Split**: 80% train / 10% validation / 10% test
- **Features**: Sequence, function description, GO terms, EC numbers

### Data Quality Filters
- Sequence length: 50-1000 amino acids
- Function description: minimum 50 characters
- Valid amino acids only
- Reference citations removed

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch transformers accelerate peft bitsandbytes
pip install fair-esm
pip install datasets sentencepiece
pip install biopython biotite
```

### Code Files

The project consists of the following IPYNB files, to be run in sequence. The project was run in Google Colab:

#### 1. Data Preparation (`01_dataset_preparation.ipynb`)
```python
# Fetches and processes UniProt data
# Creates train/val/test splits
# Saves processed datasets as JSON
```

**Outputs**: 
- `data/processed/train.json`
- `data/processed/val.json`
- `data/processed/test.json`

#### 2. Embedding Generation (`02_embedding_gen.ipynb`)
```python
# Generates ESM-2 embeddings for all proteins
# Saves global embeddings (mean pooling)
# Optional: saves per-residue embeddings
```

**Outputs**:
- `data/embeddings/train_embeddings.pt` 
- `data/embeddings/train_metadata.json`
- Similar files for val/test splits

#### 3. Model Training (`03_training.ipynb`)
```python
# Trains the multimodal model
# Implements multi-task objective
# Saves model + projector checkpoints
```

**Outputs**:
- `trained_mllm_v2/` (LoRA weights + tokenizer)
- `trained_mllm_v2/protein_projector.pt`

#### 4. Evaluation (`04_evaluationmetrics.ipynb`)
```python
# Runs inference on Test split (100 samples)
# Evaluates Quantitative Metrics
# Displays sample predictions
# Saves predictions + evaluation metrics artifacts
```

## 📈 Results

### Quantitative Metrics

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **BERTScore F1** | **0.84 ± 0.04** | Strong semantic understanding |
| ROUGE-L F1 | 0.31 ± 0.11 | Different but valid terminology |
| ROUGE-2 F1 | 0.11 ± 0.11 | Scientific paraphrasing |
| BLEU-1 | 0.09 ± 0.10 | Expected for scientific text |
| Jaccard Similarity | 0.08 ± 0.09 | Low lexical overlap by design |

**Key Finding**: High BERTScore with low n-gram metrics indicates the model captures biological meaning while using different (but equally valid) scientific terminology.

### Performance Characteristics

- **Sequence Length**: Best performance on sequences <400 amino acids
- **Protein Families**: Excellent results for well-characterized families (immunoglobulins, enzymes)
- **Length Ratio**: Predictions average 60 words (ground truth: 89 words)

### Sample Predictions

**Example 1: Near-Perfect Match**
```
Protein: Immunoglobulin lambda variable 3-19
Ground Truth: "V region of the variable domain of immunoglobulin light chains..."
Prediction: "V region of the variable domain of immunoglobulin light chains..."
ROUGE-L: 1.0000
```

**Example 2: Semantic Equivalence**
```
Protein: NADH dehydrogenase subunit
Ground Truth: "Accessory subunit of mitochondrial Complex I..."
Prediction: "Accessory subunit of the mitochondrial membrane respiratory chain..."
BERTScore: 0.95 | ROUGE-L: 0.85
```

## 🔧 Training Configuration

### Hyperparameters

```python
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 5e-4 (with linear warmup)
MAX_SEQ_LENGTH = 384 tokens

# Loss weights
ALIGNMENT_WEIGHT = 0.3
CONTRASTIVE_WEIGHT = 0.1
CONTRASTIVE_TEMPERATURE = 0.07

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
```

### Optimization Strategy

- **Optimizer**: AdamW with weight decay
- **Scheduler**: Linear warmup (10% of steps)
- **Mixed Precision**: FP16 with gradient scaling
- **Gradient Clipping**: max_norm=1.0
- **Hardware**: CUDA GPU (tested on Google Colab T4)

## 🎯 Key Design Decisions

### 1. Global Embeddings with Mean Pooling
- **Rationale**: Sufficient for function prediction tasks
- **Benefit**: Simpler architecture, faster training
- **Alternative**: Per-residue processing (available but not used in final model)

### 2. Frozen ESM-2 Encoder
- **Rationale**: Preserves powerful pretrained representations
- **Benefit**: Fewer trainable parameters, faster convergence
- **Trade-off**: Less task-specific adaptation

### 3. Multi-Task Training
- **LM Loss**: Ensures fluent generation
- **Alignment Loss**: Semantic correspondence between modalities
- **Contrastive Loss**: Discriminative capability within batches

### 4. Prompt Engineering
```
[PROTEIN]
LENGTH: {sequence_length}
SEQUENCE: {sequence}
[/PROTEIN]
FUNCTION:
```
- Provides structural context to language model
- Mirrors evaluation-time behavior
- Enables length-aware generation

## 📊 Evaluation Metrics Explained

### Why BERTScore > ROUGE/BLEU for this task?

**BERTScore** captures semantic similarity using contextual embeddings:
- Recognizes that "transporter" ≈ "channel protein"
- Handles scientific paraphrasing naturally
- More aligned with biological accuracy

**ROUGE/BLEU** measure n-gram overlap:
- Penalize valid terminology variations
- Don't capture conceptual equivalence
- Less meaningful for scientific domains

**Example:**
```
Ground Truth: "ATP synthase subunit"
Prediction: "F0F1 ATP synthase complex component"
ROUGE: Low | BERTScore: High ✓
```

## 🚧 Limitations & Future Work

### Current Limitations

1. **Sequence Length**: Performance degrades for very long proteins (>800 AA)
2. **Novel Proteins**: Lower accuracy for proteins with no similar training examples
3. **Multi-Domain Proteins**: Struggles with proteins having multiple distinct functions
4. **Confidence**: No uncertainty quantification in predictions

### Future Improvements

- [ ] Expand training data to include rare protein families
- [ ] Implement confidence scoring (e.g., ensemble methods, entropy-based)
- [ ] Architecture upgrades:
  - Hierarchical encoders for long sequences
  - Attention mechanisms for multi-domain proteins
- [ ] Multi-species support beyond human proteins
- [ ] Integration with structure prediction models (AlphaFold features)
- [ ] Fine-grained evaluation: GO term prediction accuracy


## 📚 References

- Lin, Z., et al. (2022). "Language models of protein sequences at the scale of evolution." bioRxiv. Meta AI.
- Microsoft Research (2023). "Phi-2: The surprising power of small language models."
- Li, J., et al. (2022). "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation." ICML.
- Chen, H., et al. (2024). "Multi-Modal Generative AI: Multi-modal LLM, Diffusion and Beyond." arXiv:2409.14993.
- Wang, L., et al. (2025). "A Comprehensive Review of Protein Language Models." arXiv:2502.06881.

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 📧 Contact

**Sridipta Roy**  
Email: roy.sr@northeastern.edu  
GitHub: [@Sridipta-Roy](https://github.com/Sridipta-Roy)

---

**Note**: This is a research project developed in Google Colab. For production use, additional engineering would be required for robustness, scalability, and deployment.
