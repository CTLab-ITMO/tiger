# Semantic IDs Generation and TIGER Training

## Overview

This project implements a complete pipeline for generating **Semantic IDs** and training the **TIGER** recommendation model. The system transforms content embeddings into compact semantic identifiers that enable efficient large-scale recommendation.

## Architecture Pipeline

The complete training pipeline consists of 5 sequential steps:

1. **SASRec Training**: Generate collaborative embeddings from user interaction data
2. **RQ-VAE Training**: Learn hierarchical vector quantization on content embeddings. Loss also uses collaborative embeddings from previous step. 
3. **Semantic IDs Generation**: Create semantic identifiers for all items using trained RQ-VAE
4. **Data Format Conversion**: Transform interaction data format for TIGER compatibility
5. **TIGER Training**: Train the recommendation model using hierarchical semantic IDs

## Project Structure

```
/.../tiger/
├── tiger/                              # Main TIGER implementation
│   ├── train_sasrec.py                # Step 1: SASRec collaborative embedding training
│   ├── train_tiger.py                 # Step 5: TIGER model training  
│   └── configs/train/                 # Training configurations
│       ├── sasrec_train_config.json   # SASRec config
│       └── tiger_train_config.json    # TIGER config
├── RQ-VAE/                            # RQ-VAE and Semantic IDs generation
│   ├── train_rqvae.py                 # Step 2: RQ-VAE model training
│   ├── generate_indices.py            # Step 3: Semantic IDs generation
│   ├── generate_all_data_inter.py     # Step 4: Data format conversion
│   └── configs/                       # RQ-VAE configurations
│       ├── rqvae_config.json          # RQ-VAE training config
│       └── generate_indices_rqvae.json # Semantic IDs generation config
├── data/Beauty/                       # Example dataset
│   ├── all_data.txt                   # User interaction data
│   └── final_data_reduced.pkl         # Item content embeddings
├── checkpoints/                       # Model checkpoints and embeddings
│   ├── [sasrec_embeddings.pt]         # Collaborative embeddings (Step 1 output)
│   ├── [rqvae_models/]               # RQ-VAE checkpoints (Step 2 output)
│   └── [semantic_ids.json]           # Generated semantic IDs (Step 3 output)
└── tensorboard_logs/                  # Training logs and metrics
    ├── [sasrec_logs/]                # SASRec training metrics
    └── [tiger_logs/]                 # TIGER training metrics
```

## Semantic IDs Concept

**Semantic IDs** are hierarchical compact representations of items generated through vector quantization:
- Each item gets a sequence of discrete tokens (e.g., `[<a_42>, <b_15>, <c_238>, <d_91>]`)
- These tokens represent semantic clusters at different granularity levels
- Enable efficient indexing and retrieval for recommendation systems

## Complete Training Pipeline

### Step 1: SASRec Collaborative Embedding Training

Generate dense collaborative embeddings from user-item interaction patterns.

```bash
cd /.../tiger/tiger
python train_sasrec.py --params configs/train/sasrec_train_config.json
```

**Purpose**: Learn collaborative patterns and generate item embeddings  
**Input**: Raw user interaction data (`data/Beauty/all_data.txt`)  
**Output**: Collaborative embeddings saved in `checkpoints/`

### Step 2: RQ-VAE Training

Train Residual Vector Quantization Variational AutoEncoder to learn hierarchical quantization.

```bash
cd /.../tiger/RQ-VAE
python train_rqvae.py --params configs/rqvae_config.json
```

**Purpose**: Learn to quantize content embeddings into discrete hierarchical codes  
**Input**: final_data_reduced.pkl (content item embeddings) and collaborative embeddings from Step 1  
**Output**: Trained RQ-VAE model for semantic ID generation

### Step 3: Semantic IDs Generation

Generate hierarchical semantic identifiers for all items using the trained RQ-VAE model.

```bash
cd /.../tiger/RQ-VAE
python generate_indices.py --params configs/generate_indices_rqvae.json
```

**Purpose**: Convert all item embeddings into semantic ID sequences  
**Input**: Trained RQ-VAE model + content embeddings  
**Output**: Semantic IDs for all items (e.g., `{item_0: ["<a_42>", "<b_15>", "<c_238>", "<d_91>"]}`)

### Step 4: Data Format Conversion

Transform interaction data from SASRec format to TIGER-compatible format.

```bash
cd /.../tiger/RQ-VAE
python generate_all_data_inter.py
```

**Purpose**: Convert data representation for TIGER training  
**Reason**: SASRec uses one data format, TIGER requires different interaction representation  
**Output**: TIGER-compatible interaction files

### Step 5: TIGER Model Training

Train the final recommendation model using hierarchical semantic IDs.

```bash
cd /.../tiger/tiger
python train.py --params ../configs/train/tiger_train_config.json
```

**Purpose**: Train recommendation model with semantic ID representations  
**Input**: Semantic IDs from Step 3 + converted data from Step 4  
**Output**: Trained TIGER model for recommendation

## Monitoring and Outputs

### Training Logs
```bash
# Monitor training progress
tensorboard --logdir /.../tiger/tensorboard_logs
```

### Generated Artifacts

**Checkpoints Directory**: `/.../tiger/checkpoints/`
- Collaborative embeddings from SASRec training
- RQ-VAE model checkpoints 
- **Semantic IDs**: Generated hierarchical identifiers for all items
- Final TIGER model weights