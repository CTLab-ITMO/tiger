# TIGER-pytorch

This is an external implementation of the TIGER model in PyTorch. The original model is described in the NeurIPS '23 paper **"Recommender Systems with Generative Retrieval"** by Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, and Maheswaran Sathiamoorthy.

**Link to the paper:** [http://arxiv.org/abs/2305.05065](http://arxiv.org/abs/2305.05065)

**Note:** This is not the original implementation but an external PyTorch implementation of the paper.

If you use this code from the repository, please cite the work:
```
@inproceedings{rajput2023recommender,
  title={Recommender Systems with Generative Retrieval},
  author={Rajput, Shashank and Mehta, Nikhil and Singh, Anima and Keshavan, Raghunandan H and Vu, Trung and Heldt, Lukasz and Hong, Lichan and Tay, Yi and Tran, Vinh Q and Samost, Jonah and others},
  booktitle={Advances in Neural Information Processing Systems},
  pages={33415--33437},
  year={2023}
}
```

## Datasets

This repository includes processed data for the Beauty dataset to run experiments immediately. For other datasets (Sports and Toys), please download the preprocessed data from: [https://zenodo.org/records/17351848](https://zenodo.org/records/17351848)

The dataset structure includes the following files for each domain:
```
data/
├── Beauty/
│   ├── Beauty_5.json
│   ├── content_embeddings.pkl
│   ├── index_rqkmeans.json
│   ├── index_rqvae.json
│   ├── inter_new.json
│   └── inter.json
├── Sport/
│   ├── content_embeddings.pkl
│   ├── index_rqkmeans.json
│   ├── index_rqvae.json
│   ├── inter_new.json
│   ├── inter.json
│   └── Sports_and_Outdoors_5.json
└── Toys/
    ├── content_embeddings.pkl
    ├── index_rqkmeans.json
    ├── index_rqvae.json
    ├── inter.json
    └── Toys_and_Games_5.json
```

If you want to work with raw data from the Amazon Review dataset, you can download it from the official source: [https://jmcauley.ucsd.edu/data/amazon/](https://jmcauley.ucsd.edu/data/amazon/). All data processing and preparation scripts are available in the `notebooks` folder.

## Usage

**Important:** All commands should be run from the `tiger/tiger` directory, not from the project root.

**Training SASRec:**
```bash
python ./train_sasrec.py --params ./configs/sasrec_train_config.json
```

**Training TIGER:**
```bash
python ./train_tiger.py --params ./configs/tiger_train_config.json
```

## Reproducibility

Please note that our reported numbers for both SASRec and TIGER are lower than those claimed in the original papers. This is because we do not apply candidate filtering from the final top-k recommendations, which was used in the original evaluations. It will be changed later.

## Results

| Model  | Dataset | N@5 | N@10 | H@5 | H@10 |
|--------|---------|-----|------|-----|------|
| SASRec | Beauty  | 0.01234 | 0.01908 | 0.02392 | 0.04494 |
| TIGER  | Beauty  | **0.02038** | **0.02729** | **0.03260** | **0.05518** |
| SASRec | Sports  | 0.00804 | 0.01147 | 0.01539 | 0.02607 |
| TIGER  | Sports  | **0.01357** | **0.01780** | **0.02141** | **0.03590** |
| SASRec | Toys    | 0.01353 | 0.01964 | 0.02694 | **0.04574** |
| TIGER  | Toys    | **0.01625** | **0.02174** | **0.02689** | 0.04399 |

## Implementation Differences

This implementation differs from the original TIGER paper in several key aspects. We have not yet fine-tuned the models to achieve optimal performance - this will be addressed in future work. Currently, most parameters are taken directly from the original papers.

### Key Differences:

**Training Configuration:**
- **Constant learning rate**: We use a fixed learning rate throughout training instead of a learning rate scheduler

**Tokenization Methods:**
- **Dual tokenization support**: We provide both RQ-VAE and RQ-KMeans implementations for item tokenization, allowing for experimentation with different quantization approaches

**Content Embeddings:**
- **LLaMA 7B for content embeddings**: We use LLaMA 7B model instead of T5 checkoint for generating content embeddings from item descriptions

## Future Work

We plan to investigate these implementation differences more thoroughly and address the following:

- Fine-tune hyperparameters to achieve the best performance metrics
- Implement proper learning rate scheduling strategies
- Decoder-only implementation: Add a decoder-only variant of the TIGER model
- Conduct ablation studies on the impact of different tokenization methods