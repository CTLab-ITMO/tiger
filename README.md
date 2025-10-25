# Recommender Systems with Generative Retrieval

This repository provides an external PyTorch implementation of the TIGER model. The original approach is introduced in the NeurIPS ’23 paper **“Recommender Systems with Generative Retrieval”** by Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, and Maheswaran Sathiamoorthy.

**Link to the paper:** [http://arxiv.org/abs/2305.05065](http://arxiv.org/abs/2305.05065)


If you use this code, please cite:
```
@inproceedings{rajput2023recommender,
  title     = {Recommender Systems with Generative Retrieval},
  author    = {Rajput, Shashank and Mehta, Nikhil and Singh, Anima and Keshavan, Raghunandan H and Vu, Trung and Heldt, Lukasz and Hong, Lichan and Tay, Yi and Tran, Vinh Q and Samost, Jonah and others},
  booktitle = {Advances in Neural Information Processing Systems},
  pages     = {33415--33437},
  year      = {2023}
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

Metadata JSON files are not included and precomputed content embeddings are provided for all datasets.

If you want to work with raw data from the Amazon Review dataset, you can download it from the official source: [https://jmcauley.ucsd.edu/data/amazon/](https://jmcauley.ucsd.edu/data/amazon/). All data processing and preparation scripts are available in the `notebooks` folder.

## Requirements

In order to launch our implementation install the following packages (tested versions in parentheses):

```bash
pip install murmurhash  # (==1.0.13)
pip install numpy  # (==2.1.2)
pip install tensorboard  # (==2.20.0)
pip install torch  # (==2.6.0+cu126)
pip install transformers  # (==4.50.3)
```

## Usage

**Important:** Execute all commands from the `tiger/tiger` directory (not from the project root).

**Training SASRec:**
```bash
python ./train_sasrec.py --params ./configs/sasrec_train_config.json
```

**Training TIGER:**
```bash
python ./train_tiger.py --params ./configs/tiger_train_config.json
```

## Reproducibility

Because there is no official TIGER reference implementation, we compare on Beauty against the results reported in ["Learnable Item Tokenization for Generative Recommendation"](http://arxiv.org/abs/2405.07314). Results for Yelp and Instruments will be added later. It is also noticable that SasRec is underperforms on this dataset. We are currently investigating this.

## Results

Following the original setup, we report NDCG (N) and Hit Rate (H) with @5, @10, and additionaly @20 cutoffs.


| Model  | Dataset | N@5         | N@10        | N@20        | H@5         | H@10        | H@20        |
|--------|---------|-------------|-------------|-------------|-------------|-------------|-------------|
| SASRec | Beauty  | 0.02087     | 0.02718     | 0.03447     | 0.03197     | 0.051647    | 0.08071     |
| TIGER  | Beauty  | **0.02524** | **0.03191** | **0.03940** | **0.03756** | **0.05822** | **0.08800** |
| SASRec | Sports  | 0.01217     | 0.01550     | 0.01952     | 0.01806     | 0.02846     | 0.04444     |
| TIGER  | Sports  | **0.01484** | **0.01921** | **0.02358** | **0.02315** | **0.03680** | **0.05868** |
| SASRec | Toys    | 0.02274     | 0.02831     | 0.03436     | 0.03462     | 0.05202     | 0.07624     |
| TIGER  | Toys    | **0.02359** | **0.02917** | **0.03531** | **0.03488** | **0.05224** | **0.07696** |

## Implementation Differences

The original paper states “Overall, the model has around 13 million parameters.” Our implementation currently has approximately 2.5× fewer parameters (~5M vs. ~13M), which is the primary architectural discrepancy. It remains to be determined which components are under‑parameterized in our implementation.

### Minor Differences:

Also, in our implementation there are some other differences.

- **Constant learning rate**: a fixed LR is used for the entire training; no scheduler is applied. With an initial LR of 0.01 and inverse square‑root decay, we observed instability during first 10k steps and inferior convergence compared with a fixed LR.

- **Early Stopping**: training stops if the validation metric (NDCG@20) does not improve for 40 epochs; this consistently yielded the strongest checkpoints in our runs.

- **LLaMA 7B inference for content embeddings**: we use a LLaMA‑7B checkpoint for content‑based embeddings instead of a T5 checkpoint as in the original work. We plan to evaluate T5 later as well.

## Future Work

We plan to investigate these implementation differences more thoroughly and address the following:

- Add a coverage metric to the evaluation protocol
- Add [LETTER](http://arxiv.org/abs/2405.07314) and [PLUM](https://arxiv.org/pdf/2510.07784v1) implementations.
- Add more datasets (e.g.: Yelp, Instruments, Books)
- Fine-tune hyperparameters to achieve the best performance metrics
- Add a decoder-only variant of the TIGER model
- Conduct ablation studies on the impact of different tokenization methods (RQ-VAE vs RQ-Kmeans)
