# Semantic IDs Generation and TIGER Training

## Overview

расписать бы здесь

### TIGER Model Training

```bash
cd /.../tiger/tiger
python train_tiger.py --params ../configs/tiger_train_config.json
```
### SASRec Model Training

```bash
cd /.../tiger/tiger
python train_sasrec.py --params ../configs/sasrec_train_config.json
```

## Monitoring and Outputs

### Training Logs
```bash
tensorboard --logdir /.../tiger/tensorboard_logs
```