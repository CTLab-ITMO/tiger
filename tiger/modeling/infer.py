from utils import parse_args, create_logger, fix_random_seed, DEVICE

from dataset import BaseDataset
from dataloader import BaseDataloader
from models import BaseModel, TorchModel
from metric import BaseMetric, StatefullMetric

import json
import numpy as np
import torch
import datetime


logger = create_logger(name=__name__)
seed_val = 42


def inference(dataloader, model, metrics, pred_prefix, labels_prefix):
    running_metrics = {}
    for metric_name, metric_function in metrics.items():
        running_metrics[metric_name] = []

    if isinstance(model, TorchModel):
        model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):

            for key, value in batch.items():
                batch[key] = value.to(DEVICE)
            batch[pred_prefix] = model(batch)

            for key, values in batch.items():
                batch[key] = values.cpu()

            for metric_name, metric_function in metrics.items():
                running_metrics[metric_name].extend(metric_function(
                    inputs=batch,
                    pred_prefix=pred_prefix,
                    labels_prefix=labels_prefix,
                ))
            
        for metric_name, metric_function in metrics.items():
            if isinstance(metric_function, StatefullMetric):
                running_metrics[metric_name] = metric_function.reduce(running_metrics[metric_name])

    logger.debug('Inference procedure has been finished!')
    logger.debug('Metrics are the following:')
    for metric_name, metric_value in running_metrics.items():
        logger.info('{}: {}'.format(metric_name, np.mean(metric_value)))


def main():
    fix_random_seed(seed_val)
    config = parse_args()

    logger.debug('Inference config: \n{}'.format(json.dumps(config, indent=2)))

    dataset = BaseDataset.create_from_config(config['dataset'])

    _, _, eval_dataset = dataset.get_samplers()

    eval_dataloader = BaseDataloader.create_from_config(
        config['dataloader']['validation'],
        dataset=eval_dataset
    )

    model = BaseModel.create_from_config(config['model'], **dataset.meta)

    if isinstance(model, TorchModel):
        model = model.to(DEVICE)
        checkpoint_path = '../checkpoints/{}_final_state.pth'.format(config['experiment_name'])
        model.load_state_dict(torch.load(checkpoint_path))

    metrics = {
        metric_name: BaseMetric.create_from_config(metric_cfg , **dataset.meta)
        for metric_name, metric_cfg in config['metrics'].items()
    }

    _ = inference(
        dataloader=eval_dataloader, 
        model=model, 
        metrics=metrics, 
        pred_prefix=config['pred_prefix'], 
        labels_prefix=config['label_prefix']
    )


if __name__ == '__main__':
    main()
