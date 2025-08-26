import itertools
import json
import random
import torch

import utils
from utils import parse_args, create_logger, DEVICE, Params, dict_to_str, fix_random_seed

from train import train
from infer import inference

from callbacks import BaseCallback, EvalCallback, ValidationCallback
from dataset import BaseDataset
from dataloader import BaseDataloader
from loss import BaseLoss
from models import BaseModel
from optimizer import BaseOptimizer

logger = create_logger(name=__name__)
seed_val = 42


def main():
    fix_random_seed(seed_val)
    config = parse_args()

    logger.debug('Training config: \n{}'.format(json.dumps(config, indent=2)))

    dataset_params = Params(config['dataset'], config['dataset_params'])
    model_params = Params(config['model'], config['model_params'])
    loss_function_params = Params(config['loss'], config['loss_params'])
    optimizer_params = Params(config['optimizer'], config['optimizer_params'])

    logger.debug('Everything is ready for training process!')

    start_from = config.get('start_from', 0)
    num = config.get('num_exps', None)

    list_of_params = list(itertools.product(
        dataset_params,
        model_params,
        loss_function_params,
        optimizer_params
    ))

    if num is None:
        num = len(list_of_params)
    else:
        random.shuffle(list_of_params)

    cnt = 0

    for dataset_param, model_param, loss_param, optimizer_param in list_of_params[start_from:num]:
        cnt += 1
        if cnt < start_from:
            continue

        model_name = '_'.join([
            config['experiment_name'],
            dict_to_str(dataset_param, config['dataset_params']),
            dict_to_str(model_param, config['model_params']),
            dict_to_str(loss_param, config['loss_params']),
            dict_to_str(optimizer_param, config['optimizer_params'])
        ])

        logger.debug('Starting {}'.format(model_name))

        dataset = BaseDataset.create_from_config(dataset_param)

        train_sampler, validation_sampler, eval_sampler = dataset.get_samplers()

        train_dataloader = BaseDataloader.create_from_config(
            config['dataloader']['train'],
            dataset=train_sampler,
            **dataset.meta
        )

        validation_dataloader = BaseDataloader.create_from_config(
            config['dataloader']['validation'],
            dataset=validation_sampler,
            **dataset.meta
        )

        eval_dataloader = BaseDataloader.create_from_config(
            config['dataloader']['validation'],
            dataset=eval_sampler,
            **dataset.meta
        )

        if utils.tensorboards.GLOBAL_TENSORBOARD_WRITER is not None:
            utils.tensorboards.GLOBAL_TENSORBOARD_WRITER.close()
        utils.tensorboards.GLOBAL_TENSORBOARD_WRITER = utils.tensorboards.TensorboardWriter(model_name, use_time=False)

        model = BaseModel.create_from_config(model_param, **dataset.meta).to(DEVICE)
        loss_function = BaseLoss.create_from_config(loss_param)
        optimizer = BaseOptimizer.create_from_config(optimizer_param, model=model)

        callback = BaseCallback.create_from_config(
            config['callback'],
            model=model,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            eval_dataloader=eval_dataloader,
            optimizer=optimizer,
            **dataset.meta
        )

        best_model_checkpoint = train(
            dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            callback=callback,
            epoch_cnt=config.get('train_epochs_num'),
            best_metric=config.get('best_metric')
        )

        eval_model = BaseModel.create_from_config(model_param, **dataset.meta).to(DEVICE)
        eval_model.load_state_dict(best_model_checkpoint)

        for cl in callback._callbacks:
            if isinstance(cl, EvalCallback):
                metrics = cl._metrics
                pred_prefix = cl._pred_prefix
                labels_prefix = cl._labels_prefix
                break
        else:
            for cl in callback._callbacks:
                if isinstance(cl, ValidationCallback):
                    metrics = cl._metrics
                    pred_prefix = cl._pred_prefix
                    labels_prefix = cl._labels_prefix
                    break
            else:
                assert False

        inference(eval_dataloader, eval_model, metrics, pred_prefix, labels_prefix)

        logger.debug('Saving best model checkpoint...')
        checkpoint_path = '../checkpoints/{}_final_state.pth'.format(model_name)
        torch.save(best_model_checkpoint, checkpoint_path)
        logger.debug('Saved model as {}'.format(checkpoint_path))


if __name__ == '__main__':
    main()
