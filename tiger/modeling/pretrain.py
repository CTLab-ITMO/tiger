import utils
from utils import parse_args, create_logger, fix_random_seed, DEVICE

from dataset import BaseDataset
from dataloader import BaseDataloader
from models import BaseModel
from optimizer import BaseOptimizer
from loss import BaseLoss
from callbacks import BaseCallback

import copy
import json
import torch

logger = create_logger(name=__name__)
seed_val = 42


def pretrain(dataloader, model, optimizer, loss_function, callback, epoch_cnt):
    step_num = 0
    best_checkpoint = None

    logger.debug('Start pretraining...')

    for epoch in range(epoch_cnt):
        logger.debug(f'Start epoch {epoch}')
        for step, batch in enumerate(dataloader):
            model.train()

            for key, values in batch.items():
                batch[key] = batch[key].to(DEVICE)

            batch.update(model.pretrain(batch))
            loss = loss_function(batch)

            optimizer.step(loss)
            callback(batch, step_num)
            step_num += 1

            best_checkpoint = copy.deepcopy(model.state_dict())

    logger.debug('Pretraining procedure has been finished!')
    return best_checkpoint


def main():
    fix_random_seed(seed_val)
    config = parse_args()

    utils.tensorboards.GLOBAL_TENSORBOARD_WRITER = \
        utils.tensorboards.TensorboardWriter(config['experiment_name'])

    logger.debug('Training config: \n{}'.format(json.dumps(config, indent=2)))

    dataset = BaseDataset.create_from_config(config['dataset'])

    train_sampler, test_sampler = dataset.get_samplers()

    train_dataloader = BaseDataloader.create_from_config(
        config['dataloader']['train'],
        dataset=train_sampler,
        **dataset.meta
    )

    validation_dataloader = BaseDataloader.create_from_config(
        config['dataloader']['validation'],
        dataset=test_sampler,
        **dataset.meta
    )

    model = BaseModel.create_from_config(config['model'], **dataset.meta).to(DEVICE)

    loss_function = BaseLoss.create_from_config(config['loss'])

    optimizer = BaseOptimizer.create_from_config(config['optimizer'], model=model)

    callback = BaseCallback.create_from_config(
        config['callback'],
        model=model,
        dataloader=validation_dataloader,
        optimizer=optimizer
    )

    logger.debug('Everything is ready for pretraining process!')

    # Pretrain process
    pretrain(
        dataloader=train_dataloader,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        callback=callback,
        epoch_cnt=config['pretrain_epochs_num']
    )

    logger.debug('Saving model...')
    checkpoint_path = '../checkpoints/pretrain_{}_final_state.pth'.format(config['experiment_name'])
    torch.save(model.state_dict(), checkpoint_path)
    logger.debug('Saved model as {}'.format(checkpoint_path))


if __name__ == '__main__':
    main()
