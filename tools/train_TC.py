import argparse
import os
import pprint
import pandas as pd
import logging
import timeit
import time
import icecream as ic


import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

import _init_paths  # make it as absolute import
import models
import datasets
from torch.utils.data import WeightedRandomSampler, ConcatDataset


from configs import *  # import config
from utils.criterion import *
from utils.function import train_TC as train, validate
from utils.utils import create_logger, FullModel, init_seeds

from modified_model import ModifiedModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train segmentation network with TC loss"
    )
    parser.add_argument("--cfg_file_path", default="", type=str)
    parser.add_argument("--seed", type=int, default=304)
    parser.add_argument(
        "opts", help="modify the config options", nargs=argparse.REMAINDER
    )
    args = parser.parse_args()
    update_config(config, args)
    return args


def main():
    args = parse_args()
    loss_df = pd.DataFrame()
    cfg_file_path = args.cfg_file_path
    seed = args.seed
    init_seeds(seed)
    torch.multiprocessing.set_sharing_strategy('file_descriptor')

    torch.cuda.empty_cache()

    logger, model_output_dir, tb_dir = create_logger(
        config, args.cfg_file_path, "train_tc"
    )
    logger.info(pprint.pformat(config))
    logger.info(f"In TC loss alfa is adjusted to {config.LOSS.ALFA}")

    writer_dict = {
        "writer": SummaryWriter(tb_dir),
        "train_global_steps": 0,
        "valid_global_steps": 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED


    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    gpus = list(config.GPUS)
    if torch.cuda.device_count() != len(gpus):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in gpus])
    logger.info('Current GPU devices: ' + ','.join([str(g) for g in gpus]))
        # raise Exception("The gpu numbers do not match!")

    model_size = config.MODEL.SIZE
    num_of_classes = config.DATASET.NUM_CLASSES

    pidnet_model = models.pidnet2.get_pidnet_model(
        model_size=model_size, num_of_classes=num_of_classes  # tc loss
    )

    pretrained_pt_file_path_str = config.MODEL.PRETRAINED
    model = models.pidnet2.load_pretrained_pt_file(
        model=pidnet_model, pt_file_path_str=pretrained_pt_file_path_str
    )

    #Modified the model
    #model=ModifiedModel(model)


    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)

    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = datasets.Mavis_TC(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TRAIN_SET,
        num_classes=config.DATASET.NUM_CLASSES,
        class_weights=config.TRAIN.CLASS_WEIGHTS,
        mean=config.TRAIN.MEAN,
        std=config.TRAIN.STD,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        do_augment=True,
        aug_human=True,
        do_test=False,
    )
    train_freiburg_dataset = datasets.Mavis_TC(
        root=config.DATASET.ROOT,
        list_path = config.DATASET.TRAIN_FREIBURG_SET,
        num_classes=config.DATASET.NUM_CLASSES,
        class_weights=config.TRAIN.CLASS_WEIGHTS,
        mean=config.TRAIN.MEAN,
        std=config.TRAIN.STD,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        do_augment=True,
        aug_human=True,
        do_test=False,
    )

    train_concat_dataset = ConcatDataset([train_dataset, train_freiburg_dataset])

    # Define the trust weights for each dataset
    trust_weights = config.TRAIN.TRUST_WEIGHTS 
    assert len(trust_weights) == 2, 'trust weights format should be a list of 2 elements'


    train_class_weights = torch.concat([train_dataset.class_weights.unsqueeze(0), train_freiburg_dataset.class_weights.unsqueeze(0)], \
                                       dim = 0)
    total_samples = torch.tensor([len(train_dataset), len(train_freiburg_dataset)]) * torch.tensor(trust_weights)
    train_class_weights = 1/(torch.matmul(total_samples, 1/train_class_weights))

    train_class_weights = train_class_weights/torch.sum(train_class_weights)
    
    # Define the sampler for the concatenated dataset
    weights = [trust_weights[0] if i < len(train_dataset) else trust_weights[1] for i in range(len(train_concat_dataset))]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_concat_dataset))
    logging.info(f'train_concat_dataset length: {len(train_concat_dataset)}')

    # Define the data loader for the concatenated dataset
    logging.info(f"training class weight: {train_class_weights}")


    train_loader = torch.utils.data.DataLoader(
        train_concat_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=True,
        sampler = sampler
    )

    validate_dataset = datasets.Mavis_TC(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.VALID_SET,
        num_classes=config.DATASET.NUM_CLASSES,
        class_weights=train_class_weights,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        mean=config.VALID.MEAN,
        std=config.VALID.STD,
        do_augment=False,
        aug_human=False,
        do_test=False,
    )

    validate_loader = torch.utils.data.DataLoader(
        validate_dataset,
        batch_size=config.VALID.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=True,
    )

    # criterion
    if config.LOSS.USE_OHEM:
        sem_criterion = OhemCrossEntropy(
            ignore_label=config.TRAIN.IGNORE_LABEL,
            thres=config.LOSS.OHEMTHRES,
            min_kept=config.LOSS.OHEMKEEP,
            weight=train_dataset.class_weights,
        )
    elif config.LOSS.USE_FocalTverskyLoss:
        sem_criterion = FocalTverskyLoss(
            smooth=1,
            alpha=0.7,
            beta=0.3,
            gamma=0.75,
            num_classes=config.DATASET.NUM_CLASSES,
            ignore_index=config.TRAIN.IGNORE_LABEL,
        )
    else:
        sem_criterion = CrossEntropy(
            ignore_label=config.TRAIN.IGNORE_LABEL, weight=train_dataset.class_weights
        )

    bd_criterion = BondaryLoss()

    model = FullModel(model, sem_criterion, bd_criterion).cuda()
    #torch._dynamo.config.suppress_errors = True
    #model = FullModel(model, sem_criterion, bd_criterion)
    #model = torch.compile(model)
    #model=model.cuda()

    if len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=gpus).cuda()

    # optimizer
    if config.TRAIN.OPTIMIZER == "sgd":
        params_dict = dict(model.named_parameters())
        params = [{"params": list(params_dict.values()), "lr": config.TRAIN.LR}]

        optimizer = torch.optim.SGD(
            params,
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WD,
            nesterov=config.TRAIN.NESTEROV,
        )
    elif config.TRAIN.OPTIMIZER == "adam":
        params_dict = dict(model.named_parameters())
        params = [{"params": list(params_dict.values()), "lr": config.TRAIN.LR}]

        optimizer = torch.optim.Adam(
            params,
            lr=config.TRAIN.LR,
            betas=(0.9, 0.999),
            eps=1e-07,
            weight_decay=config.TRAIN.WD,
            amsgrad=False,
        )
    else:
        raise ValueError("Only Support SGD optimizer")

    epoch_iters = int(
        train_concat_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus)
    )
    assert epoch_iters == len(train_loader), f'{epoch_iters} {len(train_loader)} {train_concat_dataset.__len__()}'

    best_mIoU = 0
    last_epoch = 0
    flag_rm = ".pth.tar" in config.TRAIN.RESUME_CHECKPOINT_PATH
    lr = config.TRAIN.LR

    if flag_rm:
        model_state_file = config.TRAIN.RESUME_CHECKPOINT_PATH
        if os.path.exists(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={"cuda:0": "cpu"})
        
            # ic(model.module.model.state_dict().keys())
            best_mIoU = checkpoint["best_mIoU"]
            last_epoch = checkpoint["epoch"]
            dct = checkpoint["state_dict"]
            # ic(dct.keys())

            model.module.model.load_state_dict(
                {
                    k.replace("model.", ""): v
                    for k, v in dct.items()
                    if k.startswith("model.")
                }
            )
            optimizer.load_state_dict(checkpoint["optimizer"])
            # lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"=> loaded checkpoint (epoch {checkpoint['epoch']}, lr {lr}, best_mIoU: {best_mIoU})"
            )

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    real_end = end_epoch

    for epoch in range(last_epoch, real_end):
        current_trainloader = train_loader
        if current_trainloader.sampler is not None and hasattr(
            current_trainloader.sampler, "set_epoch"
        ):
            current_trainloader.sampler.set_epoch(epoch)
        start_train_time = timeit.default_timer()

        tc_array, loss_array = train(
            config=config,
            epoch=epoch,
            num_epoch=config.TRAIN.END_EPOCH,
            epoch_iters=epoch_iters,
            base_lr=config.TRAIN.LR,
            num_iters=num_iters,
            trainloader=train_loader,
            optimizer=optimizer,
            model=model,
            writer_dict=writer_dict,
        )

        loss_df["TC_" + str(epoch)] = tc_array
        loss_df["Other_" + str(epoch)] = loss_array
        loss_df.to_csv(os.path.join(model_output_dir, "loss.csv"), index=False)

        end_train_time = timeit.default_timer()
        logger.info(f"seconds to train:{end_train_time-start_train_time}")

        for _ in range(1):
            start_valid_time = timeit.default_timer()
            logger.info("go to validate")
            valid_loss, metrics, cm = validate(
                config=config,
                testloader=validate_loader,
                model=model,
                writer_dict=writer_dict,
            )
            mean_IoU = metrics["AVG_IOU"]
            logger.info(cm)
            end_valid_time = timeit.default_timer()
            logger.info(f"seconds to valid: {end_valid_time - start_valid_time}")
            logger.info(f"metrices: {metrics}")
            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                best_weights_file_path_str = os.path.join(model_output_dir, "best.pt")
                torch.save(model.module.state_dict(), best_weights_file_path_str)
                logging.info(f"update best.pt with mean_IoU = {mean_IoU}")

                checkpoint_file_path_str = os.path.join(
                    model_output_dir, "best.pth.tar"
                )
                logger.info(
                    f"=> saving checkpoint to {checkpoint_file_path_str} every epochs"
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "best_mIoU": best_mIoU,
                        "state_dict": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr": optimizer.param_groups[-1]["lr"],
                    },
                    checkpoint_file_path_str,
                )
                logging.info(f"update best.pth.tar with mean_IoU = {mean_IoU}")
            msg = f"Loss: {valid_loss:.3f}, MeanIoU: {mean_IoU: 4.4f}, Best_mIoU: {best_mIoU: 4.4f}"
            logging.info(msg)

        if epoch % 1 == 0:
            checkpoint_file_path_str = os.path.join(
                model_output_dir, "checkpoint.pth.tar"
            )
            logger.info(f"saving checkpoint to {checkpoint_file_path_str}")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "best_mIoU": best_mIoU,
                    "state_dict": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr": optimizer.param_groups[-1]["lr"],
                },
                checkpoint_file_path_str,
            )
    final_weights_file_path = os.path.join(model_output_dir, "final_state.pt")
    torch.save(model.module.state_dict(), final_weights_file_path)

    writer_dict["writer"].close()
    end = timeit.default_timer()
    duration = (end - start) / 3600
    logger.info("Hours:")
    logger.info(duration)
    logger.info("Done")


if __name__ == "__main__":
    main()
