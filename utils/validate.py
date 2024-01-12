# import numpy as np
import torch
from torch.nn import functional as F
from torchmetrics import ConfusionMatrix
from utils import (
    AverageMeter,
    get_confusion_matrix,
    FullModel
)

import datasets
import models
from criterion import (
    # CrossEntropy,
    OhemCrossEntropy,
    BondaryLoss
)


def validate(valid_loader, model):
    try:
        # no backward
        model.to("cuda:0")
        model.eval()
        ave_loss = AverageMeter()

        # confusion_np_m = (10, 10, 2)
        # confusion_matrix = np.zeros(
        #     (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums)
        # )
        num_classes = 10
        cm = ConfusionMatrix(
            num_classes=num_classes
        ).to("cuda:0")

        with torch.no_grad():

            for idx, batch in enumerate(valid_loader):
                # print("0")
                image, label, bd_gts, a, b = batch
                image = image.cuda()
                label = label.long().cuda()
                bd_gts = bd_gts.float().cuda()

                losses, pred, _, _ = model(image, label, bd_gts)
                # print("1")

                if isinstance(pred, (list, tuple)):
                    pred_list = pred
                else:
                    pred_list = [pred]
                last_pred = pred_list[-1]
                cm.update(
                    last_pred,
                    label
                )

                loss = losses.mean()
                ave_loss.update(loss.item())

        cm_t = cm.compute()
        pos = torch.sum(cm_t, 1)
        res = torch.sum(cm_t, 0)
        I = torch.diagonal(cm_t)
        U = pos + res - I
        IoU_nan = I/U
        mean_IoU = torch.nanmean(IoU_nan).cpu().item()
        return ave_loss.average(), mean_IoU
            # , IoU_tensor
    except Exception as e:
        print("Exception in validate:")
        raise e


def setup_pidnet_model():
    num_of_classes = 10

    pidnet_model = models.pidnet.get_pidnet_model(
        model_size='medium',
        num_of_classes=num_of_classes
    )

    pretrained_pt_file_path_str = 'pretrained_models/best_mizba.pt'
    model = models.pidnet.load_pretrained_pt_file(
        model=pidnet_model,
        pt_file_path_str=pretrained_pt_file_path_str
    )

    sem_criterion = OhemCrossEntropy(
        ignore_label=255,
        thres=0.9,
        min_kept=131072,
        weight=None
    )

    bd_criterion = BondaryLoss()

    model = FullModel(model, sem_criterion, bd_criterion)

    return model


def setup_valid_loader():
    validation_dataset = datasets.Mavis(
        root="data/MavisLocal",
        list_path="validation.txt",
        num_classes=10,
        multi_scale=False,
        flip=False,
        ignore_label=255,
        base_size=2048,
        crop_size=(1024, 1024)
    )

    valid_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False
    )

    return valid_loader


if __name__ == "__main__":
    import os
    os.chdir("..")
    model = setup_pidnet_model()
    valid_loader = setup_valid_loader()
    results = validate(
        valid_loader=valid_loader,
        model=model
    )
    print(results)