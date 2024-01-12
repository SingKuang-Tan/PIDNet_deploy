import os
import yaml
import attrdict

import cv2
import numpy as np
from icecream import ic
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


import _init_paths
from models.pidnet import get_seg_model

from icecream import ic


# https://github.com/jacobgil/pytorch-grad-cam
# refer: https://jacobgil.github.io/pytorch-gradcam-book/Class%20Activation%20Maps%20for%20Semantic%20Segmentation.html


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda")
torch.set_grad_enabled(True)

# construct the model with pretrained weights
config_file = "/home/zlin/PIDNet/configs/Mavis/20230920_pidnet_l_trial5_trustweights.yaml"


with open(config_file, "r") as f:
    cfg = attrdict.AttrDict(yaml.safe_load(f))

model = get_seg_model(cfg, imgnet_pretrained=False)
model.augment = False
model_state_file = cfg.TEST.MODEL_FILE
pretrained_dict = torch.load(model_state_file)
if "state_dict" in pretrained_dict:
    pretrained_dict = pretrained_dict["state_dict"]
model_dict = model.state_dict()
pretrained_dict = {
    k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict.keys()
}

# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)
model.load_state_dict(pretrained_dict)
ic(model)


model.eval().to(device)
model.requires_grad = False


def prep_frame(fr):
    # prepare a frame for inference

    image = fr.astype(np.float32)[:, :, ::-1] / 255.0
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)


def deconstruct_output(pred, orig_size, as_numpy=False):
    # resize to original size
    pred = F.interpolate(pred, size=orig_size, mode="bilinear", align_corners=False)
    pred = torch.argmax(pred, dim=1).squeeze(0)
    if as_numpy:
        pred = pred.cpu().detach().numpy()
    return pred


class PIDNetWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        result = self.model(x)
        return F.interpolate(
            result, size=[1024, 1024], mode="bilinear", align_corners=False
        )


model = PIDNetWrapper(model)


class SegmentationTarget:
    def __init__(self, cls, mask):
        self.cls = cls
        self.mask = torch.from_numpy(mask).to(device)

    def __call__(self, out):
        ic(out.shape)
        ic(self.mask.shape)
        return (out[self.cls, :, :] * self.mask).sum()


# class map
cls_map = {
    0: "void",
    1: "obstacle",
    2: "navigable_space",
    3: "vine_canopy",
    4: "vine_stem",
    5: "vine_pole",
    6: "vegetation",
    7: "tractor_hood",
    8: "sky",
    9: "human",
    10: "vehicle",
}
txt_to_class = {v: k for k, v in cls_map.items()}


use_video = True
# run on a video steam
if use_video:
    cap = cv2.VideoCapture("/home/zlin/PIDNet/test_videos/the_ultimate_test_video.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 669)
    _, og_frame = cap.read()


import glob

layers_dict = {
    "layer3": model.model.layer3,
    "layer5": model.model.layer5,
    "layer5_d": model.model.layer5_d,
    "layer5_p": model.model.layer5_,
    "ppm": model.model.spp,
    "bag": model.model.dfm,
}
subfolder_name = 'trial3_5'
os.makedirs(f'/home/zlin/PIDNet/debug/{subfolder_name}', exist_ok=True)
for layer_name, layer in layers_dict.items():
    for file_path in glob.glob("/home/zlin/PIDNet/test_images/*.png"):
        ic(file_path)
        np_frame = cv2.imread(file_path)
        np_frame = cv2.resize(np_frame, (1024, 1024), interpolation=cv2.INTER_AREA)
        orig_size = np_frame.shape[:-1]

        ic(np_frame.shape)
        basename = os.path.basename(file_path).split(".")[0]
        output_dir = f'{cfg.OUTPUT_DIR}/{cfg.DATASET.DATASET}/{cfg.MODEL.NAME}'
        grayscale_dir = f"{output_dir}/grayscale/{basename}"
        mask_dir = f"{output_dir}/mask/{basename}"
        density_dir = f"{output_dir}/density/{basename}"

        os.makedirs(grayscale_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(density_dir, exist_ok=True)

        frame = prep_frame(np_frame)

        result = model.forward(frame.to(device))
        ic(result.shape)

        # postprocess
        # result = result[1]
        out = deconstruct_output(result, orig_size)
        out = out.cpu().detach().numpy()
        ic(out.shape)

        for desired_cls in [1, 9, 10]:
        #     print(f"Running GradCAM for class {desired_cls}")

        #     # read frames
        #     # get class mask
        #     mask = 255 * np.uint8(out == desired_cls)
        #     ic(mask.shape)
        #     mask_rgb = np.repeat(mask[:, :, None], 3, axis=-1)
        #     # if True:
        #     #     # visualize class mask
        #     #     # cv2.imshow('mask', np.hstack((og_frame, mask_rgb)))
        #     #     # key = cv2.waitKey(0)
        #     #     # # cv2.destroyWindow('r')
        #     #     # if key == ord('c'):
        #     #     #     pass

        #     cv2.imwrite(
        #         f"{mask_dir}/mask_{layer_name}_{desired_cls}.png",
        #         np.concatenate((np_frame, mask_rgb), axis=-2),
        #     )

            # run gradcam and visualize output
            targets = [SegmentationTarget(desired_cls, np.float32(out == desired_cls))]
            cam = GradCAM(
                model=model,
                target_layers=[layer],
                use_cuda=device == torch.device("cuda"),
            )
            grayscale_cam = cam(input_tensor=frame, targets=targets)[
                0, : 
            ]  # (1080, 1920)
            # cv2.imwrite(
            #     f"{grayscale_dir}/grayscale_{layer_name}_{desired_cls}.png",
            #     grayscale_cam,
            # )

            ic(grayscale_cam.shape)
            output = show_cam_on_image(
                np_frame.astype(np.float32) / 255.0, grayscale_cam, use_rgb=False
            )
            # if True:
            # cv2.imshow('grayscale', grayscale_cam)
            # if cv2.waitKey(0) == ord('c'):
            #     pass

            cv2.imwrite(f"{density_dir}/sample_{layer_name}_{desired_cls}.png", output)
