# for test images (from any image folder, without image lists )

import _init_paths  # make it as absolute import
import argparse
import os
import models
import torch
import torch.nn.functional as F

from configs import * 
from torchvision import datasets, transforms, io
# from torchvision.utils import draw_segmentation_masks
from typing import Optional, Tuple, List, Union
from utils.utils import FullModel
from utils.criterion import BondaryLoss, OhemCrossEntropy


from configs import constants as C


def input_transform(image):
	mean = [0.485, 0.456, 0.406]
	std=[0.229, 0.224, 0.225]
	image = image.astype(np.float32)[:, :, ::-1]
	image = image / 255.0
	image -= mean
	image /= std
	return image

def parse_args():

    parser = argparse.ArgumentParser(
        description='Run PIDNet model on images in a folder and returns images with color segmentation on it'
    )

    parser.add_argument(
        '--cfg_file_path',
        help='experiment configure file name',
        default = '/home/zlin/PIDNet/configs/Mavis/20231009_pidnet_l_trial5_final_trustweights.yaml',
        type=str
    )

    parser.add_argument(
        '--pt_file_path',
        help='Pytorch best.pt weights for PIDNet',
        default = '/home/zlin/PIDNet/output/20230916_batches1_8_freiburg_linlin/20231009_pidnet_l_trial5_final_trustweights/best_backup.pt',
        type=str
    )

    # parser.add_argument(
    #     '--input-image-list-file',
    #     help='folder containing images to test',
    #     default = '/data/segmentation/zlin_data/pidnet_dataset/generated_dataset/test_images',
    #     type=str
    # )

    parser.add_argument(
        '--input_images_folder',
        help='folder containing images to test',
        default = '/data/segmentation/zlin_data/pidnet_dataset/generated_dataset/test_images',
        type=str
    )

    parser.add_argument(
        '--output_images_folder',
        help='folder to output color segmented images',
        default = '/home/zlin/PIDNet/output/20230916_batches1_8_freiburg_linlin/20231009_pidnet_l_trial5_final_trustweights/test_images',
        type=str
    )


    parser.add_argument(
        'opts',
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()

    return args


def mock_parse_args():
    class MockArgs:
        pass
    return MockArgs()

import numpy as np 
import cv2 


def convert_color(label, color_map):
    temp = np.zeros(label.shape + (3,)).astype(np.uint8)
    # print("labe shape: ", label.shape)
    if isinstance(color_map, dict):
        for k, v in color_map.items():
            # print("v is: ", v, type(v))

            temp[label == k] = v
    else:
        for k, v in enumerate(color_map):
            # print("v is: ", v, type(v))
            temp[label == k] = v
    return temp

def draw_segmentation_masks(
        image: torch.Tensor,
        masks: torch.Tensor,
        alpha: float = 0.8,
        colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
) -> torch.Tensor:

    """
    Draws segmentation masks on given RGB image.
    The values of the input image should be uint8 between 0 and 255.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
        masks (Tensor): Tensor of shape (num_masks, H, W) or (H, W) and dtype bool.
        alpha (float): Float number between 0 and 1 denoting the transparency of the masks.
            0 means full transparency, 1 means no transparency.
        colors (color or list of colors, optional): List containing the colors
            of the masks or single color for all masks. The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
            By default, random colors are generated for each mask.

    Returns:
        img (Tensor[C, H, W]): Image Tensor, with segmentation masks drawn on top.
    """

    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(draw_segmentation_masks)
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"The image must be a tensor, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"The image dtype must be uint8, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim != 3:
        raise ValueError("masks must be of shape (H, W) or (batch_size, H, W)")
    if masks.dtype != torch.bool:
        raise ValueError(f"The masks must be of dtype bool. Got {masks.dtype}")
    if masks.shape[-2:] != image.shape[-2:]:
        raise ValueError("The image and the masks must have the same height and width")

    num_masks = masks.size()[0]
    if colors is not None and num_masks > len(colors):
        raise ValueError(f"There are more masks ({num_masks}) than colors ({len(colors)})")

    if num_masks == 0:
        print("masks doesn't contain any mask. No mask was drawn")
        return image

    if colors is None:
        raise ValueError("no colors set")

    if not isinstance(colors, list):
        colors = [colors]
    if not isinstance(colors[0], (tuple, str)):
        raise ValueError("colors must be a tuple or a string, or a list thereof")
    if isinstance(colors[0], tuple) and len(colors[0]) != 3:
        raise ValueError("It seems that you passed a tuple of colors instead of a list of colors")

    out_dtype = torch.uint8

    colors_ = []
    for color in colors:
        colors_.append(torch.tensor(color, dtype=out_dtype).cuda())

    img_to_draw = image.detach().clone()
    for mask, color in zip(masks, colors_):
        img_to_draw[:, mask] = color[:, None]

    out = image * (1 - alpha) + img_to_draw * alpha
    return out.to(out_dtype)

import glob 
import os 
from PIL import Image
from PIL import ImageFile
import imghdr


ImageFile.LOAD_TRUNCATED_IMAGES = True



if __name__ == '__main__':

    test = True

    args = parse_args()

    cfg_file = args.cfg_file_path
    pt_file = args.pt_file_path
    # input_image_list_file = args.input_image_list_file
    test_image_folder = args.input_images_folder
    output_image_folder = args.output_images_folder
    mask_dir = os.path.join(output_image_folder, 'mask')
    color_dir = os.path.join(output_image_folder, 'color')
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)

    update_config(config, args)

    transform = transforms.Compose([
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])

    # dataset = datasets.ImageFolder(
    #     test_image_folder,
    #     transform=transform
    # )
    # #
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=8,
    #     shuffle=True
    # )

    model_size = config.MODEL.SIZE
    num_of_classes = config.DATASET.NUM_CLASSES
    pidnet_model = models.pidnet.get_pidnet_model(
        model_size=model_size,
        num_of_classes=num_of_classes
    )
    if config.TEST.MODEL_FILE.endswith('pt'):
        pretrained_pt_file_path_str = config.TEST.MODEL_FILE

        model = models.pidnet.load_pretrained_pt_file(
            model=pidnet_model,
            pt_file_path_str=pretrained_pt_file_path_str
        )
    else:
        model = models.pidnet.get_seg_model(config, imgnet_pretrained = False)
    model.augment = False


    model.eval()
    model.cuda()
    try:
        with torch.no_grad():
            for input_image_file_path in glob.glob(os.path.join(test_image_folder, '*.png')):
                image_file_name = input_image_file_path.split("/")[-1]
                try:
                    ori_image = cv2.imread(input_image_file_path)
                except:
                    img = np.array(Image.open(input_image_file_path))

                h, w = ori_image.shape[:2]

                transformed_image = input_transform(ori_image)
                transformed_image = transformed_image.transpose((2, 0, 1)).copy()
                image_tensor = torch.from_numpy(transformed_image).unsqueeze(0).cuda()

                pred = model(
                    image_tensor
                )
            

                pred_mask = F.interpolate(
                    pred,
                    size=(h, w),
                    mode='bilinear',
                    align_corners=True
                )

                # pred_mask = 1, 10, 1028, 1920
                pred_mask_t_gpu = torch.argmax(pred_mask, dim=1).squeeze()
                pred_mask_t_cpu = np.array(pred_mask_t_gpu.squeeze().detach().cpu())

                pred_color = convert_color(pred_mask_t_cpu, C.LABEL_TO_COLOR)
                

                # booleanMask = (
                #     pred_mask_t_gpu == torch.arange(10, device="cuda:0")[:, None, None, None]
                # )
                # # 10, 1, 1080, 1920
                # pred_b_mask_t_gpu = booleanMask.transpose(1, 0)
                # # 1, 10, 1080, 1920

                # # image_uint8_gpu = image_f_gpu.to(torch.uint8)

                # seg_image_t = draw_segmentation_masks(
                #     image=image_uint8_gpu,
                #     masks=pred_b_mask_t_gpu.squeeze(0),
                #     alpha=0.6,
                #     colors=class_color_list
                # )


                cv2.imwrite(
                os.path.join(color_dir, image_file_name),
                pred_color            
                )
                    
                cv2.imwrite(
                    os.path.join(mask_dir, image_file_name),
                    pred_mask_t_cpu
                )
                    
    except:
        print('test_image_file', input_image_file_path)
                    
                
                