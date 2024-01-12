import cv2 
import albumentations as A 
import random 



# crop, resize, flip 
def basic_transform(aim_size):
	return A.Compose([
		A.OneOf([
		A.RandomResizedCrop(aim_size[0], aim_size[1], scale=(0.3, 0.7), ratio=(0.85, 1.15), interpolation=1, always_apply=True),
		# A.RandomSizedCrop(min_max_height = [aim_size[0] * 0.4, aim_size[0] * 0.8], \
		# height = aim_size[0], width = aim_size[1], w2h_ratio = 1, always_apply = True),
		A.CropAndPad(percent= -0.2, pad_mode = 0, keep_size = True, always_apply = True)
		]),
		A.OneOf([
			A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=1),
			A.VerticalFlip(p=1),
			A.HorizontalFlip(p=1),
		])
		],
		additional_targets={'image0': 'image', 'image1': 'image'}		
		)


def shadow_transform():

    return A.Compose([
        A.OneOf([
            # A.RandomRain(p=0.4, brightness_coefficient=0.99, blur_value=3),
			A.RandomToneCurve(scale=0.1, p = 1),
            A.RandomSunFlare(angle_lower = 0, angle_upper=0.8, num_flare_circles_lower = 2, num_flare_circles_upper = 3, flare_roi=(0, 0, 1, 0.2),src_radius=200, src_color=(224, 242, 150), p=1),
			A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.5, p = 1),
        ], p =0.8),
        A.OneOf([
            A.GaussNoise(p =0.5, mean = 10),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.MultiplicativeNoise(multiplier =(0.9, 1.1), per_channel = True, elementwise = True,  p=0.5)
        ]),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, always_apply=False, p=0.8),
		
	])


def autumn_transform():
    return A.Compose(
    [
		A.OneOf([
		A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
		A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=1),
		A.InvertImg(p=1)
		]),
		A.OneOf([
			A.ChannelShuffle(p = 0.5),
			A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
		]),
		A.Spatter(mean=0.65, std=0.3, gauss_sigma=2, cutout_threshold=0.68, intensity=0.6, mode='mud', always_apply=False, p=0.5)

    ])

def light_transform():
	# become yellow color 
	# have the heavy dark shadow 
	return A.Compose(
	[	
		A.RandomSunFlare(angle_lower = 0, angle_upper=0.8, num_flare_circles_lower = 2, num_flare_circles_upper = 3, flare_roi=(0, 0, 1, 0.2),src_radius=200, src_color=(224, 242, 150), p=0.5),
		A.OneOf([
			# A.ColorJitter(brightness=0.2, contrast=0, saturation=0, hue=0),
			A.ChannelDropout(channel_drop_range = (1, 1), fill_value =83),
			A.HueSaturationValue(hue_shift_limit = (-20, 12), sat_shift_limit= (-18, -3), val_shift_limit = (56, 81)),
			A.RGBShift(r_shift_limit = (62, 126), g_shift_limit = (67, 154), b_shift_limit = (65, 96))
		]),
		A.RandomBrightness(limit = (0.2, 0.4)), 
		A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, always_apply=False, p=0.8),
		A.OneOf([
            A.GaussNoise(p =0.5, mean = 10),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.MultiplicativeNoise(multiplier =(0.9, 1.1), per_channel = True, elementwise = True,  p=0.5),
			A.Downscale(scale_max = 0.8, scale_min = 0.6), # quality
        ]),
	])
