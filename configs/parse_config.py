import configs.default_config as default_config
from pathlib import Path
import logging
import time


# def update_default_config(
#     default_cfg: default_config,
#     cfg_file_path: str
# ):
#     default_cfg.defrost()
#     default_cfg.merge_from_file(cfg_file_path)
#     # default_cfg.merge_from_list(args.opts)
#     default_cfg.freeze()
#     return default_cfg


def make_model_time_dir(
    cfg: default_config,
    # make_dir: bool
) -> Path:
    """
    Parse output_dir
    make dir if not found
    """
    output_dir_path = make_dir(cfg.OUTPUT_DIR)

    model_name = cfg.MODEL.NAME
    time_str = time.strftime('%Y-%m-%d-%H-%M')

    model_time_dir_path = output_dir_path / model_name / time_str
    model_time_dir_path.mkdir(parents=True, exist_ok=True)

    return model_time_dir_path


def make_dir(dir_path_str:str) -> Path:
    dir_path = Path(dir_path_str)
    if not dir_path.exists():
        print(f'{dir_path_str} not found')
        print(f'Creating directory {dir_path_str}')
    return dir_path

def make_tensorboard_log_dir(
    model_time_path_obj: Path
) -> Path:
    tensorboard_log_dir = model_time_path_obj / 'tb'
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    return tensorboard_log_dir


def make_logger(
    model_time_path_obj: Path,
    log_file_name: str
):
    """
    make logger
    """
    log_file_path_obj = model_time_path_obj / log_file_name

    head = '%(asctime)-15s %(message)s'

    logging.basicConfig(
        filename=str(log_file_path_obj),
        format=head
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger
