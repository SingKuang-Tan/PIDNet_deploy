python3.8 /home/zlin/PIDNet/tools/test_video.py --input /home/zlin/PIDNet/test_videos/the_ultimate_test_video.mp4 --cfg_file_path /home/zlin/PIDNet/configs/Mavis/20231009_pidnet_l_trial5_final_trustweights.yaml 2>&1 | tee output.txt \
&& python3.8 /home/zlin/PIDNet/tools/test_video.py --input /home/zlin/PIDNet/test_videos/coastal_0.mp4 --cfg_file_path /home/zlin/PIDNet/configs/Mavis/20231009_pidnet_l_trial5_final_trustweights.yaml \
&& python3.8 /home/zlin/PIDNet/tools/test_video.py --input /home/zlin/PIDNet/test_videos/front-2023-04-12-10-35-00.mp4 --cfg_file_path /home/zlin/PIDNet/configs/Mavis/20231009_pidnet_l_trial5_final_trustweights.yaml \
&& python3.8 debug/val_miou.py --cfg_file_path /home/zlin/PIDNet/configs/Mavis/20231009_pidnet_l_trial5_final_trustweights.yaml 

