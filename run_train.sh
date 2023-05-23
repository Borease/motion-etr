offset_mode=quad   # quad/lin/bilin
name=MTR_Gopro_${offset_mode}
blur_direction=reblur    # reblur/deblur

python train.py \
       --name=$name \
       --offset_mode=${offset_mode} \
       --gpu_ids=0,1 \
       --blur_direction=${blur_direction} \
       --dataset_mode=aligned \
       --niter_decay=800 \
       --dataroot='/home/dell/Documents/METHOD/SPT/SPT_Detection/Motion-ETR/Gopro_align_data'\
       --resize_or_crop='crop' \
	--fineSize=256 \
	--save_latest_freq=100 \
	--print_freq=20
