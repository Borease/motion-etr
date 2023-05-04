offset_mode=quad
name=MTRdeblur_${offset_mode}
blur_direction=deblur    # reblur/deblur

python test.py \
       --name=$name \
       --offset_mode=${offset_mode} \
       --gpu_ids=0 \
       --no_crop \
       --blur_direction=${blur_direction} \
       --checkpoints_dir='./pretrain_models' \
       --dataroot='/home/dell/Documents/METHOD/SPT/SPT_Detection/Motion-ETR/Gopro_align_data' # path to images (should have subfolders trainA, trainB, valA, valB, etc) 


# python metrics.py --res_root="./exp_results/$name/test_latest/images/" --ref_root="./exp_results/$name/test_latest/images/"
