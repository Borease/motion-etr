offset_mode=quad
name=simPSF_20230517_PSNR22p61_${offset_mode}
blur_direction=reblur    # reblur/deblur

# python test.py \
#        --name=$name \
#        --offset_mode=${offset_mode} \
#        --gpu_ids=0 \
#        --no_crop \
#        --blur_direction=${blur_direction} \
#        --dataset_mode='aligned' \
#        --checkpoints_dir='./pretrain_models' \
#        --dataroot='/home/dell/Documents/METHOD/SPT/SPT_Detection/Motion-ETR/simPSF_quad_test_input' #final path to image folder will be join(dataroot,phase)

# python metrics.py --res_root="./exp_results/$name/test_latest/images" --ref_root="./exp_results/$name/test_latest/images"

python visualization.py \
       --name=$name \
       --offset_mode=${offset_mode} \
       --gpu_ids=0 \
       --no_crop \
       --blur_direction=${blur_direction} \
       --dataset_mode='aligned' \
       --offset_grid=17 \
       --set_how_many=True \
       --how_many=8 \
       --visualize_traj=False  \
       --checkpoints_dir='./pretrain_models' \
       --results_dir='/mnt/f58069a5-1cf3-43b8-bb9b-ea74327327c9/WZH-DataCenter/PROCESS-SPT/2022/20221109-10_PA646_U2OS_Xlone_FOXA2-Halo_testPSFExposurePixelSize/SNR_20/20221114_Cell01_FOXA2-Halo_10uMPA646_30p5ms_2kframe_03/MotionBlur_ImgPair/test_0518_1018AM/prediction_testing4' \
       --dataroot='/mnt/f58069a5-1cf3-43b8-bb9b-ea74327327c9/WZH-DataCenter/PROCESS-SPT/2022/20221109-10_PA646_U2OS_Xlone_FOXA2-Halo_testPSFExposurePixelSize/SNR_20/20221114_Cell01_FOXA2-Halo_10uMPA646_30p5ms_2kframe_03/MotionBlur_ImgPair/test_0518_1018AM' #final path to image folder will be join(dataroot,phase)
       # --results_dir='/home/dell/Documents/METHOD/SPT/SPT_Detection/Motion-ETR/exp_results/simPSF_20230510'\
       # --dataroot='/home/dell/Documents/METHOD/SPT/SPT_Detection/Motion-ETR/exp_inputs/simPSF_20230510'
       # --results_dir='/home/dell/Documents/METHOD/SPT/SPT_Detection/Motion-ETR/exp_results/MTR_Gopro_test'\
       # --dataroot='/home/dell/Documents/METHOD/SPT/SPT_Detection/Motion-ETR/exp_inputs/MTR_Gopro_test'


