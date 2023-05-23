# import debugpy

# # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
# debugpy.listen(5678)
# print("Waiting for debugger attach")
# debugpy.wait_for_client()
# debugpy.breakpoint()
# print('break on this line')

import time
import os
from options.visualization_option import VisualOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util.metrics import PSNR
import util.util as util
import numpy as np 
import torch
import cv2
import csv

from ssim import SSIM
from PIL import Image

def save_image(visuals, img_path):
	'''This function seems not used'''
	# import ipdb; ipdb.set_trace()
	root, file = os.path.split(img_path)
	# root, dir = os.path.split(root)
	new_path = './results/real_images/'
	if not os.path.exists(new_path):
		os.mkdir(new_path)
	name = os.path.splitext(file)[0]

	deblur_file = os.path.join(new_path,name+'_sharp.png')
	blur_file = os.path.join(new_path,name+'_blurry.png')

	for label, image_numpy in visuals.items():
		if label == 'fake_B':
			image_pil = Image.fromarray(image_numpy)
			image_pil.save(deblur_file)
		if label == 'real_A':
			image_pil = Image.fromarray(image_numpy)
			image_pil.save(blur_file)


opt = VisualOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
# opt.dataroot='/home/yjz/datasets/Synthetic_motion_flow/test_syn'
# opt.dataset_mode='aligned'


extract_frame = opt.extract_frame
visualize_traj = opt.visualize_traj 
visualize_flow = opt.visualize_flow # True when reblur，false when deblur
visualize_testing_flow_offset = opt.visualize_testing_flow_offset

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# test
avgPSNR = 0.0
avgSSIM = 0.0
counter = 0

for i, data in enumerate(dataset):
	if (i >= opt.how_many) & opt.set_how_many:
		break

	counter = i+1
	model.set_input(data)
	model.test()
	if opt.blur_direction == 'reblur':
		# save real blurry and fake reblur image
		reblur_res = model.fake_A
		blurry_input = model.real_A
		reblur_res_np = util.tensor2im(reblur_res)
		blurry_input_np = util.tensor2im(blurry_input)
		img_path = model.get_image_paths()
		root, file = os.path.split(img_path[0])
		name = os.path.splitext(file)[0]
		print('process image %s' % name)
		# write image
		# input_dir = opt.dataroot.split('/')[-1]
		# output_dir = os.path.join(opt.results_dir,opt.name, input_dir)
		# output_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
		output_dir = os.path.join(opt.results_dir)
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		reblur_name = os.path.join(output_dir,name+'_reblur.png')
		blur_name = os.path.join(output_dir,name+'_blurry.png')
		util.save_image(reblur_res_np, reblur_name)
		util.save_image(blurry_input_np, blur_name)

		# save reblur image at each offset
		each_reblur = model.fake_A_n
		B, n_offset, C, H, W = each_reblur.shape
		each_reblur = each_reblur.view(B * n_offset, C, H, W)
		# Assuming `order_new` is a list of indices specifying the new order along axis=0 on GPU
		order_new = [1,2,3,4,5,6,0,7,8,9,10,11,12,13,14] # first move right part, then move left part
		order_new = torch.LongTensor(order_new).cuda()
		# Reorder the `each_reblur` tensor along axis=0 according to `order_new`
		each_reblur_new = each_reblur.index_select(0, order_new)
		for i in range(n_offset):
			each_reblur_img = util.tensor2im(each_reblur_new[i:i+1,:,:,:])
			each_reblur_name = os.path.join(output_dir,name+'_EachReblurClip%02d.png'%i)
			util.save_image(each_reblur_img, each_reblur_name)

		# rescale raw image without clip and save as tif
		for i in range(n_offset):
			each_reblur_img = util.tensor2im_raw(each_reblur_new[i:i+1,:,:,:])
			each_reblur_img = np.sum(each_reblur_img, axis=2)
			each_reblur_img = cv2.normalize(each_reblur_img, None, 1.0, 0.0, cv2.NORM_MINMAX)
			# Find the maximum pixel value in the image
			# Sometimes max pixel may not be exactly 1 due to imprecision
			max_val = np.max(each_reblur_img)
			# Round the maximum pixel value to 1 if it's close enough
			if abs(max_val - 1) < 1e-6:
				each_reblur_img[np.unravel_index(np.argmax(each_reblur_img, axis=None), each_reblur_img.shape)] = 1	
			each_reblur_name = os.path.join(output_dir,name+'_EachReblurRescale%02d.tiff'%i)
			cv2.imwrite(each_reblur_name,each_reblur_img)
		
		# Mobility estimation by calculating the brightest pixel movement in intermediate reblur images
		# Resctrict movement calculation in centered 13x13 pixel range
		for i in range(n_offset):
			each_reblur_img = util.tensor2im_raw(each_reblur_new[i:i+1,:,:,:])
			each_reblur_img = np.sum(each_reblur_img, axis=2) # convert to 32x32x1 size
			center_img = each_reblur_img[9:23,9:23]
			center_max = center_img.max()
			# Find the pixel coordinates in the range [9:23,9:23] that have a value equal to max_val
			[row_idx,col_idx] = np.nonzero(each_reblur_img[9:23,9:23] == center_img.max())
			row_idx += 9 # Adjust the pixel coordinates to account for the cropped region
			col_idx += 9
			# save center_max row and col to CSV file, note index start from 0
			# If the file doesn't exist, it will be created automatically
			with open(os.path.join(output_dir,'max_pixel_coordinate.csv'), 'a', newline='') as csvfile:
				writer = csv.writer(csvfile)
				# Add a header row if the file is empty
				empty_file = csvfile.tell() == 0
				headers = ['Name','Frame','Row_idx','Col_idx']
				writer.writerow(headers) if empty_file else None
				# Write the new data row
				writer.writerow([name, i, row_idx[0], col_idx[0]])


		

	if opt.blur_direction == 'deblur':
		deblur_res = model.fake_B
		blurry_input = model.real_A
		deblur_res_np = util.tensor2im(deblur_res)
		blurry_input_np = util.tensor2im(blurry_input)
		img_path = model.get_image_paths()
		root, file = os.path.split(img_path[0])
		name = os.path.splitext(file)[0]
		print('process image %s' % name)
		# write image
		input_dir = opt.dataroot.split('/')[-1]
		output_dir = os.path.join(opt.results_dir,opt.name, input_dir)
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		
		# deblur_name = os.path.join(output_dir,name+'_deblur.png')
		# blur_name = os.path.join(output_dir,name+'_blurry.png')
		
		# util.save_image(deblur_res_np, deblur_name)
		# util.save_image(blurry_input_np, blur_name)

	# visualization
	if visualize_traj:
		vis_dir = os.path.join(output_dir, 'traj')
		if not os.path.exists(vis_dir):
			os.mkdir(vis_dir)
		traj = model.draw_quadratic_line()
		vis_name = os.path.join(vis_dir,name+'_traj.png')
		util.save_image(traj, vis_name)
	if visualize_flow:
		vis_dir = os.path.join(output_dir, 'vis_flow')
		if not os.path.exists(vis_dir):
			os.mkdir(vis_dir)
		flow = model.visualize_offset_flow()
		vis_name = os.path.join(vis_dir,name+'_flow.png')
		util.save_image(flow, vis_name)
	if extract_frame:
		frame_dir = os.path.join(output_dir,'frame')
		if not os.path.exists(frame_dir):
			os.mkdir(frame_dir)
		frames = model.save_everyframe()
		for i in range(len(frames)):
			frame = frames[i]
			frame_name = os.path.join(frame_dir,name+'_frame%02d.png'%i)
			util.save_image(frame,frame_name)
	if visualize_testing_flow_offset:
		model.visualize_offset_flow_test(output_dir,name)




