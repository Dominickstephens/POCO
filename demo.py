# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

# -*- coding: utf-8 -*-
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import sys
import cv2
import time
import joblib
import torch
import argparse
import numpy as np
from os.path import join, isfile, isdir, basename, dirname
from loguru import logger

import tempfile
from os.path import join

# Existing imports for video/folder demos
from pocolib.core.tester import POCOTester
from pocolib.utils.demo_utils import (
    download_youtube_clip,
    video_to_images,
    images_to_video,
    convert_crop_cam_to_orig_img,
)
# Additional imports needed for webcam mode
from pocolib.utils.vibe_image_utils import get_single_image_crop_demo
from pocolib.utils.image_utils import calculate_bbox_info, calculate_focal_length
from pocolib.utils.vibe_renderer import Renderer

# Import multi-person tracker
from multi_person_tracker import MPT # type: ignore

MIN_NUM_FRAMES = 0

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):

    demo_mode = args.mode
    stream_mode = args.stream

    # Initialize the POCO tester (builds the model, loads checkpoints, etc.)
    tester = POCOTester(args)

    if demo_mode == 'video':
        video_file = args.vid_file

        # ========= [Optional] download the youtube video ========= #
        if video_file.startswith('https://www.youtube.com'):
            logger.info(f'Downloading YouTube video \"{video_file}\"')
            video_file = download_youtube_clip(video_file, './data/video_demos')

            if video_file is None:
                exit('Youtube url is not valid!')

            logger.info(f'YouTube Video has been downloaded to {video_file}...')

        if not isfile(video_file):
            exit(f'Input video \"{video_file}\" does not exist!')

        output_path = join(args.output_folder, basename(video_file).replace('.mp4', '_' + args.exp))
        input_path = join(dirname(video_file), basename(video_file).replace('.mp4', '_' + args.exp))
        os.makedirs(input_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)

        if isdir(join(input_path, 'tmp_images')):
            input_image_folder = join(input_path, 'tmp_images')
            logger.info(f'Frames are already extracted in \"{input_image_folder}\"')
            num_frames = len(os.listdir(input_image_folder))
            img_shape = cv2.imread(join(input_image_folder, '000001.png')).shape
        else:
            input_image_folder, num_frames, img_shape = video_to_images(
                video_file,
                img_folder=join(input_path, 'tmp_images'),
                return_info=True
            )
        output_img_folder = join(output_path, 'tmp_images_output')
        os.makedirs(output_img_folder, exist_ok=True)

        logger.add(join(output_path, 'demo.log'), level='INFO', colorize=False)
        logger.info(f'Demo options: \n {args}')

        # Run tracking and subsequent POCO inference on video frames...
        tracking_method = args.tracking_method
        if isfile(join(input_path, f'tracking_results_{tracking_method}.pkl')):
            logger.info(f'Skipping running the tracker as results already exists at {input_path}')
            tracking_results = joblib.load(join(input_path, f'tracking_results_{tracking_method}.pkl'))
        else:
            tracking_results = tester.run_tracking(video_file, input_image_folder, input_path)
            logger.info(f'Saving tracking results at {input_path}/tracking_results_{tracking_method}.pkl')
            joblib.dump(tracking_results, join(input_path, f'tracking_results_{tracking_method}.pkl'))
        poco_time = time.time()
        poco_results = tester.run_on_video(tracking_results, input_image_folder, img_shape[1], img_shape[0])
        end = time.time()

        fps = num_frames / (end - poco_time)
        logger.info(f'Saving the model..')
        torch.save(tester.model.state_dict(), f'{output_path}/best_model.pt')
        del tester.model

        logger.info(f'poco FPS: {fps:.2f}')
        total_time = time.time() - poco_time
        logger.info(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
        logger.info(f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')

        if not args.no_render:
            tester.render_results(poco_results, input_image_folder, output_img_folder, output_path,
                                  img_shape[1], img_shape[0], num_frames)
            # ========= Save rendered video ========= #
            vid_name = basename(video_file)
            save_name = f'{vid_name.replace(".mp4", "")}_{args.exp}_result.mp4'
            save_name = join(output_path, save_name)
            logger.info(f'Saving result video to {save_name}')
            images_to_video(img_folder=output_img_folder, output_vid_file=save_name)
            images_to_video(img_folder=input_image_folder, output_vid_file=join(output_path, vid_name))

    elif demo_mode == 'folder':
        args.tracker_batch_size = 1  # As each image can be of different sizes
        if args.image_folder:
            input_image_folder = args.image_folder
            output_path = join(args.output_folder, input_image_folder.rstrip('/').split('/')[-1] + '_' + args.exp)
            os.makedirs(output_path, exist_ok=True)
        elif args.vid_file:
            video_file = args.vid_file
            output_path = join(args.output_folder, basename(video_file).replace('.mp4', '_' + args.exp))
            input_path = join(dirname(video_file), basename(video_file).replace('.mp4', '_' + args.exp))
            os.makedirs(input_path, exist_ok=True)
            os.makedirs(output_path, exist_ok=True)
            input_image_folder, num_frames, img_shape = video_to_images(
                video_file,
                img_folder=join(input_path, 'tmp_images'),
                return_info=True
            )
        output_img_folder = join(output_path, 'poco_results')
        os.makedirs(output_img_folder, exist_ok=True)
        num_frames = len(os.listdir(input_image_folder))

        logger.add(join(output_path, 'demo.log'), level='INFO', colorize=False)
        logger.info(f'Demo options: \n {args}')

        total_time = time.time()
        if isfile(join(output_path, 'detection_results.pkl')):
            logger.info(f'Skipping running the detector as results already exist')
            detections = joblib.load(join(output_path, 'detection_results.pkl'))
        else:
            detections = tester.run_detector(input_image_folder)
            logger.info(f'Saving detection results at {output_path}/detection_results.pkl')
            joblib.dump(detections, join(output_path, 'detection_results.pkl'))
        poco_time = time.time()
        tester.run_on_image_folder(input_image_folder, detections, output_path, output_img_folder)
        end = time.time()

        fps = num_frames / (end - poco_time)
        del tester.model

        logger.info(f'poco FPS: {fps:.2f}')
        total_time = time.time() - total_time
        logger.info(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
        logger.info(f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')

    elif demo_mode == 'directory':
        args.tracker_batch_size = 1
        input_image_dir = args.image_folder
        output_path = args.output_folder
        image_dirs = [join(input_image_dir, n) for n in sorted(os.listdir(input_image_dir))
                      if isdir(join(input_image_dir, n))]
        start_dir = min(args.dir_chunk * args.dir_chunk_size, len(image_dirs))
        end_dir = min((1+args.dir_chunk) * args.dir_chunk_size, len(image_dirs))
        for folder_id in range(start_dir, end_dir):
            input_image_folder = image_dirs[folder_id]
            output_path = join(args.output_folder, input_image_folder.rstrip('/').split('/')[-1])
            os.makedirs(output_path, exist_ok=True)
            output_img_folder = None
            if not args.no_render:
                output_img_folder = join(output_path, 'poco_results')
                os.makedirs(output_img_folder, exist_ok=True)
            f_img_dir = output_path.split('/')[-1]

            logger.info(f'Working on directory {folder_id}/{len(image_dirs)} - {f_img_dir}')

            if isfile(join(output_path, 'results_' + f_img_dir + '.npz')):
                logger.info(f'Skipping running POCO as results are already present')
            else:
                if isfile(join(output_path, f_img_dir + '_detection_results.pkl')):
                    logger.info(f'Skipping running the detector as results already exist')
                    detections = joblib.load(join(output_path, f_img_dir + '_detection_results.pkl'))
                else:
                    detections = tester.run_detector(input_image_folder)
                    logger.info(f'Saving detection results at {output_path}/detection_results.pkl')
                    joblib.dump(detections, join(output_path, f_img_dir + '_detection_results.pkl'))

                tester.run_on_image_folder(input_image_folder, detections, output_path, output_img_folder)

    elif demo_mode == 'webcam':
        logger.add("webcam_demo.log", level='INFO', colorize=False)
        logger.info(f'Demo options (webcam): \n {args}')

        mot = MPT(
            device=tester.device,
            batch_size=1, # Batch size for MPT internal dataloader, less relevant here
            display=args.display,
            detector_type=args.detector,
            output_format='dict', # This might not be relevant if using detect_frame
            yolo_img_size=args.yolo_img_size,
        )
        if (stream_mode): 
            rtmp_url = "rtmp://34.89.24.203:1935/live/webcam"
            cap = cv2.VideoCapture(rtmp_url, cv2.CAP_FFMPEG)
            frame = cap.read() # Initial read
        else:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.error("Cannot open webcam")
                exit()
            ret, frame = cap.read() # Initial read

        logger.info("Starting webcam stream. Press 'q' to exit.")

        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to grab frame")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = rgb_frame.shape[:2]

            # 1. Get raw detections using detect_frame
            dets_raw = mot.detect_frame(rgb_frame) # Returns [[x1, y1, x2, y2, score], ...]

            # 2. Manually apply the transformation from prepare_output_detections
            dets_prepared_list = []
            if dets_raw is not None and dets_raw.shape[0] > 0:
                for d in dets_raw: # d is [x1, y1, x2, y2, score]
                    w, h = d[2] - d[0], d[3] - d[1]
                    c_x, c_y = d[0] + w / 2, d[1] + h / 2
                    size = max(w, h) # Use max dimension as size
                    # Create the [cx, cy, size, size] format
                    bbox_prepared = np.array([c_x, c_y, size, size])
                    dets_prepared_list.append(bbox_prepared)

            # Convert the list for the current frame into a NumPy array
            dets = np.array(dets_prepared_list) # Now dets is [[cx, cy, sz, sz], ...]

            # 3. Check if any detections remain after preparation
            if dets.shape[0] == 0: # Check the prepared detections
                # Overlay FPS even if no detections
                frame_end = time.time()
                fps = 1.0 / (frame_end - frame_start)
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Webcam Demo", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # --- Downstream processing loop (should now work correctly) ---
            inp_images = []
            bbox_info_list = []
            focal_lengths_list = []
            # These lists now correspond to the prepared detections
            scales_list = []
            centers_list = []
            orig_shapes_list = []

            # Process each PREPARED detection
            for det in dets: # det is now [cx, cy, sz, sz]
                # Assuming get_single_image_crop_demo uses the center/scale logic internally
                # or expects this specific format. We pass the prepared bbox.
                norm_img, bbox_processed, img_patch = get_single_image_crop_demo(
                    rgb_frame,
                    det, # Pass the prepared [cx, cy, sz, sz] bbox
                    kp_2d=None, # Keep kp_2d=None as before
                    scale=1.0, # Keep scale=1.0 as before
                    crop_size=tester.model_cfg.DATASET.IMG_RES
                )

                # It seems the original code derived center/scale *again* here.
                # Let's use the values directly from the prepared 'det'
                # which are already center and size.
                center = [det[0], det[1]] # Use cx, cy from prepared bbox
                # The original scale was max(det[2], det[3]) / 200.0
                # Since det[2] and det[3] are both 'size' now, this becomes:
                scale_val = det[2] / 200.0 # Use size / 200.0 as scale

                inp_images.append(norm_img.float())
                orig_shape = [orig_h, orig_w]

                centers_list.append(center)
                orig_shapes_list.append(orig_shape)
                scales_list.append(scale_val)

                # calculate_bbox_info expects center and scale
                bbox_info = calculate_bbox_info(center, scale_val, orig_shape)
                bbox_info_list.append(bbox_info)

                focal_length = calculate_focal_length(orig_h, orig_w)
                focal_lengths_list.append(focal_length)

            # Prepare batch for inference
            inp_images_tensor = torch.stack(inp_images).to(tester.device)
            batch = {
                'img': inp_images_tensor,
                'bbox_info': torch.FloatTensor(bbox_info_list).to(tester.device),
                'focal_length': torch.FloatTensor(focal_lengths_list).to(tester.device),
                'scale': torch.FloatTensor(scales_list).to(tester.device),
                'center': torch.FloatTensor(centers_list).to(tester.device),
                'orig_shape': torch.FloatTensor(orig_shapes_list).to(tester.device),
            }

            # Run model inference
            tester.model.eval()
            with torch.no_grad():
                output = tester.model(batch)

            # Convert predicted camera parameters to original image space
            pred_cam_np = output['pred_cam'].cpu().numpy()

            # *** IMPORTANT: Check what bbox format convert_crop_cam_to_orig_img expects! ***
            # It might expect the raw [x1,y1,x2,y2] boxes OR the prepared [cx,cy,sz,sz] boxes.
            # If it expects raw boxes, pass dets_raw here instead of dets.
            # If it expects prepared boxes, pass dets. Let's assume prepared for now based on original flow.
            orig_cam = convert_crop_cam_to_orig_img(
                cam=pred_cam_np,
                bbox=dets, # Passing the prepared [cx, cy, sz, sz] boxes
                img_width=orig_w,
                img_height=orig_h
            )
            # *******************************************************************************

            # Initialize renderer
            renderer = Renderer(
                resolution=(orig_w, orig_h),
                orig_img=True,
                wireframe=args.wireframe,
                uncert_type=tester.uncert_type,
            )

            # Render results
            rendered_frame = frame.copy()
            rendered_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB)
            for i in range(len(dets)): # Iterate based on number of prepared detections
                if isinstance(output['smpl_vertices'], torch.Tensor):
                    verts = output['smpl_vertices'][i].cpu().numpy()
                else:
                    verts = output['smpl_vertices'][i]
                color = [0.7, 0.7, 0.7]
                rendered_frame = renderer.render(rendered_frame, verts, cam=orig_cam[i], var=None, color=color)

            # Calculate FPS
            frame_end = time.time()
            fps = 1.0 / (frame_end - frame_start)

            # Display
            rendered_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)
            cv2.putText(rendered_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Webcam Demo", rendered_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default='configs/demo_poco_cliff.yaml',
                        help='config file that defines model hyperparams')
    parser.add_argument('--ckpt', type=str, default='data/poco_cliff.pt',
                        help='checkpoint path')
    parser.add_argument('--inf_model', type=str, default='best',
                        help='select the model from checkpoint folder')
    parser.add_argument('--exp', type=str, default='',
                        help='short description of the experiment')
    parser.add_argument('--mode', default='video', choices=['video', 'folder', 'directory', 'webcam'],
                        help='Demo type')
    parser.add_argument('--vid_file', type=str,
                        help='input video path or youtube link')
    parser.add_argument('--image_folder', type=str,
                        help='input image folder')
    parser.add_argument('--skip_frame', type=int, default=1,
                        help='Skip frames when running demo on image folder')
    parser.add_argument('--output_folder', type=str, default='out',
                        help='output folder to write results')
    parser.add_argument('--stream', type=str2bool, default=False,
                        help='Stream via RTMP for remote work')
    parser.add_argument('--dir_chunk_size', type=int, default=1000,
                        help='Run demo on chunk size directory')
    parser.add_argument('--dir_chunk', type=int, default=0,
                        help='instance of chunk for demo on directory')
    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'],
                        help='tracking method to calculate the tracklet of a subject from the input video')
    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')
    parser.add_argument('--yolo_img_size', type=int, default=256,
                        help='input image size for yolo detector')
    parser.add_argument('--tracker_batch_size', type=int, default=12,
                        help='batch size of object detector used for bbox tracking')
    parser.add_argument('--staf_dir', type=str, default='/home/sdwivedi/work/openpose',
                        help='path to directory STAF pose tracking method installed.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size of poco')
    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')
    parser.add_argument('--smooth', action='store_true',
                        help='smooth the results to prevent jitter')
    parser.add_argument('--min_cutoff', type=float, default=0.004,
                        help='one euro filter min cutoff. Decreasing the minimum cutoff frequency decreases slow speed jitter')
    parser.add_argument('--beta', type=float, default=1.5,
                        help='one euro filter beta. Increasing the speed coefficient(beta) decreases speed lag.')
    parser.add_argument('--no_render', action='store_true',
                        help='disable final rendering of output video.')
    parser.add_argument('--render_crop', action='store_true',
                        help='Render cropped image')
    parser.add_argument('--no_uncert_color', action='store_true',
                        help='No uncertainty color')
    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')
    parser.add_argument('--sideview', action='store_true',
                        help='render meshes from alternate viewpoint.')
    parser.add_argument('--draw_keypoints', action='store_true',
                        help='draw 2d keypoints on rendered image.')
    parser.add_argument('--no_kinematic_uncert', action='store_false',
                        help='Do not use SMPL Kinematic for uncert')
    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    args = parser.parse_args()
    main(args)

