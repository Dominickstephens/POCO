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
# demo.py
import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl' # Keep if needed, might conflict with cv2.imshow

import sys
import cv2
import time
import joblib
import torch
import argparse
from os.path import join, isfile, isdir, basename, dirname
from loguru import logger
import numpy as np # Added numpy

sys.path.append('.')
from pocolib.core.tester import POCOTester
from pocolib.utils.demo_utils import (
    download_youtube_clip,
    video_to_images,
    images_to_video,
    # ADDED: Need xyhw_to_xyxy if used, but detector outputs xyxy
)

MIN_NUM_FRAMES = 0


def main(args):

    demo_mode = args.mode

    # ... (keep video, folder, directory mode setup as is) ...
    if demo_mode == 'video':
        # ... video setup ...
        pass # Keep existing code
    elif demo_mode == 'folder':
        # ... folder setup ...
        args.tracker_batch_size = 1 # Ensure this is set for folder mode
        if args.image_folder:
             input_image_folder = args.image_folder
             output_path = join(args.output_folder, input_image_folder.rstrip('/').split('/')[-1] + '_' + args.exp)
             os.makedirs(output_path, exist_ok=True)
        elif args.vid_file:
             video_file = args.vid_file
             # ... rest of folder from video setup ...
             input_image_folder, num_frames, img_shape = video_to_images(
                video_file,
                img_folder=join(input_path, 'tmp_images'),
                return_info=True
            )
        else:
             logger.error("Either --image_folder or --vid_file must be provided for folder mode.")
             exit(1)
        output_img_folder = join(output_path, 'poco_results')
        os.makedirs(output_img_folder, exist_ok=True)
        num_frames = len(os.listdir(input_image_folder))

    elif demo_mode == 'directory':
        # ... directory setup ...
        args.tracker_batch_size = 1 # Ensure this is set for directory mode
        input_image_dir = args.image_folder
        output_path = args.output_folder
        pass # Keep existing code

    elif demo_mode == 'webcam':
        logger.info('Initializing webcam...')
        cap = cv2.VideoCapture(0) # Moved initialization here
        if not cap.isOpened():
            logger.error('Webcam not available!')
            exit(1)
        ret, frame = cap.read()
        if not ret:
            logger.error('Failed to capture frame for initialization')
            cap.release()
            exit(1)
        height, width = frame.shape[:2]
        logger.info(f'Webcam frame size: {width}x{height}')
        # For webcam mode, we use output_folder as the output path.
        output_path = args.output_folder
        os.makedirs(output_path, exist_ok=True)
        # Don't release cap here, keep it for the loop
        args.tracker_batch_size = 1 # Force batch size 1 for webcam detector

    else:
        raise ValueError(f'{demo_mode} is not a valid demo mode.')

    # Setup logger
    log_file_path = join(output_path, 'demo.log')
    logger.remove() # Remove default logger
    logger.add(sys.stderr, level="INFO") # Keep console output
    logger.add(
        log_file_path,
        level='INFO',
        colorize=False,
    )
    logger.info(f"Logging to {log_file_path}")
    logger.info(f'Demo options: \n {args}')

    # Initialize POCOTester (which now initializes detector and renderer)
    tester = POCOTester(args)

    total_time = time.time()

    # --- MODIFIED: Webcam Loop ---
    if args.mode == 'webcam':
        logger.info("Starting webcam demo...")
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error('Failed to capture frame from webcam')
                break

            loop_start_time = time.time()

            # 1. Detect people
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_boxes_xyxy = tester.detect_frame(frame_rgb) # Returns list of [x1,y1,x2,y2]

            # 2. Process each detection
            render_infos = [] # Store info needed for compositing
            rendered_parts = [] # Store rendered images on black bg

            for bbox_xyxy in detected_boxes_xyxy:
                # Run POCO on this bounding box
                # run_on_webcam_frame now returns the rendered part and info
                rendered_img_part, render_info = tester.run_on_webcam_frame(frame, bbox_xyxy)

                if rendered_img_part is not None and render_info is not None:
                    rendered_parts.append(rendered_img_part)
                    render_infos.append(render_info)

            # 3. Composite results onto the original frame
            final_frame = frame.copy() # Start with the original BGR frame
            # Render in reverse order of detection? Or sort by depth later? Simple overlay for now.
            for i in range(len(rendered_parts)):
                 # Convert rendered part back to BGR if it's RGB
                 rendered_bgr = cv2.cvtColor(rendered_parts[i], cv2.COLOR_RGB2BGR)
                 # Use the mask from render_info
                 mask = render_infos[i]['mask']
                 final_frame[mask] = rendered_bgr[mask]


            # 4. Display FPS and the final frame
            loop_time = time.time() - loop_start_time
            fps = 1.0 / loop_time if loop_time > 0 else 0
            cv2.putText(final_frame, f'FPS: {fps:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('POCO Webcam Demo', final_frame)

            # 5. Optional: Save frame
            # cv2.imwrite(join(output_path, f'frame_{frame_count:06d}.jpg'), final_frame)
            frame_count += 1

            # 6. Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Exit key pressed.")
                break

        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam demo ended.")

    # --- Keep other modes (video, folder, directory) ---
    elif args.mode == 'video':
        # ... (keep existing video processing logic) ...
        # Make sure it uses tester.run_tracking, tester.run_on_video, tester.render_results
        logger.info(f'Input video number of frames {num_frames}')
        orig_height, orig_width = img_shape[:2]
        total_time = time.time()
        tracking_method = args.tracking_method
        tracking_results_file = join(input_path if 'input_path' in locals() else output_path, f'tracking_results_{tracking_method}.pkl') # Adjust path finding

        if isfile(tracking_results_file):
            logger.info(f'Skipping running the tracker as results already exists at {tracking_results_file}')
            tracking_results = joblib.load(tracking_results_file)
        else:
            tracking_results = tester.run_tracking(video_file, input_image_folder, output_path) # Pass output_path for saving
            logger.info(f'Saving tracking results at {tracking_results_file}')
            joblib.dump(tracking_results, tracking_results_file)

        poco_time = time.time()
        poco_results = tester.run_on_video(tracking_results, input_image_folder, orig_width, orig_height)
        end = time.time()

        # ... (rest of video logging and rendering) ...
        fps = num_frames / (end - poco_time) if (end - poco_time) > 0 else 0
        logger.info(f'POCO FPS: {fps:.2f}')
        total_time_elapsed = time.time() - total_time
        logger.info(f'Total time spent: {total_time_elapsed:.2f} seconds (including model loading time).')
        logger.info(f'Total FPS (including model loading time): {num_frames / total_time_elapsed:.2f}' if total_time_elapsed > 0 else 'N/A')

        if not args.no_render:
             output_img_folder = join(output_path, 'poco_results') # Define output image folder
             os.makedirs(output_img_folder, exist_ok=True)
             tester.render_results(poco_results, input_image_folder, output_img_folder, output_path,
                                   orig_width, orig_height, num_frames)
             # ... rest of images_to_video ...
             vid_name = basename(video_file)
             save_name = f'{vid_name.replace(".mp4", "")}_{args.exp}_result.mp4'
             save_name = join(output_path, save_name)
             logger.info(f'Saving result video to {save_name}')
             images_to_video(img_folder=output_img_folder, output_vid_file=save_name)
             # Optionally save original frames as video too
             # images_to_video(img_folder=input_image_folder, output_vid_file=join(output_path, vid_name))


    elif args.mode == 'folder':
        # ... (keep existing folder processing logic) ...
        # Make sure it uses tester.run_detector, tester.run_on_image_folder
        logger.info(f'Number of input frames {num_frames}')
        total_time_start = time.time()
        detection_results_file = join(output_path, 'detection_results.pkl')

        if isfile(detection_results_file):
            logger.info(f'Skipping running the detector as results already exist at {detection_results_file}')
            detections = joblib.load(detection_results_file)
        else:
            detections = tester.run_detector(input_image_folder)
            logger.info(f'Saving detection results at {detection_results_file}')
            joblib.dump(detections, detection_results_file)

        poco_time = time.time()
        # Make sure output_img_folder is defined if rendering is enabled
        render_folder = output_img_folder if not args.no_render else None
        tester.run_on_image_folder(input_image_folder, detections, output_path, render_folder)
        end = time.time()

        # ... (rest of folder logging) ...
        fps = num_frames / (end - poco_time) if (end - poco_time) > 0 else 0
        logger.info(f'POCO FPS: {fps:.2f}')
        total_time_elapsed = time.time() - total_time_start
        logger.info(f'Total time spent: {total_time_elapsed:.2f} seconds (including model loading time).')
        logger.info(f'Total FPS (including model loading time): {num_frames / total_time_elapsed:.2f}' if total_time_elapsed > 0 else 'N/A')

        # Optionally convert rendered images to video
        if not args.no_render and render_folder:
             img_folder_name = basename(input_image_folder.rstrip('/'))
             save_name = f'{img_folder_name}_{args.exp}_result.mp4'
             save_name = join(output_path, save_name)
             logger.info(f'Saving result video to {save_name}')
             images_to_video(img_folder=render_folder, output_vid_file=save_name)


    elif args.mode == 'directory':
        # ... (keep existing directory processing logic) ...
        # Make sure it uses tester.run_detector, tester.run_on_image_folder
        image_dirs = [join(input_image_dir, n) for n in sorted(os.listdir(input_image_dir)) \
                                                        if isdir(join(input_image_dir, n))]
        start_dir = min(args.dir_chunk * args.dir_chunk_size, len(image_dirs))
        end_dir = min((1+args.dir_chunk) * args.dir_chunk_size, len(image_dirs))
        logger.info(f"Processing directories {start_dir} to {end_dir-1}")

        for folder_id in range(start_dir, end_dir):
            current_input_folder = image_dirs[folder_id]
            current_output_path = join(args.output_folder, basename(current_input_folder))
            os.makedirs(current_output_path, exist_ok=True)

            f_img_dir_base = basename(current_input_folder)
            detection_results_file = join(current_output_path, f'{f_img_dir_base}_detection_results.pkl')
            results_file = join(current_output_path, f'results_{f_img_dir_base}.pkl') # Check for final results instead?


            logger.info(f'Working on directory {folder_id+1}/{len(image_dirs)} - {f_img_dir_base}')
            # Check if final results exist, maybe skip entire folder
            # if isfile(results_file): # Example check
            #      logger.info(f'Skipping {f_img_dir_base} as final results seem present.')
            #      continue

            # --- Run Detection ---
            if isfile(detection_results_file):
                logger.info(f'Skipping detector, loading results from {detection_results_file}')
                detections = joblib.load(detection_results_file)
            else:
                detections = tester.run_detector(current_input_folder)
                logger.info(f'Saving detection results to {detection_results_file}')
                joblib.dump(detections, detection_results_file)

            # --- Run POCO ---
            render_folder = None
            if not args.no_render:
                render_folder = join(current_output_path, 'poco_results')
                os.makedirs(render_folder, exist_ok=True)

            tester.run_on_image_folder(current_input_folder, detections, current_output_path, render_folder)
            logger.info(f"Finished processing {f_img_dir_base}")


    # Cleanup / Final message
    del tester.model # Release GPU memory if possible
    logger.info('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default='configs/demo_poco_cliff.yaml',
                        help='config file that defines model hyperparams')

    parser.add_argument('--ckpt', type=str, default='data/poco_cliff.pt',
                        help='checkpoint path')

    parser.add_argument('--inf_model', type=str, default='best',
                        help='select the model from checkpoint folder (usually best/last)')

    parser.add_argument('--exp', type=str, default='webcam_test', # Changed default
                        help='short description of the experiment')

    parser.add_argument('--mode', default='webcam', choices=['video', 'folder', 'directory', 'webcam'], # Changed default
                        help='Demo type')

    parser.add_argument('--vid_file', type=str, default=None, # Default None
                        help='input video path or youtube link')

    parser.add_argument('--image_folder', type=str, default=None, # Default None
                        help='input image folder')

    parser.add_argument('--skip_frame', type=int, default=1,
                        help='Skip frames when running demo on image folder/video')

    parser.add_argument('--output_folder', type=str, default='out/webcam_output', # Changed default
                        help='output folder to write results')

    # --- Directory Mode Args ---
    parser.add_argument('--dir_chunk_size', type=int, default=1000,
                        help='Run demo on chunk size directory')
    parser.add_argument('--dir_chunk', type=int, default=0,
                        help='instance of chunk for demo on directory')

    # --- Tracking/Detection Args ---
    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'],
                        help='tracking method (bbox or pose) for video mode')
    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used (yolo or maskrcnn)')
    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')
    parser.add_argument('--tracker_batch_size', type=int, default=1, # Default 1 for webcam/folder
                        help='batch size of object detector')
    # parser.add_argument('--staf_dir', type=str, default='/path/to/staf', # Needs actual path if using pose tracking
                        # help='path to directory STAF pose tracking method installed.')

    # --- POCO Batch Size ---
    parser.add_argument('--batch_size', type=int, default=32, # Model forward batch size
                        help='batch size for POCO model inference (used in run_on_video/run_on_image_folder)')

    # --- Display/Rendering Args ---
    parser.add_argument('--display', action='store_true',
                        help='display the results in an OpenCV window (might not work with EGL)')
    parser.add_argument('--smooth', action='store_true',
                        help='smooth the results to prevent jitter (video mode only)')
    parser.add_argument('--min_cutoff', type=float, default=0.004, help='one euro filter min cutoff.')
    parser.add_argument('--beta', type=float, default=1.5, help='one euro filter beta.')
    parser.add_argument('--no_render', action='store_true',
                        help='disable rendering of output video/images.')
    parser.add_argument('--render_crop', action='store_true',
                        help='Render results on the cropped image patch instead of the original frame.')
    parser.add_argument('--no_uncert_color', action='store_true',
                        help='Disable using uncertainty values to color the mesh.')
    parser.add_argument('--wireframe', action='store_true',
                        help='render meshes as wireframes.')
    parser.add_argument('--sideview', action='store_true',
                        help='render meshes from alternate viewpoint (video mode only).')
    parser.add_argument('--draw_keypoints', action='store_true',
                        help='draw 2d keypoints on the rendered image.')
    # parser.add_argument('--no_kinematic_uncert', action='store_false', # Handled in tester init
                        # help='Do not use SMPL Kinematic for uncert')
    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    args = parser.parse_args()

    # Some argument validation
    if args.mode == 'video' and not args.vid_file:
        logger.error("--vid_file is required for video mode.")
        exit(1)
    if args.mode == 'folder' and not args.image_folder and not args.vid_file:
        logger.error("--image_folder or --vid_file is required for folder mode.")
        exit(1)
    if args.mode == 'directory' and not args.image_folder:
        logger.error("--image_folder (path to parent directory) is required for directory mode.")
        exit(1)


    main(args)
