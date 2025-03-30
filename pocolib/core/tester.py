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

# Code Modified from: https://github.com/mkocabas/PARE/blob/master/pare/core/tester.py
# tester.py
import os
import cv2
import time
import torch
import joblib
import colorsys
import numpy as np
from os.path import isfile, isdir, join, basename
from tqdm import tqdm
from loguru import logger
# REMOVE: from yolov3.yolo import YOLOv3  # MPT handles this
from multi_person_tracker import MPT
from torch.utils.data import DataLoader

from . import config
from ..models import POCO, SMPL, HMR
from .config import update_hparams
from ..utils.vibe_renderer import Renderer # Use vibe_renderer as in other parts of tester
from ..utils.pose_tracker import run_posetracker
from ..dataset.inference import Inference
from ..utils.smooth_pose import smooth_pose
from ..utils.poco_utils import POCOUtils
from ..utils.image_utils import overlay_text, calculate_bbox_info, calculate_focal_length
from ..utils.demo_utils import (
    convert_crop_cam_to_orig_img,
    convert_crop_coords_to_orig_img,
    prepare_rendering_results,
    xyhw_to_xyxy, # Keep this utility
    # REMOVE: get_single_image_crop_demo # Replace with a version that takes xyxy bbox
)
# ADDED: Import vibe_image_utils directly
from ..utils import vibe_image_utils # ADDED

MIN_NUM_FRAMES = 0


class POCOTester:
    def __init__(self, args):
        self.args = args
        cfg_file = self.args.cfg
        self.model_cfg = update_hparams(cfg_file)
        # Ensure kinematic uncertainty setting is applied if needed from args
        if hasattr(args, 'no_kinematic_uncert'):
             self.model_cfg.POCO.KINEMATIC_UNCERT = self.args.no_kinematic_uncert
        else:
             self.model_cfg.POCO.KINEMATIC_UNCERT = False # Default if arg not present

        self.ptfile = self.args.ckpt
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # Ensure UNCERT_TYPE is a list
        if isinstance(self.model_cfg.POCO.UNCERT_TYPE, str):
            self.uncert_type = self.model_cfg.POCO.UNCERT_TYPE.split('-')[0] #render with 1st uncerttype
        elif isinstance(self.model_cfg.POCO.UNCERT_TYPE, list):
            self.uncert_type = self.model_cfg.POCO.UNCERT_TYPE[0] #render with 1st uncerttype
        else:
             self.uncert_type = 'pose' # Default

        self.loss_ver = self.model_cfg.POCO.LOSS_VER
        self.poco_utils = POCOUtils(self.model_cfg)
        self.model = self._build_model()
        self.model.eval()

        self.smpl = SMPL(config.SMPL_MODEL_DIR, create_transl=False).to(self.device) # Changed 'cuda' to self.device

        # --- ADDED: Initialize Detector/Tracker ---
        self.mot = MPT(
            device=self.device,
            batch_size=self.args.tracker_batch_size,
            display=False, # Usually don't want display from tracker in lib mode
            detector_type=self.args.detector,
            output_format='dict',
            yolo_img_size=self.args.yolo_img_size,
        )
        # --- END ADDED ---

        # --- ADDED: Initialize Renderer ---
        # Determine resolution - need width/height for this, defer if needed or use default
        # We might need frame dimensions here, let's initialize later or pass dimensions
        # For now, initialize with default, will re-init if needed in run_on_webcam
        self.renderer = Renderer(
            resolution=(224, 224), # Default, might change
            orig_img=True,
            wireframe=self.args.wireframe,
            uncert_type=self.uncert_type,
        )
        # --- END ADDED ---


    def _build_model(self):
        # ========= Define POCO model ========= #
        model_cfg = self.model_cfg

        # Ensure UNCERT_TYPE is a list for the model constructor
        if isinstance(model_cfg.POCO.UNCERT_TYPE, str):
             model_cfg.POCO.UNCERT_TYPE = list(filter(None, model_cfg.POCO.UNCERT_TYPE.split('-')))

        if model_cfg.METHOD == 'poco':
            model = POCO(
                backbone=model_cfg.POCO.BACKBONE,
                img_res=model_cfg.DATASET.IMG_RES,
                uncert_layer=model_cfg.POCO.UNCERT_LAYER,
                activation_type=model_cfg.POCO.ACTIVATION_TYPE,
                uncert_type=model_cfg.POCO.UNCERT_TYPE,
                uncert_inp_type=model_cfg.POCO.UNCERT_INP_TYPE,
                loss_ver=model_cfg.POCO.LOSS_VER,
                num_neurons=model_cfg.POCO.NUM_NEURONS,
                num_flow_layers=model_cfg.POCO.NUM_FLOW_LAYERS,
                sigma_dim=model_cfg.POCO.SIGMA_DIM,
                num_nf_rv=model_cfg.POCO.NUM_NF_RV,
                mask_params_id=model_cfg.POCO.MASK_PARAMS_ID,
                nflow_mask_type=model_cfg.POCO.NFLOW_MASK_TYPE,
                exclude_uncert_idx=model_cfg.POCO.EXCLUDE_UNCERT_IDX,
                use_dropout=model_cfg.POCO.USE_DROPOUT,
                use_iter_feats=model_cfg.POCO.USE_ITER_FEATS,
                cond_nflow=model_cfg.POCO.COND_NFLOW,
                context_dim=model_cfg.POCO.CONTEXT_DIM,
                gt_pose_cond=model_cfg.POCO.GT_POSE_COND,
                gt_pose_cond_ratio=model_cfg.POCO.GT_POSE_COND_RATIO,
                pretrained=self.ptfile,
                inf_model=self.args.inf_model,
            ).to(self.device)
            self.backbone = model_cfg.POCO.BACKBONE
        elif model_cfg.METHOD == 'spin':
            model = HMR(
                backbone=model_cfg.SPIN.BACKBONE,
                img_res=model_cfg.DATASET.IMG_RES,
                pretrained=self.ptfile,
            ).to(self.device)
            self.backbone = model_cfg.SPIN.BACKBONE
        else:
            logger.error(f'{model_cfg.METHOD} is undefined!')
            exit()

        return model

    # --- ADDED: Method to run detection on a single frame ---
    def detect_frame(self, frame_rgb):
        """Runs detector on a single RGB frame."""
        detections = self.mot.detect([frame_rgb]) # MPT detect expects a list of images
        # Extract bbox [x1, y1, x2, y2] for the first frame (index 0)
        if 0 in detections and len(detections[0]) > 0:
            # Return list of bounding boxes for the frame
            # Format is typically [x1, y1, x2, y2, score, class_id]
            return [det[:4] for det in detections[0]]
        else:
            return []
    # --- END ADDED ---


    def run_tracking(self, video_file, image_folder, output_folder):
       # ... (keep existing run_tracking as is) ...
        # ========= Run tracking ========= #
        if self.args.tracking_method == 'bbox':
            # run multi object tracker
            # Use the initialized self.mot
            tracking_results = self.mot(image_folder) # Use self.mot
        elif self.args.tracking_method == 'pose':
            if not os.path.isabs(video_file):
                video_file = os.path.join(os.getcwd(), video_file)
            tracking_results = run_posetracker(video_file, staf_folder=self.args.staf_dir, display=self.args.display)
        else:
            logger.error(f'Tracking method {self.args.tracking_method} is not defined')

        # remove tracklets if num_frames is less than MIN_NUM_FRAMES
        for person_id in list(tracking_results.keys()):
            if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
                del tracking_results[person_id]

        return tracking_results


    def run_detector(self, image_folder):
        # run multi object tracker
        # Use the initialized self.mot
        bboxes = self.mot.detect(image_folder) # Use self.mot
        return bboxes

    @torch.no_grad()
    # MODIFIED: Added bbox parameter
    def run_on_webcam_frame(self, frame, bbox_xyxy, bbox_scale=1.0):
        """
        Process a single frame for a given bounding box.
        Args:
            frame (np.ndarray): Input frame (BGR format from OpenCV).
            bbox_xyxy (list or np.ndarray): Bounding box in [x_min, y_min, x_max, y_max] format.
            bbox_scale (float): Scale factor for cropping.
        Returns:
            np.ndarray: Rendered image for this detection.
            dict: Dictionary containing vertices and camera for compositing. Returns None if processing fails.
        """
        if bbox_xyxy is None:
            logger.warning("Received None bbox, skipping frame processing.")
            return None, None

        # --- Preprocess the Frame ---
        orig_img_bgr = frame # Keep BGR for potential final overlay if needed outside
        height, width = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Work with RGB

        # --- Convert bbox format and calculate center/scale ---
        x1, y1, x2, y2 = bbox_xyxy
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        # Use the larger dimension to determine scale for square crop
        box_height = max(w, h)
        # box_width = box_height # Ensure square aspect ratio for cropping if needed by model

        # bbox format for get_single_image_crop_demo might be [cx, cy, width, height] or similar
        # Let's use cx, cy, box_height, box_height for consistent square cropping
        bbox_xywh = [cx, cy, box_height, box_height] # Use this for cropping

        # --- Cropping ---
        # Using vibe_image_utils.generate_patch_image_cv directly for clarity
        # This requires cx, cy, bb_width, bb_height, patch_width, patch_height
        img_patch_cv, trans = vibe_image_utils.generate_patch_image_cv(
            cvimg=img_rgb.copy(),
            c_x=bbox_xywh[0],
            c_y=bbox_xywh[1],
            bb_width=bbox_xywh[2],
            bb_height=bbox_xywh[3],
            patch_width=self.model_cfg.DATASET.IMG_RES,
            patch_height=self.model_cfg.DATASET.IMG_RES,
            do_flip=False,
            scale=bbox_scale, # Use the provided scale argument
            rot=0,
        )

        raw_img = img_patch_cv.copy() # Save the unnormalized crop
        norm_img = vibe_image_utils.convert_cvimg_to_tensor(img_patch_cv)

        # Prepare input image tensor
        inp_image = norm_img.float().unsqueeze(0).to(self.device)

        # --- Prepare Auxiliary Data ---
        # Bbox info for the model might need center and scale relative to 200px
        # Scale here refers to `box_height / 200.0`
        model_scale = box_height / 200.0
        center_for_bbox_info = [cx, cy]

        bbox_info = calculate_bbox_info(center_for_bbox_info, model_scale, [height, width])
        focal_length = calculate_focal_length(height, width) # Use original image dimensions

        batch = {
            'img': inp_image,
            'bbox_info': torch.FloatTensor([bbox_info]).to(self.device), # Needs to be (1, 3)
            'focal_length': torch.FloatTensor([focal_length]).unsqueeze(0).to(self.device), # Needs to be (1, 1) or just scalar tensor
             # Pass scale and center based on the actual crop
            'scale': torch.FloatTensor([model_scale]).to(self.device), # Needs to be (1,)
            'center': torch.FloatTensor([center_for_bbox_info]).to(self.device), # Needs to be (1, 2)
            'orig_shape': torch.FloatTensor([[height, width]]).to(self.device), # Needs to be (1, 2)
        }

        # --- Run the Model ---
        try:
            output = self.model(batch)
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            return None, None


        # --- Process Outputs ---
        pred_cam = output['pred_cam'].cpu().numpy() # Shape (1, 3)

        # Convert camera parameters from cropped space to original image space.
        # `convert_crop_cam_to_orig_img` expects bbox as [cx, cy, h] where h is the box height used for scaling
        bbox_for_cam_conv = np.array([[cx, cy, box_height]]) # Shape (1, 3)
        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bbox_for_cam_conv, # Use the correct format
            img_width=width,
            img_height=height
        ) # Returns shape (1, 4) -> [sx, sy, tx, ty]

        # Extract the first (and only) set of parameters.
        orig_cam = orig_cam[0] # Shape (4,)

        verts = output['smpl_vertices'][0].cpu().numpy() # Shape (6890, 3)
        smpl_joints2d = output['smpl_joints2d'].cpu().numpy() # Shape (1, 49, 2 or 3)

        # Convert keypoint coordinates from cropped space [-1, 1] to original image space
        # `convert_crop_coords_to_orig_img` expects bbox [cx, cy, h] and keypoints normalized to [-1, 1]
        # Let's re-normalize the predicted joints2d if they are not already
        # Check if the output['smpl_joints2d'] is already normalized
        if smpl_joints2d.max() > 1.0 or smpl_joints2d.min() < -1.0:
             # If not normalized (e.g., in pixel space of the crop), normalize first
             smpl_joints2d_normalized = vibe_image_utils.normalize_2d_kp(smpl_joints2d, self.model_cfg.DATASET.IMG_RES)
        else:
             smpl_joints2d_normalized = smpl_joints2d

        # Now convert the normalized coordinates
        smpl_joints2d_orig = convert_crop_coords_to_orig_img(
            bbox=bbox_for_cam_conv, # Use the same bbox format [cx, cy, h]
            keypoints=smpl_joints2d_normalized, # Use normalized keypoints
            crop_size=self.model_cfg.DATASET.IMG_RES,
        ) # Returns shape (1, 49, 2)

        # Add confidence channel if needed (assuming 1.0 for now)
        smpl_joints2d_orig_conf = np.concatenate(
            [smpl_joints2d_orig, np.ones((1, smpl_joints2d_orig.shape[1], 1))], axis=-1
        ) # Shape (1, 49, 3)

        # Process uncertainty if available
        variance = None
        if f'var_{self.uncert_type}' in output.keys():
            variance = self.poco_utils.prepare_uncert(output[f'var_{self.uncert_type}'])
            # variance_global = self.poco_utils.get_global_uncert(variance.copy())
            # variance_global = np.clip(variance_global, 0, 0.99)
            variance = variance[0] # Get uncertainty for the single person in the batch

        # --- Prepare for Rendering ---
        # Decide which image to render on
        if self.args.render_crop:
            # Render on the cropped RGB image
            img_for_render = raw_img # Use the RGB crop
            render_res = (self.model_cfg.DATASET.IMG_RES, self.model_cfg.DATASET.IMG_RES)
            # Camera for cropped render: use the prediction directly (weak perspective)
            # Format might be [s, tx, ty]
            s, tx, ty = pred_cam[0]
            cam_to_use = [s, s, tx, ty] # vibe_renderer expects [sx, sy, tx, ty]
        else:
            # Render on the original RGB image
            img_for_render = img_rgb.copy()
            render_res = (width, height)
            cam_to_use = orig_cam # Use the converted camera [sx, sy, tx, ty]

        # --- (Re)Initialize the Renderer if resolution changed ---
        if self.renderer.resolution != render_res:
             logger.info(f"Re-initializing renderer for resolution {render_res}")
             self.renderer = Renderer(
                 resolution=render_res,
                 orig_img=True,
                 wireframe=self.args.wireframe,
                 uncert_type=self.uncert_type,
             )

        # --- Render the Results ---
        rendered_person_img = self.renderer.render(
            img=img_for_render, # Pass the correct background image
            verts=verts,
            cam=cam_to_use,
            var=variance,
            color=[0.70, 0.70, 0.70] # Default color, might be overridden by vertex colors if var exists
        )

        # --- Optionally, overlay keypoints on the *rendered* image ---
        if self.args.draw_keypoints:
            kps_to_draw = smpl_joints2d_orig_conf[0] # Get keypoints for the person (shape 49, 3)
            # If rendering crop, need to transform kps back to crop space? No, draw on the image being returned.
            # If rendering crop: kps_to_draw = output['smpl_joints2d'][0].cpu().numpy() # Use crop-space kps
            # else: kps_to_draw = smpl_joints2d_orig_conf[0] # Use original-space kps

            # Determine which set of keypoints to use based on render target
            if self.args.render_crop:
                 # Coordinates are already relative to the crop
                 kps_pixel_space = output['smpl_joints2d'][0].cpu().numpy()
                 # If they are normalized [-1, 1], convert to pixel space
                 if kps_pixel_space.max() <= 1.0 and kps_pixel_space.min() >= -1.0:
                      kps_pixel_space = vibe_image_utils.normalize_2d_kp(kps_pixel_space, self.model_cfg.DATASET.IMG_RES, inv=True)
                 kps_to_draw = np.concatenate([kps_pixel_space, np.ones((kps_pixel_space.shape[0], 1))], axis=-1) # Add conf
            else:
                 kps_to_draw = smpl_joints2d_orig_conf[0]


            for pt_idx, pt in enumerate(kps_to_draw):
                 x, y, conf = int(pt[0]), int(pt[1]), pt[2]
                 if conf > 0.1: # Draw if confidence is high enough
                      color = (255, 255, 255) if pt_idx >= 25 else (0, 0, 0) # White for GT-like, Black for OpenPose-like
                      cv2.circle(rendered_person_img, (x, y), 4, color, -1)

        # Return the rendered image part and potentially other info for compositing
        render_info = {
            'verts': verts,
            'cam': cam_to_use, # Camera used for rendering this person
            'bbox': bbox_xyxy,
            'mask': (rendered_person_img != img_for_render).any(axis=2) # Simple mask where rendered pixels differ from background
        }

        return rendered_person_img, render_info

    # ... (keep run_on_image_folder and run_on_video as they are, they seem more complete) ...
    # Make sure they use self.mot and self.renderer where appropriate

    @torch.no_grad()
    def run_on_image_folder(self, image_folder, detections, output_path, output_img_folder, bbox_scale=1.0):
        # ... (Existing code - Check if self.mot is needed here instead of re-init) ...
        # --- Ensure this method uses self.mot if applicable ---
        # --- Ensure this method uses self.renderer if applicable ---
        image_file_names = [
            join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
        ]
        image_file_names = sorted(image_file_names)


        imgnames_, scales_, centers_, Ss_, parts_, openposes_, poses_, shapes_, vars_ = \
                [], [], [], [], [], [], [], [], []

        pred_cam_, orig_cam_, verts_, betas_, pose_, joints3d_, smpl_joints2d_, bboxes_, var_ = [], [], [], [], [], [], [], [], []
        
        # Store results per image
        all_results = {}

        logger.info(f"Processing {len(image_file_names)} images...")
        for img_idx, img_fname in enumerate(tqdm(image_file_names)):
            # Skip frames if needed (though less common for folder mode)
            if img_idx % self.args.skip_frame != 0:
                 continue

            img_basename = basename(img_fname)
            # Detections are typically a dict {frame_idx: [det1, det2,...]}
            # Need to handle if detections is a list of lists [ [det1_f0, det2_f0], [det1_f1], ...]
            if isinstance(detections, dict):
                 dets = detections.get(img_idx, [])
            elif isinstance(detections, list) and img_idx < len(detections):
                 dets = detections[img_idx]
            else:
                 dets = [] # Default to empty list if format is unexpected or index is out of bounds


            img_rgb = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
            orig_height, orig_width = img_rgb.shape[:2]
            
            rendered_outputs = [] # Store rendered parts for this image

            if not dets:
                logger.warning(f'No detections found for image - {img_fname}')
                # If no detections, save the original image if rendering
                if not self.args.no_render and output_img_folder:
                     cv2.imwrite(join(output_img_folder, img_basename), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                all_results[img_basename] = [] # Store empty result for this image
                continue

            # Prepare batch inputs for all detections in the image
            batch_inp_images = []
            batch_bbox_info = []
            batch_focal_lengths = []
            batch_scales = []
            batch_centers = []
            batch_orig_shapes = []
            valid_dets = [] # Store the actual bounding boxes used

            for det_bbox_xyxy in dets:
                # Convert xyxy to cx, cy, w, h format needed for cropping/processing
                x1, y1, x2, y2 = det_bbox_xyxy[:4] # Handle potential extra values like score/class
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                box_height = max(w, h) # Use larger dimension for scale

                bbox_xywh = [cx, cy, box_height, box_height]

                # Crop image patch
                img_patch_cv, _ = vibe_image_utils.generate_patch_image_cv(
                    cvimg=img_rgb.copy(),
                    c_x=bbox_xywh[0],
                    c_y=bbox_xywh[1],
                    bb_width=bbox_xywh[2],
                    bb_height=bbox_xywh[3],
                    patch_width=self.model_cfg.DATASET.IMG_RES,
                    patch_height=self.model_cfg.DATASET.IMG_RES,
                    do_flip=False,
                    scale=bbox_scale,
                    rot=0,
                )
                norm_img = vibe_image_utils.convert_cvimg_to_tensor(img_patch_cv)
                batch_inp_images.append(norm_img)

                # Calculate auxiliary data
                model_scale = box_height / 200.0
                center_for_bbox_info = [cx, cy]
                bbox_info = calculate_bbox_info(center_for_bbox_info, model_scale, [orig_height, orig_width])
                focal_length = calculate_focal_length(orig_height, orig_width)

                batch_bbox_info.append(bbox_info)
                batch_focal_lengths.append(focal_length)
                batch_scales.append(model_scale)
                batch_centers.append(center_for_bbox_info)
                batch_orig_shapes.append([orig_height, orig_width])
                valid_dets.append(det_bbox_xyxy) # Store the xyxy bbox

            if not batch_inp_images: # If all detections were invalid for some reason
                 if not self.args.no_render and output_img_folder:
                      cv2.imwrite(join(output_img_folder, img_basename), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                 all_results[img_basename] = []
                 continue

            # Stack inputs into a batch
            inp_images_batch = torch.stack(batch_inp_images).to(self.device)
            batch = {
                'img': inp_images_batch,
                'bbox_info': torch.FloatTensor(batch_bbox_info).to(self.device),
                'focal_length': torch.FloatTensor(batch_focal_lengths).unsqueeze(-1).to(self.device), # Ensure shape (N, 1) or just (N,)
                'scale': torch.FloatTensor(batch_scales).to(self.device),
                'center': torch.FloatTensor(batch_centers).to(self.device),
                'orig_shape': torch.FloatTensor(batch_orig_shapes).to(self.device),
            }

            # Run model inference
            output = self.model(batch)

            # Post-process results for each detection in the batch
            pred_cam_batch = output['pred_cam'].cpu().numpy() # (N, 3)
            verts_batch = output['smpl_vertices'].cpu().numpy() # (N, 6890, 3)
            smpl_joints2d_batch = output['smpl_joints2d'].cpu().numpy() # (N, 49, 2 or 3)

            img_results = [] # Store results for this specific image

            # Iterate through batch results
            for idx in range(len(valid_dets)):
                 det_bbox_xyxy = valid_dets[idx]
                 pred_cam = pred_cam_batch[idx] # (3,)
                 verts = verts_batch[idx] # (6890, 3)
                 smpl_joints2d = smpl_joints2d_batch[idx] # (49, 2 or 3)

                 # Convert camera
                 x1, y1, x2, y2 = det_bbox_xyxy
                 cx = (x1 + x2) / 2
                 cy = (y1 + y2) / 2
                 box_height = max(x2 - x1, y2 - y1)
                 bbox_for_cam_conv = np.array([[cx, cy, box_height]]) # (1, 3)
                 orig_cam = convert_crop_cam_to_orig_img(
                     cam=pred_cam[np.newaxis, :], # Add batch dim
                     bbox=bbox_for_cam_conv,
                     img_width=orig_width,
                     img_height=orig_height
                 )[0] # Get first result, shape (4,)

                 # Convert joints
                 if smpl_joints2d.max() <= 1.0 and smpl_joints2d.min() >= -1.0:
                      smpl_joints2d_normalized = smpl_joints2d[np.newaxis, :, :2] # Add batch dim, select xy
                 else:
                      # If not normalized, normalize first
                      smpl_joints2d_normalized = vibe_image_utils.normalize_2d_kp(smpl_joints2d[:, :2], self.model_cfg.DATASET.IMG_RES)[np.newaxis, :, :]

                 smpl_joints2d_orig = convert_crop_coords_to_orig_img(
                     bbox=bbox_for_cam_conv,
                     keypoints=smpl_joints2d_normalized,
                     crop_size=self.model_cfg.DATASET.IMG_RES,
                 )[0] # Get first result, shape (49, 2)
                 smpl_joints2d_orig_conf = np.concatenate(
                     [smpl_joints2d_orig, np.ones((smpl_joints2d_orig.shape[0], 1))], axis=-1
                 ) # Shape (49, 3)


                 # Process uncertainty if available
                 variance = None
                 if f'var_{self.uncert_type}' in output.keys():
                     variance_batch = self.poco_utils.prepare_uncert(output[f'var_{self.uncert_type}'])
                     variance = variance_batch[idx] # Get variance for this detection

                 # Store processed results for this detection
                 person_result = {
                     'bbox': det_bbox_xyxy,
                     'pred_cam': pred_cam,
                     'orig_cam': orig_cam,
                     'verts': verts,
                     'smpl_joints2d': smpl_joints2d_orig_conf,
                     'variance': variance,
                 }
                 img_results.append(person_result)

                 # --- Rendering (if enabled) ---
                 if not self.args.no_render:
                     # Determine render target and camera
                     if self.args.render_crop:
                          # Find the crop corresponding to this detection - requires saving raw_img per detection
                          # This part needs adjustment if render_crop is used in batch mode.
                          # For simplicity, let's skip render_crop in batch folder mode for now.
                          logger.warning("render_crop is not fully supported in batch folder mode yet.")
                          img_for_render = img_rgb.copy()
                          render_res = (orig_width, orig_height)
                          cam_to_use = orig_cam
                     else:
                          img_for_render = img_rgb.copy() # Render on a fresh copy for each person? Or composite later? Let's composite.
                          render_res = (orig_width, orig_height)
                          cam_to_use = orig_cam

                     # (Re)Initialize renderer if needed
                     if self.renderer.resolution != render_res:
                          self.renderer = Renderer(
                              resolution=render_res,
                              orig_img=True,
                              wireframe=self.args.wireframe,
                              uncert_type=self.uncert_type,
                          )

                     # Define color - maybe unique per detection?
                     mc = colorsys.hsv_to_rgb(idx / len(valid_dets), 0.6, 0.8) if len(valid_dets) > 1 else [0.7, 0.7, 0.7]
                     if variance is None and not self.args.no_uncert_color: # Use default grey if no variance
                           mc = [0.7, 0.7, 0.7]
                     if self.args.no_uncert_color: # Force grey if coloring disabled
                           mc = [0.7, 0.7, 0.7]


                     mesh_filename = None
                     if self.args.save_obj:
                         mesh_folder = join(output_path, 'meshes', basename(img_fname).split('.')[0])
                         os.makedirs(mesh_folder, exist_ok=True)
                         mesh_filename = join(mesh_folder, f'person_{idx:02d}.obj')

                     # Render this person
                     rendered_person = self.renderer.render(
                         img=np.zeros_like(img_rgb), # Render on black background first to get mask
                         verts=verts,
                         cam=cam_to_use,
                         var=variance,
                         color=mc,
                         mesh_filename=mesh_filename,
                         hps_backbone=self.backbone,
                     )
                     
                     # Create a mask for compositing
                     mask = (rendered_person > 0).any(axis=2)
                     
                     # Store for later compositing
                     rendered_outputs.append({'render': rendered_person, 'mask': mask})

                     # Optionally draw keypoints (on the composite image later)
                     # Optionally draw bbox (on the composite image later)
                     # Optionally render side view (needs separate handling/compositing)


            # --- Composite Renderings ---
            if not self.args.no_render and output_img_folder:
                 final_image = img_rgb.copy()
                 for data in rendered_outputs:
                      final_image[data['mask']] = data['render'][data['mask']]

                 # --- Draw keypoints and bboxes on the final composite image ---
                 for idx, person_res in enumerate(img_results):
                      if self.args.draw_keypoints:
                           kps_to_draw = person_res['smpl_joints2d'] # Already in original image space
                           for pt_idx, pt in enumerate(kps_to_draw):
                                x, y, conf = int(pt[0]), int(pt[1]), pt[2]
                                if conf > 0.1:
                                     kp_color = (255, 255, 255) if pt_idx >= 25 else (0, 0, 0)
                                     cv2.circle(final_image, (x, y), 4, kp_color, -1)

                      # Draw bounding box
                      # x1, y1, x2, y2 = map(int, person_res['bbox'])
                      # cv2.rectangle(final_image, (x1, y1), (x2, y2), (0, 255, 0), 2)


                 # Save the final composite image
                 cv2.imwrite(join(output_img_folder, img_basename), cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

                 if self.args.display:
                      cv2.imshow('Image Folder Demo', cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
                      if cv2.waitKey(1) & 0xFF == ord('q'):
                           break # Exit if 'q' is pressed during display

            # Store results for this image (optional, for saving detailed output)
            all_results[img_basename] = img_results
            # logger.info(f'Processed image {img_idx + 1}/{len(image_file_names)}')

        # Cleanup display window
        if self.args.display:
            cv2.destroyAllWindows()

        # --- Optional: Save all results ---
        # Example: joblib.dump(all_results, join(output_path, 'all_poco_results.pkl'))


    @torch.no_grad()
    def run_on_video(self, tracking_results, image_folder, orig_width, orig_height, bbox_scale=1.0):
        # ... (Existing code - Check if self.mot is needed here instead of re-init) ...
        # --- Ensure this method uses self.mot if applicable ---
        # --- Ensure this method uses self.renderer if applicable ---
         # ========= Run poco on each person ========= #
        logger.info(f'Running poco on each tracklet...')

        poco_results = {}
        pbar = tqdm(list(tracking_results.keys()))
        for person_id in pbar:
            pbar.set_description(f"Processing track {person_id}")
            bboxes = joints2d = frame_kps = None

            if self.args.tracking_method == 'bbox':
                # Make sure bbox is in [cx, cy, w, h] or adjust Inference dataset
                # Current MPT output is likely [x1, y1, x2, y2, score, class, track_id]
                # Convert to [cx, cy, h, h] (assuming square crop based on height)
                track_data = tracking_results[person_id]['bbox'] # This might be list of lists/tuples
                bboxes_xyxy = np.array([t[:4] for t in track_data])
                cx = (bboxes_xyxy[:, 0] + bboxes_xyxy[:, 2]) / 2
                cy = (bboxes_xyxy[:, 1] + bboxes_xyxy[:, 3]) / 2
                h = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]
                w = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]
                box_height = np.maximum(w, h) # Use max dimension for scale
                # Bbox format for Inference dataset: [cx, cy, scale (box_height/200), scale] - needs verification
                # Or maybe Inference expects [cx, cy, w, h]? Let's assume [cx, cy, h, h] for now
                bboxes = np.stack([cx, cy, box_height, box_height], axis=1)

            elif self.args.tracking_method in ['pose']:
                joints2d = tracking_results[person_id]['joints2d']
                # Bbox might still be needed by Inference dataset, calculate from joints2d?
                # Or assume Inference handles joints2d input correctly.

            frames = tracking_results[person_id]['frames']

            # Create dataset for this tracklet
            dataset = Inference(
                image_folder=image_folder,
                frames=frames,
                bboxes=bboxes, # Pass the converted bboxes
                joints2d=joints2d,
                # frame_kps=frame_kps, # Pass if available
                scale=bbox_scale, # Pass the desired cropping scale
                crop_size = self.model_cfg.DATASET.IMG_RES,
            )

            # If dataset recalculates bboxes from joints, use its bboxes
            bboxes = dataset.bboxes # Use the bboxes from the dataset instance
            frames = dataset.frames
            # frame_kps = dataset.frame_kps # If used

            dataloader = DataLoader(dataset, batch_size=self.args.batch_size, num_workers=8, pin_memory=True)

            pred_cam, pred_verts, pred_pose, pred_betas, pred_var, pred_var_global, \
            pred_joints3d, pred_joints2d = [], [], [], [], [], [], [], []

            for batch_dict in dataloader:
                batch_dict = {k: v.to(device=self.device, non_blocking=True)
                              if hasattr(v, 'to') else v for k, v in batch_dict.items()}

                # --- Prepare batch for POCO model ---
                # Inference dataset should provide keys like 'img', 'bbox_info', 'focal_length', etc.
                # Let's assume it does. If not, calculate them here based on batch_dict['scale'], ['center'], ['orig_shape']
                if 'bbox_info' not in batch_dict:
                     batch_orig_shape = batch_dict['orig_shape'].cpu().numpy()
                     batch_center = batch_dict['center'].cpu().numpy()
                     batch_scale = batch_dict['scale'].cpu().numpy() # scale = box_height / 200
                     batch_bbox_info = []
                     for i in range(len(batch_scale)):
                           bbox_info = calculate_bbox_info(batch_center[i], batch_scale[i], batch_orig_shape[i])
                           batch_bbox_info.append(bbox_info)
                     batch_dict['bbox_info'] = torch.FloatTensor(batch_bbox_info).to(self.device)

                if 'focal_length' not in batch_dict:
                     batch_orig_shape = batch_dict['orig_shape'].cpu().numpy()
                     batch_focal_length = []
                     for i in range(len(batch_orig_shape)):
                          focal = calculate_focal_length(batch_orig_shape[i, 0], batch_orig_shape[i, 1])
                          batch_focal_length.append(focal)
                     batch_dict['focal_length'] = torch.FloatTensor(batch_focal_length).unsqueeze(-1).to(self.device) # Ensure shape (N, 1)


                output = self.model(batch_dict)

                pred_cam.append(output['pred_cam'])
                pred_verts.append(output['smpl_vertices'])
                pred_pose.append(output['pred_pose'])
                pred_betas.append(output['pred_shape'])
                pred_joints3d.append(output['smpl_joints3d'])
                pred_joints2d.append(output['smpl_joints2d']) # These are likely in crop space [-1, 1]

                if f'var_{self.uncert_type}' in output.keys():
                    var = self.poco_utils.prepare_uncert(output[f'var_{self.uncert_type}'], return_torch=True) # Keep as tensor for now
                    pred_var.append(var)
                    var_global = self.poco_utils.get_global_uncert(var.clone()) # Keep as tensor
                    pred_var_global.append(var_global)

            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)
            pred_joints2d_crop = torch.cat(pred_joints2d, dim=0) # In crop space [-1, 1]

            # Convert results to numpy
            pred_cam_np = pred_cam.cpu().numpy()
            pred_verts_np = pred_verts.cpu().numpy()
            pred_pose_np = pred_pose.cpu().numpy()
            pred_betas_np = pred_betas.cpu().numpy()
            pred_joints3d_np = pred_joints3d.cpu().numpy()
            pred_joints2d_crop_np = pred_joints2d_crop.cpu().numpy()

            pred_var_np = []
            pred_var_global_np = []
            if pred_var:
                pred_var_np = torch.cat(pred_var, dim=0).cpu().numpy()
                pred_var_global_np = torch.cat(pred_var_global, dim=0).cpu().numpy()

            # --- Smoothing (Optional) ---
            if self.args.smooth:
                min_cutoff = self.args.min_cutoff
                beta = self.args.beta
                logger.info(f'Running smoothing on person {person_id}, min_cutoff: {min_cutoff}, beta: {beta}')
                # smooth_pose expects pose in axis-angle format. Convert if needed.
                # Assuming pred_pose_np is already axis-angle (N, 72)
                if pred_pose_np.shape[-1] != 72:
                     # TODO: Convert pred_pose_np from matrix/6d to axis-angle if necessary
                     logger.warning("Smoothing requires axis-angle pose. Conversion not implemented here yet.")
                else:
                     pred_verts_np, pred_pose_np, pred_joints3d_np = smooth_pose(
                          pred_pose_np, pred_betas_np,
                          min_cutoff=min_cutoff, beta=beta
                     )

            # --- Coordinate Conversions ---
            # Convert camera
            # bboxes used by the dataset: [cx, cy, h, h]
            bbox_for_cam_conv = np.stack([bboxes[:, 0], bboxes[:, 1], bboxes[:, 2]], axis=1) # Shape (N, 3) [cx, cy, h]
            orig_cam = convert_crop_cam_to_orig_img(
                cam=pred_cam_np,
                bbox=bbox_for_cam_conv,
                img_width=orig_width,
                img_height=orig_height
            ) # Shape (N, 4)

            # Convert 2D joints from crop space [-1, 1] to original image space
            pred_joints2d_orig = convert_crop_coords_to_orig_img(
                bbox=bbox_for_cam_conv, # Use [cx, cy, h]
                keypoints=pred_joints2d_crop_np[:,:,:2], # Use normalized xy
                crop_size=self.model_cfg.DATASET.IMG_RES,
            ) # Shape (N, 49, 2)

            # Add confidence
            pred_joints2d_orig_conf = np.concatenate(
                [pred_joints2d_orig, np.ones((pred_joints2d_orig.shape[0], pred_joints2d_orig.shape[1], 1))],
                axis=-1
            ) # Shape (N, 49, 3)

            output_dict = {
                'pred_cam': pred_cam_np,
                'orig_cam': orig_cam,
                'verts': pred_verts_np,
                'pose': pred_pose_np,
                'betas': pred_betas_np,
                # 'joints2d': joints2d, # Input joints2d if available, maybe not needed here
                'smpl_joints3d': pred_joints3d_np,
                'smpl_joints2d': pred_joints2d_orig_conf, # Store original image space keypoints
                'var': pred_var_np if len(pred_var_np)>0 else np.array([]),
                'var_global': pred_var_global_np if len(pred_var_global_np)>0 else np.array([]),
                'bboxes': bboxes, # Store the bboxes used [cx, cy, h, h]
                'frame_ids': frames,
            }

            poco_results[person_id] = output_dict

        return poco_results


    def render_results(self, poco_results, image_folder, output_img_folder, output_path,
                       orig_width, orig_height, num_frames):
        # ... (Existing code - Check if self.renderer is used here) ...
        # --- Ensure this method uses self.renderer ---
        # ========= Render results as a single video ========= #
        # Initialize renderer with the final output resolution
        renderer = Renderer(
            resolution=(orig_width, orig_height),
            orig_img=True,
            wireframe=self.args.wireframe,
            uncert_type=self.uncert_type,
        )
        if self.args.sideview:
             side_renderer = Renderer(
                  resolution=(orig_width, orig_height), # Same resolution for side view?
                  orig_img=True, # Render on black bg for side view
                  wireframe=self.args.wireframe,
                  uncert_type=self.uncert_type,
             )


        logger.info(f'Rendering output video, writing frames to {output_img_folder}')

        # prepare results for rendering
        frame_results = prepare_rendering_results(poco_results, num_frames)
        # mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in poco_results.keys()}
        mesh_color = {k: [0.98, 0.54, 0.44] for k in poco_results.keys()} # Example fixed color

        image_file_names = sorted([
            join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg') # Added jpeg
        ])

        pbar = tqdm(range(len(image_file_names)))
        for frame_idx in pbar:
            pbar.set_description(f"Rendering frame {frame_idx}")
            img_fname = image_file_names[frame_idx]
            img = cv2.imread(img_fname)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Work with RGB for renderer background

            rendered_output_parts = [] # To store rendered person images for compositing

            if self.args.sideview:
                side_img_composite = np.ones_like(img_rgb) * 255 # White background for side view

            # Sort frame results by camera depth (approximated by scale sy) for better occlusion handling
            frame_data = frame_results[frame_idx]
            # Sorting: smaller sy (further away) should be rendered first
            sorted_persons = sorted(frame_data.items(), key=lambda item: item[1]['cam'][1])


            for person_id, person_data in sorted_persons:
                frame_verts = person_data['verts']
                frame_cam = person_data['cam'] # Already in orig_cam format [sx, sy, tx, ty]
                frame_kp = person_data['joints2d'] # Already in orig image space
                frame_var = person_data.get('var', None) # Use .get for safety
                frame_var_global = person_data.get('var_global', None)

                if frame_var_global is not None:
                    frame_var_global = np.clip(frame_var_global, 0, 0.99)

                # Determine mesh color based on uncertainty
                mc = mesh_color[person_id] # Default color
                rend_var = None
                if frame_var is not None and not self.args.no_uncert_color:
                     rend_var = frame_var.copy()
                elif self.args.no_uncert_color or frame_var is None : # Force grey if disabled or no variance
                     mc = [0.7, 0.7, 0.7]


                mesh_filename = None
                if self.args.save_obj:
                    mesh_folder = join(output_path, 'meshes', f'{person_id:04d}')
                    os.makedirs(mesh_folder, exist_ok=True)
                    mesh_filename = join(mesh_folder, f'{frame_idx:06d}.obj')

                # Render this person on a black background to get mask
                rendered_person = renderer.render(
                    img=np.zeros_like(img_rgb), # Render on black background
                    verts=frame_verts,
                    cam=frame_cam,
                    var=rend_var,
                    color=mc,
                    mesh_filename=mesh_filename,
                    hps_backbone=self.backbone,
                )
                mask = (rendered_person > 0).any(axis=2)
                rendered_output_parts.append({'render': rendered_person, 'mask': mask})


                # --- Side View Rendering ---
                if self.args.sideview:
                     rendered_person_side = side_renderer.render(
                           img=np.zeros_like(img_rgb), # Render on black
                           verts=frame_verts,
                           cam=frame_cam, # Use same camera? Or adjust? Let's use same for now.
                           var=rend_var,
                           color=mc,
                           angle=270,
                           axis=[0, 1, 0],
                           hps_backbone=self.backbone,
                     )
                     mask_side = (rendered_person_side > 0).any(axis=2)
                     side_img_composite[mask_side] = rendered_person_side[mask_side]


                # Log uncertainty value (optional)
                log_str = f'img_f:{basename(img_fname)} person:{person_id:02d} '
                if frame_var_global is not None:
                    log_str += f'var:{frame_var_global:.3f}'
                else:
                     log_str += 'var:N/A'
                with open(f'{output_path}/uncertainty.log', "a") as f:
                    print(log_str, file=f)


            # --- Composite final image ---
            final_image = img_rgb.copy()
            for data in rendered_output_parts:
                 final_image[data['mask']] = data['render'][data['mask']]

            # --- Draw keypoints on the final composite image ---
            if self.args.draw_keypoints:
                 for person_id, person_data in sorted_persons: # Iterate again to draw KPs on top
                      frame_kp = person_data['joints2d']
                      for pt_idx, pt in enumerate(frame_kp):
                           x, y, conf = int(pt[0]), int(pt[1]), pt[2]
                           if conf > 0.1:
                                kp_color = (255, 255, 255) if pt_idx >= 25 else (0, 0, 0)
                                cv2.circle(final_image, (x, y), 3, kp_color, -1)

            # --- Combine with side view ---
            if self.args.sideview:
                final_image = np.concatenate([final_image, side_img_composite], axis=1)

            cv2.imwrite(join(output_img_folder, f'{frame_idx:06d}.png'), cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

            if self.args.display:
                cv2.imshow('Video', cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if self.args.display:
            cv2.destroyAllWindows()
