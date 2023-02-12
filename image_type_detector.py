import os
from argparse import Namespace

import cv2
import numpy as np
import torch
from tqdm import tqdm

from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.pPose_nms import pose_nms
from alphapose.utils.presets import SimpleTransform
from alphapose.utils.transforms import (flip, flip_heatmap,
                                        get_func_heatmap_to_coord)
from alphapose.utils.vis import getTime
from detector.apis import get_detector


class DetectionLoader():
    def __init__(self, detector, cfg, device):
        self.cfg = cfg
        self.device = device
        self.detector = detector

        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE

        self._sigma = cfg.DATA_PRESET.SIGMA

        pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
        if cfg.DATA_PRESET.TYPE == 'simple':
            self.transformation = SimpleTransform(
                pose_dataset, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False, add_dpg=False, gpu_device=self.device)

        self.image = (None, None, None, None)
        self.det = (None, None, None, None, None, None, None)
        self.pose = (None, None, None, None, None, None, None)

    def process(self, im_name, image):
        # start to pre process images for object detection
        self.image_preprocess(im_name, image)
        # start to detect human in images
        self.image_detection()
        # start to post process cropped human image for pose estimation
        self.image_postprocess()
        return self

    def image_preprocess(self, im_name, image):
        # expected image shape like (1,3,h,w) or (3,h,w)
        img = self.detector.image_preprocess(image)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        # add one dimension at the front for batch if image shape (3,h,w)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        orig_img = image # scipy.misc.imread(im_name_k, mode='RGB') is depreciated
        im_dim = orig_img.shape[1], orig_img.shape[0]

        im_name = os.path.basename(im_name)

        with torch.no_grad():
            im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

        self.image = (img, orig_img, im_name, im_dim)

    def image_detection(self):
        imgs, orig_imgs, im_names, im_dim_list = self.image
        if imgs is None:
            self.det = (None, None, None, None, None, None, None)
            return

        with torch.no_grad():
            dets = self.detector.images_detection(imgs, im_dim_list)
            if isinstance(dets, int) or dets.shape[0] == 0:
                self.det = (orig_imgs, im_names, None, None, None, None, None)
                return
            if isinstance(dets, np.ndarray):
                dets = torch.from_numpy(dets)
            dets = dets.cpu()
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]
            ids = torch.zeros(scores.shape)

        boxes = boxes[dets[:, 0] == 0]
        if isinstance(boxes, int) or boxes.shape[0] == 0:
            self.det = (orig_imgs, im_names, None, None, None, None, None)
            return
        inps = torch.zeros(boxes.size(0), 3, *self._input_size)
        cropped_boxes = torch.zeros(boxes.size(0), 4)

        self.det = (orig_imgs, im_names, boxes, scores[dets[:, 0] == 0], ids[dets[:, 0] == 0], inps, cropped_boxes)

    def image_postprocess(self):
        with torch.no_grad():
            (orig_img, im_name, boxes, scores, ids, inps, cropped_boxes) = self.det
            if orig_img is None:
                self.pose = (None, None, None, None, None, None, None)
                return
            if boxes is None or boxes.nelement() == 0:
                self.pose = (None, orig_img, im_name, boxes, scores, ids, None)
                return

            for i, box in enumerate(boxes):
                inps[i], cropped_box = self.transformation.test_transform(orig_img, box)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)

            self.pose = (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes)

    def read(self):
        return self.pose


class DataWriter():
    def __init__(self, cfg):
        self.cfg = cfg
        self.min_box_area = 0

        self.eval_joints = list(range(cfg.DATA_PRESET.NUM_JOINTS))
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        self.item = (None, None, None, None, None, None, None)

        loss_type = self.cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss')
        num_joints = self.cfg.DATA_PRESET.NUM_JOINTS
        if loss_type == 'MSELoss':
            self.vis_thres = [0.4] * num_joints
        elif 'JointRegression' in loss_type:
            self.vis_thres = [0.05] * num_joints
        elif loss_type == 'Combined':
            if num_joints == 68:
                hand_face_num = 42
            else:
                hand_face_num = 110
            self.vis_thres = [0.4] * (num_joints - hand_face_num) + [0.05] * hand_face_num

        self.use_heatmap_loss = (self.cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss') == 'MSELoss')

    def start(self):
        # start to read pose estimation results
        return self.update()

    def update(self):
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE

        # get item
        (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name) = self.item
        if orig_img is None:
            return None
        # image channel RGB->BGR
        orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
        self.orig_img = orig_img
        if boxes is None or len(boxes) == 0:
            return None
        else:
            # location prediction (n, kp, 2) | score prediction (n, kp, 1)
            assert hm_data.dim() == 4
            if hm_data.size()[1] == 136:
                self.eval_joints = [*range(0,136)]
            elif hm_data.size()[1] == 26:
                self.eval_joints = [*range(0,26)]
            elif hm_data.size()[1] == 133:
                self.eval_joints = [*range(0,133)]
            pose_coords = []
            pose_scores = []

            for i in range(hm_data.shape[0]):
                bbox = cropped_boxes[i].tolist()
                if isinstance(self.heatmap_to_coord, list):
                    pose_coords_body_foot, pose_scores_body_foot = self.heatmap_to_coord[0](
                        hm_data[i][self.eval_joints[:-110]], bbox, hm_shape=hm_size, norm_type=norm_type)
                    pose_coords_face_hand, pose_scores_face_hand = self.heatmap_to_coord[1](
                        hm_data[i][self.eval_joints[-110:]], bbox, hm_shape=hm_size, norm_type=norm_type)
                    pose_coord = np.concatenate((pose_coords_body_foot, pose_coords_face_hand), axis=0)
                    pose_score = np.concatenate((pose_scores_body_foot, pose_scores_face_hand), axis=0)
                else:
                    pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][self.eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)
                pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
            preds_img = torch.cat(pose_coords)
            preds_scores = torch.cat(pose_scores)

            boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                pose_nms(boxes, scores, ids, preds_img, preds_scores, self.min_box_area, use_heatmap_loss=self.use_heatmap_loss)

            _result = []
            for k in range(len(scores)):
                _result.append(
                    {
                        'keypoints':preds_img[k],
                        'kp_score':preds_scores[k],
                        'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                        'idx':ids[k],
                        'bbox':[boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]]
                    }
                )

            result = {
                'imgname': im_name,
                'result': _result
            }

            # if hm_data.size()[1] == 49:
            #     from alphapose.utils.vis import vis_frame_dense as vis_frame
            # elif self.opt.vis_fast:
            #     from alphapose.utils.vis import vis_frame_fast as vis_frame
            # else:
            #     from alphapose.utils.vis import vis_frame
            # self.vis_frame = vis_frame

        return result

    def save(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name):
        self.item = (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name)


class ImageTypeDetector:


    N_DIM = 3 #x,y,visibility
    N_KEYPOINTS = 17
    K_NAMES = [
        'nose',
        'left_eye', 'right_eye',
        'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle'
    ]

    def __init__(self, cfg_file, checkpoint, use_gpu=True):
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.cfg = update_config(cfg_file)
        self.args = Namespace(cfg=None, checkpoint=None, debug=False, detector='yolo', device=torch.device(type='cuda', index=0), eval=False, flip=False, format=None, gpus=[0], inputimg='', min_box_area=0, pose_flow=False, pose_track=False, profile=False, save_img=False, showbox=False, tracking=False, vis=False, vis_fast=False)
        # Load pose model
        self.pose_model = builder.build_sppe(self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET)

        print('Loading pose model from %s...' % (checkpoint,))
        self.pose_model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.pose_model.to(self.device)
        self.pose_model.eval()


        self.det_loader = DetectionLoader(get_detector(self.args), self.cfg, self.device)


    def process(self, im_name, image):
        # Init data writer
        self.writer = DataWriter(self.cfg)

        runtime_profile = {
            'dt': [],
            'pt': [],
            'pn': []
        }
        pose = None
        try:
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = self.det_loader.process(im_name, image).read()
                if orig_img is None:
                    raise Exception("no image is given")
                if boxes is None or boxes.nelement() == 0:
                    self.writer.save(None, None, None, None, None, orig_img, im_name)
                    pose = self.writer.start()
                else:
                    # Pose Estimation
                    inps = inps.to(self.device)
                    hm = self.pose_model(inps)
                    hm = hm.cpu()
                    self.writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                    pose = self.writer.start()
            # print('===========================> Finish Model Running.')
        except Exception as e:
            raise e
            print(repr(e))
            print('An error as above occurs when processing the images, please check it')
            pass
        except KeyboardInterrupt:
            print('===========================> Finish Model Running.')

        return pose


    def _get_confident_kp(self, keypoints, kp_score, confidence=0.6):
        key_dict = {}
        for i, point in enumerate(self.K_NAMES):
            if kp_score[i] > confidence:
                key_dict[point] = keypoints[i][0],  keypoints[i][1]
        return key_dict

    @staticmethod
    def is_outfit(human_keypoints):
        return all([x > 0 and y > 0 for x,y in human_keypoints.values()]) and len(human_keypoints) > 12


    @staticmethod
    def is_backward(human_keypoints):
        try:
            if human_keypoints['left_shoulder'][0] < human_keypoints['right_shoulder'][0] and human_keypoints['left_hip'][0] < human_keypoints['right_hip'][0]:
                return True
            else:
                return False

        except:
            return False

    @staticmethod
    def is_frontal(human_keypoints):
        return len(human_keypoints) > 7




    def get_image_type(self, pose):
        if not pose:
            return 'undefined'
        if len(pose['result']) > 1:
            return 'multi'

        human_keypoints = self._get_confident_kp(pose['result'][0]['keypoints'], pose['result'][0]['kp_score'])
        if self.is_backward(human_keypoints):
            return 'backward'
        elif self.is_outfit(human_keypoints):
            return 'outfit'
        elif self.is_frontal(human_keypoints):
            return 'frontal'
        else:
            return 'undefined'


if __name__ == '__main__':
    img_path = ''

    from glob import glob

    image_type_detector = ImageTypeDetector(
        'configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml',
        'pretrained_models/fast_res50_256x192.pth',
    )

    garment_id = 'web21b02p-i11'
    img_path = '../ai_stylist/zalando/images/web21b02p-i11/bodegon.jpg'
    img_path = '../ai_stylist/Fnllu_kXoAIXo4W.jpeg'
    for img_path in list(glob(f'../ai_stylist/zalando/images/{garment_id}/*.jpg')):
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        pose = image_type_detector.process(img_path, image)
        image_type = image_type_detector.get_image_type(pose)
        print(os.path.basename(img_path), image_type)
