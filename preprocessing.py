import os
import copy
import xml.etree.ElementTree as ET

import numpy as np
from keras.utils import Sequence
from yad2k.models.keras_yolo import preprocess_true_boxes
from PIL import Image


def parse_annotation(ann_dir, img_dir, labels=[]):
    all_imgs = []
    seen_labels = {}
    
    for ann in sorted(os.listdir(ann_dir)):
        img = {'object':[]}

        tree = ET.parse(ann_dir + ann)
        
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1
                        
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]
                            
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]
                        
    return all_imgs, seen_labels

class BatchGenerator(Sequence):
    def __init__(self, images, 
                       config,
                       shuffle=True, 
                       jitter=True, 
                       norm=None):
        self.generator = None

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter  = jitter
        self.norm    = norm

        # self.anchors = [BoundBox(0, 0, config['ANCHORS'][2*i], config['ANCHORS'][2*i+1]) for i in range(int(len(config['ANCHORS'])//2))]
        self.anchors = config['ANCHORS']

        if shuffle: np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))   

    def num_classes(self):
        return len(self.config['LABELS'])

    def size(self):
        return len(self.images)    

    def load_annotation(self, image):
        annots = []

        for obj in image['object']:
            annot = [self.config['LABELS'].index(obj['name']),
                    obj['xmin'], 
                    obj['ymin'],
                    obj['xmax'], 
                    obj['ymax']]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)
    
    def load_boxes(self, images):
        boxes = []
        for image in images:
            boxes.append(self.load_annotation(image))
        return boxes
    
    def get_detector_mask(self, boxes, anchors):
        '''
        Precompute detectors_mask and matching_true_boxes for training.
        Detectors mask is 1 for each spatial position in the final conv layer and
        anchor that should be active for the given boxes and 0 otherwise.
        Matching true boxes gives the regression targets for the ground truth box
        that caused a detector to be active or 0 otherwise.
        '''
        detectors_mask = [0 for i in range(len(boxes))]
        matching_true_boxes = [0 for i in range(len(boxes))]
        for i, box in enumerate(boxes):
            detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [608, 608])

        return np.array(detectors_mask), np.array(matching_true_boxes)
    
    def process_boxes(self, boxes):
        '''processes the boxes'''

        orig_size = np.array([[608, 608]])

        if boxes is not None:
            # Box preprocessing.
            # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
            boxes = [box.reshape((-1, 5)) for box in boxes]
            # Get extents as y_min, x_min, y_max, x_max, class for comparision with
            # model output.
            boxes_extents = [box[:, [2, 1, 4, 3, 0]] for box in boxes]

            # Get box parameters as x_center, y_center, box_width, box_height, class.
            boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]
            boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]
            boxes_xy = [boxxy / orig_size for boxxy in boxes_xy]
            boxes_wh = [boxwh / orig_size for boxwh in boxes_wh]
            boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]

            # find the max number of boxes
            max_boxes = 0
            for boxz in boxes:
                if boxz.shape[0] > max_boxes:
                    max_boxes = boxz.shape[0]

            # add zero pad for training
            for i, boxz in enumerate(boxes):
                if boxz.shape[0]  < max_boxes:
                    zero_padding = np.zeros( (max_boxes-boxz.shape[0], 5), dtype=np.float32)
                    boxes[i] = np.vstack((boxz, zero_padding))

            return np.array(boxes)

    def load_image(self, filename):
        img = Image.open(filename)
        return np.array(img)
    
    def __getitem__(self, idx):
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))

        # boxes = self.load_boxes(self.images[l_bound: r_bound])
        boxes = []
        for image in self.images[l_bound: r_bound]:
            boxes.append(self.load_annotation(image))
        
        boxes = self.process_boxes(boxes)

        it = iter(self.config['ANCHORS'])
        anchors = np.array(list(zip(it, it)))
        detectors_mask, matching_true_boxes = self.get_detector_mask(boxes, anchors)

        instance_count = 0
        for train_instance in self.images[l_bound:r_bound]:
            # assign input image to x_batch
            img = Image.open(train_instance['filename'])
            img = np.array(img, dtype='float32')
            img = img / 255.
            x_batch[instance_count] = np.array(img)

            # increase instance counter in current batch
            instance_count += 1

        #print(' new batch created', idx)
        return [x_batch, boxes, detectors_mask, matching_true_boxes],  np.zeros(len(x_batch))

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)