# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
import random
import json
import argparse
from PIL import Image

import cv2

from alfworld.agents.detector.engine import train_one_epoch, evaluate
import alfworld.agents.detector.utils as utils
import torchvision
import alfworld.agents.detector.transforms as T
from alfworld.agents.detector.mrcnn import get_model_instance_segmentation, load_pretrained_model

import sys
import alfworld.gen.constants as constants

MIN_PIXELS = 100

OBJECTS_DETECTOR = constants.OBJECTS_DETECTOR
STATIC_RECEPTACLES = constants.STATIC_RECEPTACLES
ALL_DETECTOR = constants.ALL_DETECTOR

def get_object_classes(object_type):
    if object_type == "objects":
        return OBJECTS_DETECTOR
    elif object_type == "receptacles":
        return STATIC_RECEPTACLES
    else:
        return ALL_DETECTOR

class AlfredDataset(object):
    def __init__(self, root, transforms, args):
        self.root = root
        self.transforms = transforms
        self.args = args
        self.object_classes = get_object_classes(args.object_types)

        # load all image files, sorting them to
        # ensure that they are aligned
        self.get_data_files(root, balance_scenes=args.balance_scenes)


    def get_data_files(self, root, balance_scenes=False):
        if balance_scenes:
            kitchen_path = os.path.join(root, 'kitchen', 'images')
            living_path = os.path.join(root, 'living', 'images')
            bedroom_path = os.path.join(root, 'bedroom', 'images')
            bathroom_path = os.path.join(root, 'bathroom', 'images')

            kitchen = list(sorted(os.listdir(kitchen_path)))
            living = list(sorted(os.listdir(living_path)))
            bedroom = list(sorted(os.listdir(bedroom_path)))
            bathroom = list(sorted(os.listdir(bathroom_path)))

            min_size = min(len(kitchen), len(living), len(bedroom), len(bathroom))
            kitchen = [os.path.join(kitchen_path, f) for f in random.sample(kitchen, int(min_size*self.args.kitchen_factor))]
            living = [os.path.join(living_path, f) for f in random.sample(living, int(min_size*self.args.living_factor))]
            bedroom = [os.path.join(bedroom_path, f) for f in random.sample(bedroom, int(min_size*self.args.bedroom_factor))]
            bathroom = [os.path.join(bathroom_path, f) for f in random.sample(bathroom, int(min_size*self.args.bathroom_factor))]

            self.imgs = kitchen + living + bedroom + bathroom
            self.masks = [f.replace("images", "masks") for f in self.imgs]
            self.metas = [f.replace("images", "meta").replace(".png", ".json") for f in self.imgs]
        else:
            self.imgs = [os.path.join(root, "images", f) for f in list(sorted(os.listdir(os.path.join(root, "images"))))]
            self.masks = [os.path.join(root, "masks", f) for f in list(sorted(os.listdir(os.path.join(root, "masks"))))]
            self.metas = [os.path.join(root, "meta", f) for f in list(sorted(os.listdir(os.path.join(root, "meta"))))]

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        meta_path = self.metas[idx]

        # print("Opening: %s" % (self.imgs[idx]))

        with open(meta_path, 'r') as f:
            color_to_object = json.load(f)

        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)

        im_width, im_height = mask.shape[0], mask.shape[1]
        seg_colors = np.unique(mask.reshape(im_height*im_height, 3), axis=0)

        masks, boxes, labels = [], [], []
        for color in seg_colors:
            color_str = str(tuple(color[::-1]))
            if color_str in color_to_object:
                object_id = color_to_object[color_str]
                object_class = object_id.split("|", 1)[0] if "|" in object_id else ""
                if "Basin" in object_id:
                    object_class += "Basin"
                if object_class in self.object_classes:
                    smask = np.all(mask == color, axis=2)
                    pos = np.where(smask)
                    num_pixels = len(pos[0])

                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])

                    # skip if not sufficient pixels
                    # if num_pixels < MIN_PIXELS:
                    if (xmax-xmin)*(ymax-ymin) < MIN_PIXELS:
                        continue

                    class_idx = self.object_classes.index(object_class)

                    masks.append(smask)
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_idx)

                    if self.args.debug:
                        disp_img = np.array(img)
                        cv2.rectangle(disp_img, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
                        cv2.putText(disp_img, object_class, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
                        sg = np.uint8(smask[:, :, np.newaxis])*255

                        print(xmax-xmin, ymax-ymin, num_pixels)
                        cv2.imshow("img", np.array(disp_img))
                        cv2.imshow("sg", sg)
                        cv2.waitKey(0)

        if len(boxes) == 0:
            return None, None

        iscrowd = torch.zeros(len(masks), dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main(args):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = len(get_object_classes(args.object_types))+1
    # use our dataset and defined transformations
    dataset = AlfredDataset(args.data_path, get_transform(train=True), args)
    dataset_test = AlfredDataset(args.data_path, get_transform(train=False), args)

    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    indices = list(range(len(dataset)))
    dataset = torch.utils.data.Subset(dataset, indices[:-4000])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-4000:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    if args.load_model:
        model = load_pretrained_model(args.load_model)
    else:
        model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)
        # save model
        model_path = os.path.join(args.save_path, "%s_%03d.pth" % (args.save_name, epoch))
        torch.save(model.state_dict(), model_path)
        print("Saving %s" % model_path)

    print("Done training!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--save_path", type=str, default="data/")
    parser.add_argument("--object_types", choices=["objects", "receptacles", "all"], default="all")
    parser.add_argument("--save_name", type=str, default="mrcnn_alfred_objects")
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)

    parser.add_argument("--balance_scenes", action='store_true')
    parser.add_argument("--kitchen_factor", type=float, default=1.0)
    parser.add_argument("--living_factor", type=float, default=1.0)
    parser.add_argument("--bedroom_factor", type=float, default=1.0)
    parser.add_argument("--bathroom_factor", type=float, default=1.0)

    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    main(args)
