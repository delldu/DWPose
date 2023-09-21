"""Image Recognize Anything Model Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, Thu 13 Jul 2023 01:55:56 PM CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
from PIL import Image, ImageDraw

import torch
import todos
from torchvision.transforms import ToTensor, ToPILImage
from .dwpose import DWPose
from .mmpose import KEYPOINTS_COLOR, LINK_COLOR, POSE_SKELETON

import pdb


def draw_points(tensor, points):
    B, C, H, W = tensor.size()

    image = ToPILImage()(tensor.squeeze(0))
    draw = ImageDraw.Draw(image)
    points = points.cpu()

    X = []
    Y = []
    V = []
    for point in points:
        x = int(point[0].item())
        y = int(point[1].item())
        s = float(point[2].item())

        X.append(x)
        Y.append(y)
        if x < 0 or x >= W or y < 0 or y >= H or s < 0.1:
            V.append(False)
        else:
            V.append(True)

    # Draw points
    for i, v in enumerate(V):
        if v:
            color = KEYPOINTS_COLOR[i]
            x = X[i]
            y = Y[i]
            draw.ellipse(((x, y), (x + 2, y + 2)), fill=tuple(color), width=1)


    # Draw links
    for i, sk in enumerate(POSE_SKELETON):
        p1 = sk[0]
        p2 = sk[1]
        if not (V[p1] and V[p2]):
            continue

        color = LINK_COLOR[i]
        x1 = X[p1]
        y1 = Y[p1]
        x2 = X[p2]
        y2 = Y[p2]
        draw.line(((x1, y1), (x2, y2)), fill=tuple(color), width=1)

    image = ToTensor()(image)

    return image.unsqueeze(0)


def create_model():
    """
    Create model
    """
    model = DWPose()

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    print(f"Running model on {device} ...")

    return model, device


def get_model():
    """Load jit script model."""

    model, device = create_model()
    # print(model)

    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;
    # torch::jit::setTensorExprFuserEnabled(false);

    model = torch.jit.script(model)
    todos.data.mkdir("output")
    if not os.path.exists("output/DWPose.torch"):
        model.save("output/DWPose.torch")

    return model, device


def predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        image = Image.open(filename).convert("RGB")
        input_image = ToTensor()(image).unsqueeze(0).to(device)
        input_backup = input_image.clone()

        with torch.no_grad():
            output_points = model(input_image)

        output_file = f"{output_dir}/{os.path.basename(filename)}"
        output_image =  draw_points(input_backup, output_points[0])
        todos.data.save_tensor([input_backup, output_image], output_file)

    progress_bar.close()

    todos.model.reset_device()
