import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from segmentation_models_pytorch.deeplabv3.model import DeepLabV3

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image
import cv2
import albumentations as A

import time
import os
from tqdm.notebook import tqdm

import wandb

from torchsummary import summary
import segmentation_models_pytorch as smp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# path
IMAGE_PATH = "/home/vision/sy_ws/data2/JPEGImages/"
MASK_PATH = "/home/vision/sy_ws/data2/SegmentationClassPNG/"

# project name
project_name = "none"

# model
model_name = "UNet++"
encoder_name = "mobilenet_v2"
trial = "12"
n_classes = 13
model = smp.DeepLabV3Plus(
    encoder_name=encoder_name,
    classes=n_classes,
    encoder_weights="imagenet",
    activation=None,
    encoder_depth=5,
    decoder_channels=256,
    encoder_output_stride=16,
    upsampling=4,
    decoder_atrous_rates=(12, 24, 36),
)

max_lr = 1e-3
epoch = 1
weight_decay = 1e-4
batch_size = 4

# 데이터 이름이 들어있는 csv파일 불러오기
df = pd.read_csv("data_name.csv")

print("Total Images: ", len(df))

# split data
X_trainval, X_test = train_test_split(df["id"].values, test_size=0.1, random_state=19)
X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=19)

print("Train Size   : ", len(X_train))
print("Val Size     : ", len(X_val))
print("Test Size    : ", len(X_test))


class OutdoorDataset(Dataset):
    def __init__(self, img_path, mask_path, X, mean, std, transform=None, patch=False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.patches = patch
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        img = np.asarray(Image.open(self.img_path + self.X[idx] + "png").convert("RGB"))
        mask = np.asarray(Image.open(self.mask_path + self.X[idx] + "png"))

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug["image"])
            mask = aug["mask"]

        if self.transform is None:
            img = Image.fromarray(img)

        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        mask = torch.from_numpy(mask).long()

        if self.patches:
            img, mask = self.tiles(img, mask)

        return img, mask

    def tiles(self, img, mask):

        img_patches = img.unfold(1, 320, 320).unfold(2, 480, 480)
        img_patches = img_patches.contiguous().view(3, -1, 320, 480)
        img_patches = img_patches.permute(1, 0, 2, 3)

        mask_patches = mask.unfold(0, 320, 320).unfold(1, 480, 480)
        mask_patches = mask_patches.contiguous().view(-1, 320, 480)

        return img_patches, mask_patches


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

t_train = A.Compose(
    [
        A.Resize(320, 480, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.GridDistortion(p=0.2),
        A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
        A.GaussNoise(),
    ]
)

t_val = A.Compose(
    [
        A.Resize(320, 480, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(),
        A.GridDistortion(p=0.2),
    ]
)

# datasets
train_set = OutdoorDataset(IMAGE_PATH, MASK_PATH, X_train, mean, std, t_train, patch=True)
val_set = OutdoorDataset(IMAGE_PATH, MASK_PATH, X_val, mean, std, t_val, patch=True)


def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def mIoU(pred_mask, mask, smooth=1e-10, n_classes=n_classes):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes):  # loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0:  # no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def fit(
    epochs,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    model_name,
    encoder_name,
    patch=True,
):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1
    not_improve = 0

    model.to(device)
    # wandb.init(project=project_name)
    # wandb.watch(model, log="all", log_freq=10)

    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        # training loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            # training phase
            image_tiles, mask_tiles = data

            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()
                image_tiles = image_tiles.view(-1, c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)
            # forward
            output = model(image)
            loss = criterion(output, mask)
            # evaluation metrics
            iou_score += mIoU(output, mask)
            # wandb.log({"train_iou_score": iou_score / len(train_loader)})
            accuracy += pixel_accuracy(output, mask)
            # wandb.log({"train_accuarcy": accuracy / len(train_loader)})
            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

            # step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()

            # wandb train loss
            # wandb.log({"train_losses": running_loss / len(train_loader)})

        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0
            # validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    # reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles = data

                    if patch:
                        bs, n_tiles, c, h, w = image_tiles.size()

                        image_tiles = image_tiles.view(-1, c, h, w)

                        mask_tiles = mask_tiles.view(-1, h, w)

                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)

                    output = model(image)
                    # evaluation metrics
                    val_iou_score += mIoU(output, mask)
                    # wandb.log({"val_iou_score": val_iou_score / len(val_loader)})
                    test_accuracy += pixel_accuracy(output, mask)
                    # wandb.log({"val_accuracy": test_accuracy / len(val_loader)})
                    # loss
                    loss = criterion(output, mask)
                    test_loss += loss.item()

                    # wandb loss
                    # wandb.log({"Epoch": e, "val_losses": test_loss / len(val_loader)})
            # calculatio mean for each batch
            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(val_loader))

            if min_loss > (test_loss / len(val_loader)):
                print(
                    "Loss Decreasing.. {:.3f} >> {:.3f} ".format(
                        min_loss, (test_loss / len(val_loader))
                    )
                )
                min_loss = test_loss / len(val_loader)
                decrease += 1
                if decrease % 5 == 0:
                    print("saving model...")
                    # checkpoint = {
                    #     "state_dict": model.state_dict(),
                    #     "optimizer": optimizer.state_dict(),
                    # }
                    # torch.save(
                    #     checkpoint,
                    #     "/home/vision/sy_ws/pt_file/{0}/checkpoint/{0}-{1}-{2:.3f}.pt".format(
                    #         model_name, encoder_name, val_iou_score / len(val_loader)
                    #     ),
                    # )

            if (test_loss / len(val_loader)) > min_loss:
                not_improve += 1
                min_loss = test_loss / len(val_loader)
                print(f"Loss Not Decrease for {not_improve} time")
                if not_improve == 7:
                    print("Loss not decrease for 7 times, Stop Training")
                    break

            # iou
            val_iou.append(val_iou_score / len(val_loader))
            train_iou.append(iou_score / len(train_loader))
            train_acc.append(accuracy / len(train_loader))
            val_acc.append(test_accuracy / len(val_loader))

            print(
                "Epoch:{}/{}..".format(e + 1, epochs),
                "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
                "Val Loss: {:.3f}..".format(test_loss / len(val_loader)),
                "Train mIoU:{:.3f}..".format(iou_score / len(train_loader)),
                "Val mIoU: {:.3f}..".format(val_iou_score / len(val_loader)),
                "Train Acc:{:.3f}..".format(accuracy / len(train_loader)),
                "Val Acc:{:.3f}..".format(test_accuracy / len(val_loader)),
                "Time: {:.2f}m".format((time.time() - since) / 60),
            )

    history = {
        "train_loss": train_losses,
        "val_loss": test_losses,
        "train_miou": train_iou,
        "val_miou": val_iou,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "lrs": lrs,
    }

    print("Total time: {:.2f} m".format((time.time() - fit_time) / 60))
    # wandb.log({"Train time": (time.time() - fit_time) / 60})
    return history


# dataloader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr, epochs=epoch, steps_per_epoch=len(train_loader)
)

history = fit(
    epoch, model, train_loader, val_loader, criterion, optimizer, sched, model_name, encoder_name
)
# checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
# torch.save(
#     checkpoint, "/home/vision/sy_ws/pt_file/{0}/{1}-{2}.pt".format(model_name, encoder_name, trial)
torch.save(
    model, "/home/vision/sy_ws/model_save/{0}/{1}-{2}.pt".format(model_name, encoder_name, trial)
)

# visualization
def plot_loss(history):
    plt.plot(history["val_loss"], label="val", marker="o")
    plt.plot(history["train_loss"], label="train", marker="o")
    plt.title("Loss per epoch")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(), plt.grid()
    # plt.show()


def plot_score(history):
    plt.plot(history["train_miou"], label="train_mIoU", marker="*")
    plt.plot(history["val_miou"], label="val_mIoU", marker="*")
    plt.title("Score per epoch")
    plt.ylabel("mean IoU")
    plt.xlabel("epoch")
    plt.legend(), plt.grid()
    # plt.show()


def plot_acc(history):
    plt.plot(history["train_acc"], label="train_accuracy", marker="*")
    plt.plot(history["val_acc"], label="val_accuracy", marker="*")
    plt.title("Accuracy per epoch")
    plt.ylabel("Accuracy")
    plt.xlabel("epoch")
    plt.legend(), plt.grid()
    # plt.show()


class OutdoortestDataset(Dataset):
    def __init__(self, img_path, mask_path, X, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        img = np.asarray(Image.open(self.img_path + self.X[idx] + "png").convert("RGB"))
        mask = np.asarray(Image.open(self.mask_path + self.X[idx] + "png"))

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug["image"])
            mask = aug["mask"]

        if self.transform is None:
            img = Image.fromarray(img)

        mask = torch.from_numpy(mask).long()

        return img, mask


t_test = A.Resize(320, 480, interpolation=cv2.INTER_NEAREST)
test_set = OutdoortestDataset(IMAGE_PATH, MASK_PATH, X_test, transform=t_test)


def predict_image_mask_miou(
    model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        score = mIoU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score


def predict_image_mask_pixel(
    model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        acc = pixel_accuracy(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, acc


def miou_score(model, test_set):
    score_iou = []
    test_time = time.time()
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, score = predict_image_mask_miou(model, img, mask)
        score_iou.append(score)
    print("test time :", time.time() - test_time)
    # wandb.log({"inference_time": time.time() - test_time})
    return score_iou


def pixel_acc(model, test_set):
    accuracy = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, acc = predict_image_mask_pixel(model, img, mask)
        accuracy.append(acc)
    return accuracy


mob_miou = miou_score(model, test_set)
mob_acc = pixel_acc(model, test_set)
# wandb.log({"Test set mIoU": np.mean(mob_miou)})
# wandb.log({"Test set pixel accuracy": np.mean(mob_acc)})
print("Test Set mIoU(DeepLabV3+)", np.mean(mob_miou))
print("Test Set Pixel Accuracy(DeepLabV3+)", np.mean(mob_acc))


image, mask = test_set[0]
pred_mask, score = predict_image_mask_miou(model, image, mask)

# for i in range(len(test_set)):
#     image, mask = test_set[i]
#     pred_mask, score = predict_image_mask_miou(model, image, mask)
#     b = np.asarray(mask)
#     a = np.asarray(pred_mask)
#     wandb.log(
#         {
#             "test_iou_score": score,
#             "mask": [wandb.Image(plt.imshow(mask), caption=f"{np.unique(b)}")],
#             "pred_mask": [wandb.Image(plt.imshow(pred_mask), caption=f"{np.unique(a)}")],
#         }
#     )

# )


# 실제 정답이미지에 존재하는 라벨
b = np.asarray(mask)
print("real answer:", np.unique(b))
# 실제 정답이미지에 존재하는 라벨
a = np.asarray(pred_mask)
print("predict answer:", np.unique(a))

