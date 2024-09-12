import os
import warnings
from argparse import ArgumentParser
from pathlib import Path

warnings.filterwarnings("ignore")

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MatthewsCorrCoef, PrecisionRecallCurve
# from torcheval.metrics import MulticlassAUPRC, MulticlassAccuracy, 
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pytorch_lightning.loggers import TensorBoardLogger

import sklearn

import datetime
import pandas as pd
import numpy as np

from sklearn.metrics import auc

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    Resize,
    RandomHorizontalFlip
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorchvideo.models.resnet
import pytorchvideo
from pytorchvideo.data import Kinetics

from collections import Counter
from os.path import join as opj
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import DeviceStatsMonitor
import random
import PIL
import torchvision
# torch.use_deterministic_algorithms(True, warn_only=True) # https://github.com/pytorch/pytorch/issues/89492

class IdentityTransform:
    def __call__(self, sample):
        """
        Args:
            sample (PIL Image or Tensor): Image or tensor to be transformed.

        Returns:
            PIL Image or Tensor: Same input image or tensor.
        """
        return sample

class ColorJitterVideo(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip

    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, data_path=None):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.data_path = data_path

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor
    
    def apply_transform(self, img, brightness, contrast, saturation, hue):

        # Apply transformations
        if brightness is not None:
            img = torchvision.transforms.functional.adjust_brightness(img, brightness)
        if contrast is not None:
            img = torchvision.transforms.functional.adjust_contrast(img, contrast)
        if saturation is not None:
            img = torchvision.transforms.functional.adjust_saturation(img, saturation)
        if hue is not None:
            img = torchvision.transforms.functional.adjust_hue(img, hue)

        return img
    
    def __call__(self, clip):
        """
        Args:
        clip (list): list of PIL.Image

        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        if isinstance(clip[0], np.ndarray):
            raise TypeError(
                'Color jitter not yet implemented for numpy arrays')
        elif isinstance(clip[0], PIL.Image.Image):
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)

            # Apply to all images
            jittered_clip = []
            for img in clip:
                for func in img_transforms:
                    jittered_img = func(img)
                jittered_clip.append(jittered_img)
        elif isinstance(clip[0], torch.Tensor):  # List of Tensors
                brightness, contrast, saturation, hue = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
                clip_dimension = 1
                bcsh_str = f"b{brightness:.6f}"
                # print("clip.shape", clip.shape)
                # for tensor in clip:
                #     print(tensor.shape)
                jittered_clip = [self.apply_transform(torchvision.transforms.functional.to_pil_image(clip[:, i, :, :]), brightness, contrast, saturation, hue) for i in range(clip.shape[clip_dimension])]
                if self.data_path is not None:
                    save_dir = '/lsdf/kit/iai/projects/iai-aida/Daten_Deininger/projects/embryo_project/training_datasets/videos_allt/results/epochs100_mvit_lr-autofind_cd8_uts8_bs1_img_size224_wd0.0001_opt-sgd_cj-strength0.1_config2/split0'
                    save_name = opj(save_dir, os.path.basename(self.data_path)+ f'_{bcsh_str}_')
                    # print(self.data_path)
                    clip_pil = [torchvision.transforms.functional.to_pil_image(clip[:, i, :, :]) for i in range(clip.shape[clip_dimension])]
                    for i, c in enumerate(clip_pil):
                        c.save(save_name+ f'_{i}.png')
                    for i, c in enumerate(jittered_clip):
                        c.save(save_name+ f'{i}_cj.png')
                jittered_clip = torch.stack([torchvision.transforms.functional.to_tensor(img) for img in jittered_clip], dim=clip_dimension)
                # print("jittered_clip.shape", jittered_clip.shape)
                assert clip.shape == jittered_clip.shape
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        return jittered_clip


def make_kinetics_resnet():
    return 

class KineticsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path,
        val_path,
        batch_size,
        clip_dur,
        unif_temp_sub=16,
        test_path=None,
        num_workers=8,
        vit_img_size=224,
        cj_bn=0.0,
        norm=True,
        div255=True
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.unif_temp_sub = unif_temp_sub
        self.model = model
        self.vit_img_size = vit_img_size
        self.norm = norm
        self.div255 = div255     


        self.clip_dur = clip_dur
        self.cj_bn = cj_bn

    def _dataloader(self, data_path, mode):
        # values here are specific to pneumonia dataset and should be updated for custom data
        if self.model=='resnet':
            transform = Compose(
                [
                ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(self.unif_temp_sub),### subsamples 8 images from T in range 0-153, so it will select at the beginning, middle and end, so probably day 0, 20, 40, 60, 80, 100, 120, 140 (these are 8)
                        Lambda(lambda x: x / 255.0) if self.div255 else IdentityTransform(),
                        Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)) if self.norm else IdentityTransform(),
                        # RandomShortSideScale(min_size=256, max_size=320),
                        # RandomCrop(244),
                        RandomHorizontalFlip(p=0.5),
                    ]
                    ),
                ),
                ]
            )
        else:
            transform = Compose(
                [
                ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(self.unif_temp_sub),### subsamples 8 images from T in range 0-153, so it will select at the beginning, middle and end, so probably day 0, 20, 40, 60, 80, 100, 120, 140 (these are 8)
                        Lambda(lambda x: x / 255.0) if self.div255 else IdentityTransform(),
                        Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)) if self.norm else IdentityTransform(),
                        ColorJitterVideo(brightness=self.cj_bn) if self.cj_bn != 0.0 else IdentityTransform(),
                        Resize(self.vit_img_size),
                        RandomHorizontalFlip(p=0.5),
                    ]
                    ),
                ),
                ]
            )
        dataset = Kinetics(
            data_path=os.path.join(data_path),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform" if mode!='train' else 'uniform', self.clip_dur), #original : random, what is this?
            decode_audio=False,
            transform=transform
        )
        
        print(mode, "dataset.num_videos:", dataset.num_videos)
        return torch.utils.data.DataLoader(dataset,
                                            batch_size=self.batch_size,
                                            num_workers=self.num_workers
        )   

    def train_dataloader(self):
        return self._dataloader(self.train_path, mode='train')
    
    def val_dataloader(self):
        return self._dataloader(self.val_path, mode='val')
    
    def test_dataloader(self):
        return self._dataloader(self.test_path, mode='test')
    
class VideoClassificationLightningModule(pl.LightningModule):
    def __init__(self,
                 model,
                 num_classes,
                 ce_weights,
                 test_path,
                 nr_test_videos,
                 save_path,
                 optimizer_str,
                 lr=1e-1,
                 weight_decay=0,
                 vit_img_size=224,
                 unif_temp_sub=154):
        super().__init__()
        self.optimizer_str = optimizer_str
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.test_path = test_path
        self.save_path = save_path
        self.nr_test_videos = nr_test_videos
        self.vit_img_size = vit_img_size
        self.model=model
        self.unif_temp_sub=unif_temp_sub
        if model=='resnet':
            self.model = pytorchvideo.models.resnet.create_resnet(
                        input_channel=3, # RGB input from Kinetics
                        model_depth=50, # For the tutorial let's just use a 50 layer network
                        model_num_class=num_classes, # Kinetics has 400 classes so we need out final head to align
                        norm=nn.BatchNorm3d,
                        activation=nn.ReLU
            )
        else:
            # self.model = torch.load("/home/iai/oc9627/MVIT_B_16x4.pyth")
            self.model = torch.hub.load("facebookresearch/pytorchvideo", model='mvit_base_16x4', pretrained=True)
            self.model.head.proj = nn.Linear(in_features=768, out_features=self.num_classes, bias=True)     

        self.loss_fn = (
                nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ce_weights[1])) if self.num_classes == 1 else nn.CrossEntropyLoss(weight=torch.tensor(ce_weights))
            )
                # create accuracy metric
        self.acc = Accuracy(
            task="binary" if num_classes == 1 else "multiclass", num_classes=self.num_classes
        )
        # self.prcurve = PrecisionRecallCurve(task='multiclass', num_classes=self.num_classes, average=None)

        # self.multacc = Accuracy(task='multiclass', num_classes=num_classes, average=None)
        self.multacc = Accuracy(task='binary' if num_classes == 1 else "multiclass", num_classes=num_classes)

        
        self.class_idx = {}
        self.class_idx['abnormal'] = 0
        self.class_idx['normal'] = 1

        # self.df = self.get_test_gt()

        self.test_filenames, self.test_predictions, self.test_targets = [], [], []

    
    def forward(self, x):
        return self.model(x)
    
    def get_prauc(self, prcurve):
                # Calculate AUC for each class
        class_auc_list = []
        for precision, recall, thresholds in zip(prcurve[0], prcurve[1], prcurve[2]):
            mask = ~torch.isnan(recall)
            # Check if there are enough points to compute AUC
            if len(mask.nonzero()) >= 2:
                class_auc = auc(recall[mask].cpu().numpy(), precision[mask].cpu().numpy())
                class_auc_list.append(class_auc)
            else:
                # print("Not enough points to compute AUC for this class.")
                class_auc_list.append(np.nan)

        return class_auc_list
    
    def _step(self, batch, mode):
        x = batch["video"]
        y = batch["label"]
        preds = self.model(x)

        if self.num_classes == 1:
            preds = preds.flatten()
            y = y.float()

        loss = self.loss_fn(preds, y)
        acc = self.acc(preds, y)
        multacc = self.multacc(preds, y)
        # prcurve = self.prcurve(preds, y)
        # prauc = self.get_prauc(prcurve)
        self.log(
            f"{mode}/Loss", loss, on_epoch=True, prog_bar=True, logger=True
        )
        mode_lower = mode.lower()
        self.log(
            f"{mode_lower}_loss", loss, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            f"{mode}/Acc", acc, on_epoch=True, prog_bar=True, logger=True
        )

        if mode=='Test':
            return loss, preds.cpu().numpy(), y.cpu().numpy(), np.array(batch['video_name'])
        else:
            return loss


    def training_step(self, batch, batch_idx):
        # print('using lr: ', self.lr)
        # The model expects a video tensor of shape (B, C, T, H, W), which is the
        # format provided by the dataset
        loss = self._step(batch, mode='Train')

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, mode='Val')

        return loss
    
    def test_step(self, batch, batch_idx):
        _, preds, targets, filenames = self._step(batch, mode='Test')
        if len(targets) > 1:
            self.test_targets.extend(np.squeeze(targets).tolist())
            self.test_filenames.extend(np.squeeze(filenames).tolist())
            self.test_predictions.extend(np.squeeze(preds).tolist())
        else:
            self.test_targets.extend(targets.tolist())  
            self.test_filenames.extend(filenames.tolist())  
            self.test_predictions.extend(preds.tolist())  
        print(len(self.test_predictions), self.nr_test_videos, flush=True)
        print("len(self.test_filenames), len(self.test_predictions), len(self.test_targets):")
        print(len(self.test_filenames), len(self.test_predictions), len(self.test_targets))
        print("self.test_filenames:", self.test_filenames)
        print("self.test_predictions:", self.test_predictions)
        print("self.test_targets:", self.test_targets)
        if len(self.test_predictions) == self.nr_test_videos:
            print('saving output csv')
            df = pd.DataFrame.from_dict({'filename': self.test_filenames, 
                                         'predicted': self.test_predictions,
                                         'target_2': self.test_targets})
            df['predicted_cls'] = df['predicted'].apply(lambda x: np.argmax(x))
            df.to_csv(os.path.join(self.save_path, 'outputs.csv'), index=False)

    def configure_optimizers(self):
        """
        Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
        usually useful for training video models.
        """
        print('using lr: ', self.lr)
        print('using weight decay: ', self.weight_decay)
        if self.optimizer_str=='adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_str=='sgd':
            return torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

def get_nr_videos_in_test_dir(dir, nr_classes):
    nr_videos = 0
    nr_videos += len(os.listdir(opj(dir, 'abnormal'))) 
    nr_videos += len(os.listdir(opj(dir, 'normal')))
    return nr_videos


if __name__ == "__main__":
    parser = ArgumentParser()
    # Required arguments
    parser.add_argument(
        "model",
        help="""Choose one of the predefined ResNet models provided by torchvision. e.g. 50""",
        type=str,
        choices=['resnet', 'mvit'],
    )
    parser.add_argument(
        "num_classes", help="""Number of classes to be learned.""", type=int
    )
    parser.add_argument("num_epochs", help="""Number of Epochs to Run.""", type=int)
    parser.add_argument(
        "train_set", help="""Path to training data folder.""", type=Path
    )
    parser.add_argument("val_set", help="""Path to validation set folder.""", type=Path)
    # Optional arguments
    parser.add_argument(
        "-amp",
        "--mixed_precision",
        help="""Use mixed precision during training. Defaults to False.""",
        action="store_true",
    )
    parser.add_argument(
        "-ts", "--test_set", help="""Optional test set path.""", type=Path
    )
    parser.add_argument('-cew','--ce_weights', nargs='+', type=float, default=None, required=False)
    parser.add_argument(
        "-o",
        "--optimizer",
        help="""PyTorch optimizer to use. Defaults to adam.""",
        default="adam",
        choices = ['adam', 'sgd']
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="Adjust learning rate of optimizer.",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "-wd",
        "--weight_decay",
        help="Adjust learning rate of optimizer.",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--cj_bn",
        help="Adjust learning rate of optimizer.",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "-cd",
        "--clip_duration",
        type=int,
        default=8,
    )
    parser.add_argument(
        "-uts",
        "--unif_temp_sub",
        type=int,
        default=8,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="""Manually determine batch size. Defaults to 16.""",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--vit_img_size",
        help="""Manually determine batch size. Defaults to 16.""",
        type=int,
        default=224,
    )
    parser.add_argument(
        "-tr",
        "--transfer",
        help="""Determine whether to use pretrained model or train from scratch. Defaults to True.""",
        action="store_true",
    )
    parser.add_argument(
        "-to",
        "--tune_fc_only",
        help="Tune only the final, fully connected layers.",
        action="store_true",
    )
    parser.add_argument(
        "--no_img_norm",
        action="store_true",
    )
    parser.add_argument(
        "--no_img_div255",
        action="store_true"
    )
    parser.add_argument(
        "--ckpth_best",
        help="""for model testing: best model checkpoint""",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-uaflr",
        "--use_autofindlr",
        help="use_autofindlr",
        action="store_true",
    )
    parser.add_argument(
        "--test_only",
        help="do not train the model, only test model",
        action="store_true",
    )
    parser.add_argument(
        "-s", "--save_path", help="""Path to save model trained model checkpoint."""
    )
    parser.add_argument(
        "-g", "--gpus", help="""Enables GPU acceleration.""", type=int, default=None
    )
    parser.add_argument(
        "-tb_outdir", "--tb_outdir", help="""tb_outdir""", type=Path
    )
    args = parser.parse_args()
    # Print parsed arguments
    for arg, value in vars(args).items():
        print(f'{arg}: {value}')    
    seed_everything(42)  
    model = VideoClassificationLightningModule(model=args.model,
                                               num_classes=args.num_classes,
                                               ce_weights=args.ce_weights,
                                               test_path = args.test_set,
                                               nr_test_videos = get_nr_videos_in_test_dir(args.test_set, args.num_classes) if args.test_set is not None else 0,
                                               optimizer_str=args.optimizer,
                                               lr=args.learning_rate,
                                               save_path=args.save_path,
                                               vit_img_size=args.vit_img_size,
                                               unif_temp_sub=args.unif_temp_sub,
                                               weight_decay=args.weight_decay)
    data_module = KineticsDataModule(train_path = args.train_set,                  
                                    val_path = args.val_set,
                                    test_path = args.test_set,
                                    clip_dur=args.clip_duration,
                                    unif_temp_sub = args.unif_temp_sub,
                                    batch_size=args.batch_size,
                                    vit_img_size = args.vit_img_size,
                                    cj_bn = args.cj_bn,
                                    div255= not args.no_img_div255,
                                    norm= not args.no_img_norm
)

    save_path = args.save_path if args.save_path is not None else "./models"
    print("save_path:", save_path)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_path,
        filename="resnet-model-{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_last=False,
    )

    stopping_callback = pl.callbacks.EarlyStopping(monitor='val_loss')
    if args.use_autofindlr:
        auto_findlr_callback = pl.callbacks.LearningRateFinder()
        callbacks = [checkpoint_callback, auto_findlr_callback]
    else:
        callbacks = [checkpoint_callback]

    # Instantiate lightning trainer and train model
    trainer_args = {
        "accelerator": "gpu",
        "devices": "auto",
        "strategy": "auto",
        "max_epochs": args.num_epochs,
        "callbacks": callbacks,
        "precision": 16,
        "logger": TensorBoardLogger(args.tb_outdir, name="my_model"),
        "log_every_n_steps": 10**10,
        "deterministic": False
    }
    print("args.use_autofindlr", args.use_autofindlr)
    trainer = pl.Trainer(**trainer_args)
    if not args.test_only:
        trainer.fit(model, data_module)
    else:
        assert args.test_set is not None
        assert args.ckpth_best is not None
        trainer.test(model, data_module, ckpt_path=args.ckpth_best)
