import os
import csv
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
import albumentations as A
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import segmentation_models_pytorch as smp

from pathlib import Path
from torch.utils import data
from torch.nn.functional import one_hot
from torch.optim import lr_scheduler

from torch import nn
from torchvision import models
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from albumentations.pytorch import ToTensorV2


parser = argparse.ArgumentParser(description="Age prediction model")
parser.add_argument("--train", action='store_true', help="Training model", default=False)
parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file to load into model", default=None)
parser.add_argument("--predict", action='store_true', help="Predicting from model")
parser.add_argument("--dataPath", type=str, help="Path to data. If training should have train and val subfolders, if predicting should have subfolder called 'to_predict'")

args = parser.parse_args()

assert args.train == False or args.predict == False, "Cannot both train and predict. Pick one."



def generate_gradcam(model, image_tensor, orig_image_np, save_path):
    model.eval()

    target_layer = model.model.layer4[-1]
    device = next(model.parameters()).device
    model = model.to(device)

    class RegressionModelOutputTarget:
        def __init__(self, output_index):
            self.output_index = output_index

        def __call__(self, model_output):
            if model_output.dim() == 1:
                return model_output[self.output_index]
            return model_output[:, self.output_index]

    with torch.enable_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device).requires_grad_()
        model_output = model(image_tensor)
        targets = [RegressionModelOutputTarget(0)]

        cam = GradCAM(model=model, target_layers=[target_layer])
        with cam:
            grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]

    image_np = orig_image_np.astype(np.float32) / 255.0
    if image_np.shape[0] == 3:
        image_np = np.transpose(image_np, (1, 2, 0))

    grayscale_cam = cv2.resize(grayscale_cam, (image_np.shape[1], image_np.shape[0]))
    cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)

    cv2.imwrite(save_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))


class wing_dataset(data.Dataset):
    def __init__(self, dataPath='wings', _set='train', transform = None, augment=None, predict=False):
        self.set = _set
        self.dataPath = dataPath
        self.transform = transform 
        self.augment = augment
        self.predict = predict
        self.to_tensor = A.Compose([ToTensorV2()])

        self.root = Path(f'{self.dataPath}')
        self.image_path = self.root / self.set
        self.ages_path = self.root / 'ages.csv'
        self.ages = pd.read_csv(self.ages_path, index_col=0)['Age']
        self.img_list = self.get_filenames(self.image_path)

        
    def __len__(self):
        return(len(self.img_list))
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        fp = self.img_list[idx]
        fn = Path(fp).name
        
        if not self.predict:
            age = int(self.ages[fn])
        else:
            age = 0
        
        if self.transform:
            img = self.transform(image=img)['image']
                
        if self.augment:
            img = self.augment(image=img)['image']
        img = self.to_tensor(image=img)['image']

        return {'image': img, 'target': age, 'fp': fp, 'fn': fn}
    
    def get_filenames(self, path):
        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list

class WingAgePredictionExperiment(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

        params = smp.encoders.get_preprocessing_params("resnext50_32x4d")
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        self.loss_fn = nn.MSELoss()

    def forward(self, images):
        images = (images - self.mean) / self.std
        pred = self.model(images) #['out'] # out for pytorch model
        return pred
    
    def training_step(self, batch, batch_idx):
        images = batch['image'].to(self.device).float()
        targets = batch['target'].to(self.device).float()
        pred = self.forward(images).squeeze()
        loss = self.loss_fn(pred, targets)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['image'].to(self.device).float()
        targets = batch['target'].to(self.device).float()

        pred = self.forward(images).squeeze()
        val_loss = self.loss_fn(pred, targets)

        self.log("val_loss", val_loss)

    def predict_step(self, batch, batch_idx):
        images = batch['image'].to(self.device).float()
        fps = batch['fp']
        fns = [x.split('/')[-1] for x in fps]

        preds = self.forward(images)
        predictions = []

        for i, fn in enumerate(fns):
            pred = preds[i]

            if pred.dim() != 1:
                raise ValueError(f"Unexpected shape for pred: {pred.shape}, expected [1] for regression")

            pred_value = pred.item()
            predictions.append((fn, pred_value))

        csv_path = os.path.join(self.trainer.datamodule.dataPath, "AgePredictions.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(["Filename", "PredictedAge"])
            writer.writerows(predictions)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def on_validation_epoch_end(self) -> None:
        self.save_predictions()
    
    def save_predictions(self):
        fns = []
        labels = []
        preds = []
        _sets = []
        for _set in ['val','train']:
            if _set == 'train':
                dataloader = self.trainer.datamodule.train_dataloader()
            else:
                dataloader = self.trainer.datamodule.val_dataloader()

            for batch in dataloader:
                input, label, fn = batch['image'], batch['target'], batch['fn']
                input = input.to(self.device).float()
                labels += label.cpu().detach().numpy().tolist()
                preds += self.forward(input).cpu().detach().squeeze().numpy().tolist()
                fns += fn
                _sets += [_set]*len(fn)

        out = pd.DataFrame(fns, columns=['Filename'])
        out['labels'] = labels
        out['preds'] = preds
        out['set'] = _sets
        out.to_csv(os.path.join(
                self.logger.log_dir,
                f"Epoch_{self.current_epoch}_pred.csv",
            ))


class AgeRegressionData(pl.LightningDataModule):
    def __init__(self, dataPath: str = "path/to/dir"):
        super().__init__()
        self.dataPath = dataPath
    
    def train_dataloader(self):
        transform = A.Compose([A.Rotate(25, border_mode=cv2.BORDER_CONSTANT, value=255, mask_value=0),
                                        A.Affine(translate_percent=(0.05,0.05),scale=(0.85,1.05),shear=1,cval=255,cval_mask=0,mode=cv2.BORDER_CONSTANT),
                                        A.Resize(270, 270),
                                        A.RandomResizedCrop(width=256,height=256,scale=(0.85,1))
                                        ], additional_targets={'mask':'mask'})
        augment = A.Compose([
            A.ColorJitter(brightness = 0.01, contrast=0.05, saturation=0.05, hue=0.05),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = wing_dataset(dataPath=self.dataPath, _set='train', transform=transform, augment=augment)
        return torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

    
    def val_dataloader(self):
        transform = A.Compose([A.Rotate(0, border_mode=cv2.BORDER_CONSTANT, value=255, mask_value=0),
                                        A.Resize(256, 256)
                                        ], additional_targets={'mask':'mask'})
        augment = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_dataset = wing_dataset(dataPath=self.dataPath, _set='val', transform=transform, augment=augment)
        return torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=2)

class PredictAgeRegressionData(pl.LightningDataModule):
    def __init__(self, dataPath: str = "path/to/dir"):
        super().__init__()
        self.dataPath = dataPath

    def predict_dataloader(self):
        transform = A.Compose([A.Rotate(0, border_mode=cv2.BORDER_CONSTANT, value=255, mask_value=0),
                                        A.Resize(256, 256)
                                        ], additional_targets={'mask':'mask'})
        augment = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        prediction_dataset = wing_dataset(dataPath=self.dataPath, _set='to_predict', transform=transform, augment=augment, predict=True)
        return torch.utils.data.DataLoader(prediction_dataset, batch_size=8, shuffle=False, num_workers=2)

tb_logger = TensorBoardLogger(
    save_dir='logs',
    name='WingingItResNet',
)

seed_everything(42, True)
Path(f"{tb_logger.log_dir}/train").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/val").mkdir(exist_ok=True, parents=True)
if args.predict:
    Path(f"{args.dataPath}/predictions").mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    if args.checkpoint:
        experiment = WingAgePredictionExperiment.load_from_checkpoint(args.checkpoint)
    else:
        experiment = WingAgePredictionExperiment()
    trainer = Trainer(logger=tb_logger,
                callbacks=[
                    LearningRateMonitor(),
                    ModelCheckpoint(
                        save_top_k=2,
                        dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                        monitor="val_loss",
                        save_last=True,
                    ),
                ],
                max_epochs=100,
                log_every_n_steps=10)
    
    if args.train:
        data = AgeRegressionData(args.dataPath)
        trainer.fit(experiment, data)

    if args.predict:
        data = PredictAgeRegressionData(args.dataPath)
        trainer.predict(experiment, dataloaders=data)
        dataloader = data.predict_dataloader()
        os.makedirs(os.path.join(args.dataPath, "predictions", "gradcams"), exist_ok=True)

        for batch in dataloader:
            for i in range(len(batch['image'])):
                image_tensor = batch['image'][i]
                image_path = batch['fp'][i]
                fn = batch['fn'][i]

                orig_image = cv2.imread(image_path)
                if orig_image is None:
                    continue
                orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

                save_path = os.path.join(args.dataPath, "predictions", "gradcams", f"cam_{fn}")

                with torch.enable_grad():
                    generate_gradcam(experiment, image_tensor, orig_image, save_path)
