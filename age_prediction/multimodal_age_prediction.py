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


parser = argparse.ArgumentParser(description="Segmentation model runner")
parser.add_argument("--train", action='store_true', help="Flag for Training model", default=False)
parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file to load into model", default=None)
parser.add_argument("--predict", action='store_true', help="Flag for just predicting from model (exclusive from train)")
parser.add_argument("--dataPath", type=str, help="Path to find data. If training should have train and val subfolders, if predicting should have subfolder called 'to_predict'")

args = parser.parse_args()

assert args.train == False or args.predict == False, "Cannot both train and predict. Pick one."



def generate_gradcam(model, image_tensor, orig_image_np, save_path):
    model.eval()

    target_layer = model.image_backbone.layer4[-1]
    device = next(model.parameters()).device
    model = model.to(device)

    class RegressionModelOutputTarget:
        def __init__(self, output_index):
            self.output_index = output_index

        def __call__(self, model_output):
            return model_output[:, self.output_index] if model_output.dim() > 1 else model_output[self.output_index]

    # Create dummy features (batch size = 1, feature dim = model.geometry_fc[0].in_features)
    feat_input_dim = model.features_fc[0].in_features
    dummy_features = torch.zeros((1, feat_input_dim), device=device)

    with torch.enable_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device).requires_grad_()
        dummy_features.requires_grad = True
        model_output = model(image_tensor, dummy_features)
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
    def __init__(self, dataPath='wings', _set='train', transform=None, augment=None, predict=False):
        self.set = _set
        self.dataPath = dataPath
        self.transform = transform # This is transofrmations like rotation, resizing that should be applied to both the mask and input image
        self.augment = augment # This is tranformations only applied to the input image ie, color jitter, contrast, brightness, etc.
        self.predict = predict
        self.to_tensor = A.Compose([ToTensorV2()])

        self.root = Path(f'{self.dataPath}')
        self.image_path = self.root / self.set
        self.ages_path = self.root / 'ages.csv'
        self.ages = pd.read_csv(self.ages_path, index_col=0)['Age']
        self.feat_path = self.root / 'features.csv'
        self.features = pd.read_csv(self.feat_path, index_col=0)
        self.img_list = self.get_filenames(self.image_path)

        
    def __len__(self):
        return(len(self.img_list))
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        fp = self.img_list[idx]
        fn = Path(fp).name
        
        # Only try to get age if not in prediction mode
        if not self.predict:
            age = int(self.ages[fn])
        else:
            age = 0  # Default value for prediction mode
        
        if self.transform:
            img = self.transform(image=img)['image']
                
        if self.augment:
            img = self.augment(image=img)['image']
        img = self.to_tensor(image=img)['image']
        geom = self.features.loc[fn].values.astype(np.float32)
        return {'image': img, 'target': age, 'fp': fp, 'fn': fn, 'features': torch.tensor(geom)}
    
    def get_filenames(self, path):
        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list

class WingAgePredictionExperiment(pl.LightningModule):
    def __init__(self, features_input_dim=13):
        super().__init__()

        self.image_backbone = models.resnet50(pretrained=True)
        image_feat_dim = self.image_backbone.fc.in_features
        self.image_backbone.fc = nn.Identity()

        self.features_fc = nn.Sequential(
            nn.Linear(features_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        self.final_fc = nn.Linear(image_feat_dim + 32, 1)

        params = smp.encoders.get_preprocessing_params("resnext50_32x4d")
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        self.loss_fn = nn.MSELoss()

    def forward(self, images, features):
        images = (images - self.mean) / self.std
        image_feat = self.image_backbone(images)
        features_feat = self.features_fc(features)
        combined = torch.cat((image_feat, features_feat), dim=1)
        return self.final_fc(combined).squeeze()
    
    def training_step(self, batch, batch_idx):
        images = batch['image'].float().to(self.device)
        features = batch['features'].float().to(self.device)
        targets = batch['target'].float().to(self.device)

        preds = self.forward(images, features)
        loss = self.loss_fn(preds, targets)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['image'].to(self.device).float()
        features = batch['features'].float().to(self.device)
        targets = batch['target'].to(self.device).float()

        pred = self.forward(images, features).squeeze()
        val_loss = self.loss_fn(pred, targets)

        self.log("val_loss", val_loss)

    def predict_step(self, batch, batch_idx):
        images = batch['image'].to(self.device).float()
        features = batch['features'].to(self.device).float()
        fps = batch['fp']
        fns = [x.split('/')[-1] for x in fps]

        preds = self.forward(images, features)
        predictions = []

        for i, fn in enumerate(fns):
            pred = preds[i]  # Tensor with a single value
            pred_value = pred.item()  # Convert to scalar
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

        for _set in ['val', 'train']:
            if _set == 'train':
                dataloader = self.trainer.datamodule.train_dataloader()
            else:
                dataloader = self.trainer.datamodule.val_dataloader()

            for batch in dataloader:
                input = batch['image'].to(self.device).float()
                features = batch['features'].to(self.device).float()
                label = batch['target']
                fn = batch['fn']

                prediction = self.forward(input, features).detach().cpu().squeeze().numpy()
                label = label.cpu().numpy()

                labels += label.tolist()
                preds += prediction.tolist() if isinstance(prediction.tolist(), list) else [prediction.tolist()]
                fns += fn
                _sets += [_set] * len(fn)

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
                                        A.Affine(rotate=5,translate_percent=(0.05,0.05),scale=(0.85,1.05),shear=1,cval=255,cval_mask=0,mode=cv2.BORDER_CONSTANT),
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

# For reproducibility
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
        # Access the prediction dataset again
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

                """
                with torch.enable_grad():
                    generate_gradcam(experiment, image_tensor, orig_image, save_path)
                """
