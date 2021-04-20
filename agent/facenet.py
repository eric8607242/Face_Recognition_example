import os
import os.path as osp
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data.sampler import BalancedBatchSampler
from loss.utils import RandomNegativeTripletSelector
from loss.triplet import OnlineTripletLoss

from data.dataset.pair import PairFaceDataset
from model.facenet import FaceNet
from utils.metric import evaluate



__all__ = [ "FaceNetAgent" ]

class FaceNetAgent:
    """Train FaceNet model with Triplet Loss"""
    def __init__(self, config):
        # Torch environment
        # ======================================================
        self.config = config
        self.device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")

        # Dataset
        # ===================================================================
        train_transform = T.Compose([
                            T.Resize(config['dataset']['size']),
                            T.RandomHorizontalFlip(),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                            ])
        valid_transform = T.Compose([
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                            ])
        train_dataset = ImageFolder(root=config['dataset']['train']['root'],
                                    transform=train_transform)
        valid_dataset = PairFaceDataset(root=config['dataset']['valid']['root'],
                                    transform=valid_transform)
        # Triplet Sampler
        train_labels = [ s[1] for s in train_dataset.samples ]
        label_to_indices = torch.load(config['dataset']['train']['label_to_indices'])
        sampler = BalancedBatchSampler(train_labels,
                                    P=config['dataloader']['P'],
                                    K=config['dataloader']['K'],
                                    label_to_indices=label_to_indices)
        # DataLoader
        self.train_loader = DataLoader(dataset=train_dataset,
                                batch_sampler=sampler,
                                num_workers=config['dataloader']['num_workers'])
        self.valid_loader = DataLoader(dataset=valid_dataset,
                                batch_size=config['dataloader']['batch_size'],
                                num_workers=config['dataloader']['num_workers'],
                                shuffle=False)
        # FaceNet Model
        # ===================================================================
        model = FaceNet(in_channels=config['model']['in_channels'],
                        n_features=config['model']['n_features'])
        self.model = model.to(self.device)

        # Learning objective
        # ===================================================================
        margin = config['loss']['margin']
        selector = RandomNegativeTripletSelector(margin=margin)
        self.criterion = OnlineTripletLoss(margin, selector)

        # Optimizer & Scheduler
        # ===================================================================
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=config['optimizer']['lr'],
                                    weight_decay=config['optimizer']['weight_decay'])
        self.schedular = OneCycleLR(self.optimizer,
                                    max_lr=config['optimizer']['lr'],
                                    epochs=config['train']['n_epochs'],
                                    steps_per_epoch=len(self.train_loader))
        # Tensorboard Writer
        # ======================================================
        self.logdir = config['train']['logdir']
        self.board = SummaryWriter(logdir=self.logdir)

        # Current state
        self.current_acc = 0
        self.current_epoch = -1

    def resume(self):
        checkpoint_path = osp.join(self.logdir, 'best.pth')
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.schedular.load_state_dict(checkpoint['schedular'])
        self.current_acc = checkpoint['current_acc']
        self.current_epoch = checkpoint_path['current_epoch']
        print("Resume training at epoch '{}'".format(self.current_epoch))

    def train(self):
        for epoch in range(self.current_epoch+1, self.config['train']['n_epochs']):
            self.current_epoch = epoch
            self.train_one_epoch()
            self._validate()

    def train_one_epoch(self):
        current_epoch = self.current_epoch
        n_epochs = self.config['train']['n_epochs']
        loop = tqdm(self.train_loader,
                    leave=True,
                    desc=f"Train Epoch:{current_epoch}/{n_epochs}")
        triplet_losses = []
        self.model.train()
        for batch_idx, (imgs, labels) in enumerate(loop):
            # Move data
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            # Forward & Backward
            embeddings = self.model(imgs)
            self.optimizer.zero_grad()
            loss, count = self.criterion(embeddings, labels)
            loss.backward()
            self.optimizer.step()
            self.schedular.step()
            triplet_losses.append(loss.item())
            loop.set_postfix(
                    lr=self.optimizer.param_groups[0]['lr'],
                    loss=sum(triplet_losses)/len(triplet_losses),
                    triplets=count
                    )
        # Export training result
        epoch_loss = sum(triplet_losses)/len(triplet_losses)
        self.board.add_scalar("Train Loss", epoch_loss, self.current_epoch)

    def _validate(self):
        all_labels = []
        all_embeds1, all_embeds2 = [], []
        self.model.eval()
        for idx, ((imgs1, imgs2), labels) in enumerate(self.valid_loader):
            # Move data sample
            batch_size = labels.size(0)
            imgs1 = imgs1.to(self.device)
            imgs2 = imgs2.to(self.device)
            labels = labels.to(self.device)
            # Extract embeddings
            embeds1 = self.model(imgs1)
            embeds2 = self.model(imgs2)
            # Accumulates
            all_labels.append(labels.detach().cpu().numpy())
            all_embeds1.append(embeds1.detach().cpu().numpy())
            all_embeds2.append(embeds2.detach().cpu().numpy())
        # Evaluate
        labels = np.concatenate(all_labels)
        embeds1 = np.concatenate(all_embeds1)
        embeds2 = np.concatenate(all_embeds2)
        TP_ratio, FP_ratio, accs, best_thresholds = evaluate(embeds1, embeds2, labels)
        # Save Checkpoint
        acc = accs.mean()
        thresh = best_thresholds.mean()
        print((f"Epoch {self.current_epoch}/{self.config['train']['n_epochs']},"
            f" Valid Acc: {acc:.2f},"
            f" Valid Thresh: {thresh:.2f}"))
        if acc > self.current_acc:
            self.current_acc = acc
            self._save_checkpoint()

    def finalize(self):
        pass

    def _save_checkpoint(self):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'schedular': self.schedular.state_dict(),
            'current_acc': self.current_acc,
            'current_epoch': self.current_epoch,
            }
        checkpoint_path = osp.join(self.logdir, 'best.pth')
        torch.save(checkpoint, checkpoint_path)
        print("Save checkpoint to '{}'".format(checkpoint_path))
