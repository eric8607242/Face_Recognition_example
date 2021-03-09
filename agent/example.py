import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from tensorboardX import SummaryWriter

from data.dataset.pair import PairFaceDataset
from model import get_model
from loss import get_margin
from utils.metric import evaluate

__all__ = [ "ExampleAgent" ]


class ExampleAgent:

    def __init__(self, config):
        # Environment
        # ===================================================================
        self.config = config
        self.device = config['train']['device'] if torch.cuda.is_available() else "cpu"

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
        self.train_loader = DataLoader(dataset=train_dataset,
                                batch_size=config['dataloader']['batch_size'],
                                num_workers=config['dataloader']['num_workers'],
                                shuffle=True)
        self.valid_loader = DataLoader(dataset=valid_dataset,
                                batch_size=config['dataloader']['batch_size'],
                                num_workers=config['dataloader']['num_workers'],
                                shuffle=False)

        # Model
        # ===================================================================
        model = get_model(config['model']['model_name'])
        margin = get_margin(config['model']['margin_name'], config['model']['n_features'], config['model']['n_classes'], config['model']['margin'], config['model']['s'])

        """
        model = ExampleNet(n_features=config['model']['n_features'],
                            n_classes=config['model']['n_classes'])
        """
        self.model = model.to(self.device)
        self.margin = margin.to(self.device)

        # Optimizer
        # ===================================================================
        self.optimizer = optim.Adam([{'params':self.model.parameters(), 'weight_decay':config['optimizer']['weight_decay']},
                                     {'params':self.margin.identity_weights}],
                                    lr=config['optimizer']['lr'])

        # Scheduler
        # ===================================================================
        # self.scheduler = None

        # Loss Function
        # ===================================================================
        self.criterion = nn.CrossEntropyLoss()

        # Training State
        # ===================================================================
        self.current_epoch = -1
        self.current_acc = 0

        # Tensorboard
        # ===================================================================
        self.tensorboard = SummaryWriter(logdir=config['train']['logdir'])

    def train(self):
        for epoch in range(self.current_epoch+1, self.config['train']['n_epochs']):
            self.current_epoch = epoch
            self._train_one_epoch()
            self._validate()
            # self.scheduler.step()

    def finalize(self):
        pass

    def resume(self):
        checkpoint_path = osp.join(self.config['train']['logdir'], 'best.pth')
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.current_epoch = checkpoint['current_epoch']
        self.current_acc = checkpoint['current_acc']

    def _train_one_epoch(self):
        running_corrects = 0
        running_loss = 0
        self.model.train()
        for idx, (imgs, labels) in enumerate(self.train_loader):
            # Move data sample
            batch_size = imgs.size(0)
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            # Forward & Backward
            self.optimizer.zero_grad()
            #_, outputs = self.model(imgs)
            outputs = self.model(imgs)
            outputs = self.margin(outputs, labels)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            # Status
            preds = torch.max(outputs.data, 1)[1]
            corrects = float(torch.sum(preds==labels.data))
            running_corrects += corrects
            running_loss += loss.item()*batch_size
            # Logging
            if idx % self.config['train']['n_intervals'] == 0:
                print((f"Epoch {self.current_epoch}:{self.config['train']['n_epochs']}"
                    f"({int(idx/len(self.train_loader)*100)}%)"
                    f" Train Loss: {loss.item():.2f},"
                    f" Train Acc: {corrects/batch_size:.2f}"))
        # Logging
        epoch_acc = running_corrects / len(self.train_loader.dataset)
        epoch_loss = running_loss / len(self.train_loader.dataset)
        print((f"Epoch {self.current_epoch}:{self.config['train']['n_epochs']},"
            f" Train Loss: {epoch_loss:.2f},"
            f" Train Acc: {epoch_acc:.2f}"))
        self.tensorboard.add_scalar("Train Acc", epoch_acc, self.current_epoch)
        self.tensorboard.add_scalar("Train Loss", epoch_loss, self.current_epoch)

    def _validate(self):
        self.model.eval()
        all_labels = []
        all_embeds1, all_embeds2 = [], []
        for idx, ((imgs1, imgs2), labels) in enumerate(self.valid_loader):
            # Move data sample
            batch_size = labels.size(0)
            imgs1 = imgs1.to(self.device)
            imgs2 = imgs2.to(self.device)
            labels = labels.to(self.device)
            # Extract embeddings
            #embeds1, _ = self.model(imgs1)
            embeds1 = self.model(imgs1)
            embeds1 = F.normalize(embeds1, p=2)
            #embeds2, _ = self.model(imgs2)
            embeds2 = self.model(imgs2)
            embeds2 = F.normalize(embeds2, p=2)
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
        print((f"Epoch {self.current_epoch}:{self.config['train']['n_epochs']},"
            f" Valid Acc: {acc:.2f},"
            f" Valid Thresh: {thresh:.2f}"))
        if acc > self.current_acc:
            self.current_acc = acc
            self._save_checkpoint()

    def _save_checkpoint(self):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_epoch': self.current_epoch,
            'current_acc': self.current_acc,
            }
        checkpoint_path = osp.join(self.config['train']['logdir'], 'best.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Save checkpoint to '{checkpoint_path}'")
