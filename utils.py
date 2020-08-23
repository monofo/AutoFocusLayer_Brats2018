
from glob  import glob
import os
import time
import warnings
from datetime import datetime

import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

warnings.filterwarnings("ignore")

class PytorchTrainer:
    
    def __init__(self, model, optimizer, criterion, device, config):
        self.config = config
        self.epoch = 0
        
        self.base_dir = './result/' + config.base_dir
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5

        self.model = model
        self.device = device

        param_optimizer = list(self.model.named_parameters())

        self.optimizer = optimizer
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.criterion = criterion
        self.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss = self.train_one_epoch(train_loader)

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss = self.validation(validation_loader)

            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        t = time.time()
        for step, data in enumerate(val_loader):
            images = data["images"]
            targets = data["labels"]
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f},' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                targets = targets.to(self.device).long()
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                outputs = self.model(images)
                outputs = outputs.permute(0,2,3,4,1).contiguous().view(-1, self.config.num_classes)

                start_index = []
                end_index = []
                for i in range(3):
                    start = int((self.config.crop_size[i] - self.config.center_size[i])/2)
                    start_index.append(start)
                    end_index.append(start + self.config.center_size[i])
                targets = targets[:, start_index[0]:end_index[0], start_index[1]: end_index[1], start_index[2]: end_index[2]]
                targets = targets.contiguous().view(-1).cuda() 

                loss = self.criterion(outputs, targets)
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        for step, data in enumerate(train_loader):
            images = data["images"]
            targets = data["labels"]
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f},' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            
            targets = targets.to(self.device).long()
            images = images.to(self.device).float()
            batch_size = images.shape[0]

            self.optimizer.zero_grad()
            outputs = self.model(images)
            outputs = outputs.permute(0,2,3,4,1).contiguous().view(-1, self.config.num_classes)


            start_index = []
            end_index = []
            for i in range(3):
                start = int((self.config.crop_size[i] - self.config.center_size[i])/2)
                start_index.append(start)
                end_index.append(start + self.config.center_size[i])
            targets = targets[:, start_index[0]:end_index[0], start_index[1]: end_index[1], start_index[2]: end_index[2]]

            targets = targets.contiguous().view(-1).cuda() 

            loss = self.criterion(outputs, targets)
            loss.backward()
            summary_loss.update(loss.detach().item(), batch_size)

            self.optimizer.step()

            if self.config.step_scheduler:
                self.scheduler.step()

        return summary_loss
    
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
