from torch import nn as nn
from torch.utils.data import DataLoader
import torch
import sys
import os
sys.path.append(os.path.abspath('nets/'))
import utils.Datasets as Datasets


class run(object):
    def __init__(self, root_path, num_classes, batch_size, model_type, model_path, pretrained, optim):
        self.epoch = 0
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 训练前所准备的数据集，自己写好的dataset，用dataloader加载
        self.model = self._generate_model(num_classes, model_type, model_path, pretrained)
        self.optimizer, self.scheduler = self._generate_optim(optim)
        self.dataloader = self._generate_dataloader(root_path)
        self.criteon = self._generate_criteon()
        self.bacth_results = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
        self.epoch_results = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    def _generate_criteon(self):
        criteon = nn.CrossEntropyLoss()
        if self.device.type == 'cuda':
            criteon = criteon.cuda()
        return criteon

    def _generate_optim(self, optim):
        if optim == 'SGD':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-2, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-2,
                                         betas=(0.9, 0.999), eps=1e-08)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35], gamma=0.1)
        return optimizer, scheduler

    def _generate_model(self, num_classes, model_type, model_path=None, pretrained=None):
        from nets.Resnet import ResNet, BasicBlock

        if model_type == 'resnet18':
            model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

        elif 'GCN' in model_type:
            from nets.GCN import GCN
            res_block = []
            for i in model_type.split('_')[1]:
                res_block.append(int(i))
            gcn_layers = int(model_type.split('_')[2])
            attention = True if model_type.split('_')[-1] == 'att' else False

            if 'patch' in model_type:
                model = GCN(self.batch_size, 2048, num_classes, res_block, gcn_layers, patch=True, attention=attention)
            else:
                model = GCN(self.batch_size, 512, num_classes, res_block, gcn_layers, attention=attention)

            if self.device.type == 'cuda':
                model.edge_index = model.edge_index.cuda()
                model.batch = model.batch.cuda()


        if self.device.type == 'cuda':
            model = model.cuda()
        return model

    def _generate_dataloader(self, root_path):
        import albumentations as A
        import albumentations.pytorch.transforms as T
        train_transform = A.Compose([
            # A.RandomCrop(width=600, height=731),  # Al随机裁剪
            A.RandomCrop(width=768, height=560),  # Fe随机裁剪
            A.HorizontalFlip(p=0.5),  # 水平翻转
            A.Rotate(limit=30),  # 随机旋转（正负30度）
            A.RandomBrightnessContrast(),  # 随机亮度和对比度调整
            # A.Resize(width=400, height=487),  # Al放缩
            A.Resize(width=320, height=175),  # Fe放缩
            A.ToFloat(max_value=255),
            T.ToTensorV2(),
        ])

        train_ann_path = 'utils/fe-few-shot-train.json'
        train_dataset = Datasets.my_dataset(root_path+'/train/', train_ann_path, transform=train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6, drop_last=True)

        test_transform = A.Compose([
            # A.Resize(width=400, height=487),
            A.Resize(width=320, height=175),
            A.ToFloat(max_value=255),
            T.ToTensorV2()])
        test_ann_path = root_path + '/test/test.json'
        test_dataset = Datasets.my_dataset(root_path+'/test/', test_ann_path, transform=test_transform)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6, drop_last=True)
        return {'train_loader': train_dataloader, 'test_loader': test_dataloader}

    def eval_model(self):
        self.model.eval()
        with torch.no_grad():
            it = 0
            for x, label in self.dataloader['test_loader']:
                it += 1
                if self.device.type == 'cuda':
                    x = x.cuda()
                    label = label.cuda()
                pred = self.model(x)
                loss = self.criteon(pred, label)
                logits_pred = torch.argmax(pred, dim=1)
                acc = (logits_pred == label).float().sum() / label.size(0)
                self.bacth_results['test_loss'].append(loss.item())
                self.bacth_results['test_acc'].append(acc.item())
        self.epoch_results['test_loss'].append(
            sum(self.bacth_results['test_loss'][self.epoch * it: (self.epoch+1) * it]) / it)
        self.epoch_results['test_acc'].append(
            sum(self.bacth_results['test_acc'][self.epoch * it: (self.epoch+1) * it]) / it)
        return

    def train_one_epoch(self):
        it = 0
        self.model.train()
        for x, label in self.dataloader['train_loader']:
            it += 1
            if self.device.type == 'cuda':
                x = x.cuda()
                label = label.cuda()  # 两种写法，一种是用.cuda()，另一种是to(device)，将数据转移到指定设备上计算
            pred = self.model(x)
            loss = self.criteon(pred, label)
            logits_pred = torch.argmax(pred, dim=1)
            acc = (logits_pred==label).float().sum() / label.size(0)
            self.bacth_results['train_loss'].append(loss.item())
            self.bacth_results['train_acc'].append(acc.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.scheduler.step()
        self.epoch_results['train_loss'].append(
            sum(self.bacth_results['train_loss'][self.epoch*it: (self.epoch+1)*it]) / it)
        self.epoch_results['train_acc'].append(
            sum(self.bacth_results['train_acc'][self.epoch*it: (self.epoch+1)*it]) / it)
        return

    def fine_tune(self, names):
        for key, value in self.models.named_parameters():
            if key not in names:
                value.requires_grad = False

    def model_init(self):
        from torchvision.ops.misc import ConvNormActivation
        from torchvision.models.mobilenetv3 import InvertedResidual
        def init(weight):
            for m in weight.children():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0.001)
                    nn.init.constant_(m.bias, 0)
                # 也可以判断是否为conv2d，使用相应的初始化方式
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Sequential) or isinstance(m, ConvNormActivation) or isinstance(m, InvertedResidual):
                    init(m)
        init(self.model)