"""
@File  : train.py
@Author: CodeCat
@Time  : 2021/7/9 10:56
"""
import os
import sys
import math
import argparse
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from utils.data_utils import get_dataset_dataloader
from utils.train_val_utils import evaluate
from models.googlenet import get_googlenet


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    # 获取数据集
    train_dataset, val_dataset, train_dataloader, val_dataloader = get_dataset_dataloader(args.data_path, args.batch_size)

    # 获取模型
    model = get_googlenet(args.flag, num_classes=args.num_classes)

    # 优化器
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5E-5)

    # cosine
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_acc = 0.0

    start = time.time()
    for epoch in range(args.epochs):
        # train
        model.train()

        loss_function = nn.CrossEntropyLoss()

        accu_loss = torch.zeros(1).to(device)
        accu_num = torch.zeros(1).to(device)
        optimizer.zero_grad()

        sample_num = 0
        dataloader = tqdm(train_dataloader)
        for step, data in enumerate(dataloader):
            images, labels = data
            sample_num += images.shape[0]

            if model.aux_logits:
                pred, pred_aux2, pred_aux1 = model(images.to(device))
                pred_classes = torch.max(pred+0.3*pred_aux2+0.3*pred_aux1, dim=1)[1]
                accu_num += torch.eq(pred_classes, labels.to(device)).sum()

                loss0 = loss_function(pred, labels.to(device))
                loss1 = loss_function(pred_aux2, labels.to(device))
                loss2 = loss_function(pred_aux1, labels.to(device))
                loss = loss0 + 0.3 * loss1 + 0.3 * loss2
            else:
                pred = model(images.to(device))
                pred_classes = torch.max(pred, dim=1)[1]
                accu_num += torch.eq(pred_classes, labels.to(device)).sum()
                loss = loss_function(pred, labels.to(device))

            loss.backward()
            accu_loss += loss.detach()

            dataloader.desc = "[train epoch {}] loss: {:3f}, acc: {:3f}".format(
                epoch, accu_loss.item() / (step + 1), accu_num.item() / (step + 1)
            )

            if not torch.isfinite(loss):
                print("WARNING: non-finite loss, ending training ", loss)
                sys.exit(1)

            optimizer.step()
            optimizer.zero_grad()

        train_loss, train_acc = accu_loss.item() / len(dataloader), accu_num.item() / sample_num


        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_dataloader,
            device=device,
            epoch=epoch
        )

        # tensorboard
        tags = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'learning_rate']
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]['lr'], epoch)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "./weights/googlenet.pth")
    end = time.time()
    print("Training 耗时为:{:.1f}".format(end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--data_path', type=str, default='/data/flower_photos')
    parser.add_argument('--flag', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0')

    opt = parser.parse_args()
    print(opt)
    main(opt)