import torch
import torch.nn as nn
import numpy as np
from CountNet import CountingModel
import cv2
from data.data_load import *
from utils import check_lr, Adder
from tensorboardX import SummaryWriter
from valid import valid


def train(net, args):
    kernel = get_kernel(args.kernel_size, args.alpha)
    constant = np.sum(kernel)
    iter_loss_adder = Adder()
    epoch_loss_adder = Adder()
    img_dir = r'G:\papers\TheCellCount(paper)\CellCounting\multiclass_dataset\MoNuSAC\MoNuSAC_train'
    transform = [PairCompose([
        PairToTensor(),
        PairSplice(512),
        transforms.RandomChoice([
            PairRandomHorizontalFilp(),
            PairColorJitter(0.3, 0.3, 0.3, 0.2),
            PairCutout(2)])
            ]), PairCutmix(2)]
    celldata = cell_dataset(img_dir, args.kernel_size, args.alpha, args.n_classes, transform)
    train_dataset, val_dataset = random_split(
        dataset=celldata,
        lengths=[0.7, 0.3],
        generator=torch.Generator().manual_seed(0)
    )
    train_data = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_data = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    max_iter = len(train_data)
    criteria = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)
    writer = SummaryWriter()
    best_mse = 100
    for epoch in range(args.num_epoch):
        for iter_idx, data in enumerate(train_data):
            image, label0, label1, label2, label3 = data
            image = image.cuda()
            label0 = label0.cuda()
            label1 = label1.cuda()
            label2 = label2.cuda()
            label3 = label3.cuda()
            optimizer.zero_grad()
            out = net(image)
            loss0 = criteria(out[0], label0)
            loss1 = criteria(out[1], label1)
            loss2 = criteria(out[2], label2)
            loss3 = criteria(out[3], label3)
            loss = loss0 + loss1 + loss2 + loss3
            
            writer.add_scalar('loss0', loss0, iter_idx+(epoch-1)*max_iter)
            writer.add_scalar('loss1', loss1, iter_idx+(epoch-1)*max_iter)
            writer.add_scalar('loss2', loss2, iter_idx+(epoch-1)*max_iter)
            writer.add_scalar('loss3', loss3, iter_idx+(epoch-1)*max_iter)
            writer.add_scalar('loss', loss, iter_idx+(epoch-1)*max_iter)

            loss.backward()
            optimizer.step()
            iter_loss_adder(loss.item())
            epoch_loss_adder(loss.item())

            if iter_idx % args.print_freq == 0:
                lr = check_lr(optimizer)
                print("Epoch: %03d Iter: %4d/%4d LR: %.10f Loss: %7.4f" % (
                    epoch, iter_idx + 1, max_iter, lr, iter_loss_adder.average()))
                iter_loss_adder.reset()
            overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
            torch.save({'model': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch}, overwrite_name)
            
        if epoch % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch)
            torch.save({'model': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch}, save_name)
        print("EPOCH: %02d\nEpoch Pixel Loss: %7.4f" % (
            epoch, epoch_loss_adder.average()))
        epoch_loss_adder.reset()
        scheduler.step()
        if epoch % args.valid_freq == 0:
            val_gopro = valid(net, val_data)
            writer.add_scalar('val_gopro', val_gopro, epoch)
            if val_gopro < best_mse:
                torch.save({'model': net.state_dict()},
                            os.path.join(args.model_save_dir, 'Best.pkl'))