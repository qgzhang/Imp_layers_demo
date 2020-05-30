import argparse, os, sys
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import cv2
from torch import optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from models import *
from dataloader import *
import visdom
from utils.utils import *
from time import time
import pdb
import warnings
warnings.filterwarnings("ignore")


def init_config():
    parser = argparse.ArgumentParser(description='Graph Matching')
    parser.add_argument('--data_path', type=str, default='D:/datasets/CUB_200_2011/')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--save_path', type=str, default='D:/results/CUB5000/model.pt')
    parser.add_argument('--show_images', action='store_true', default=True)
    parser.add_argument('--grid_res', type=int, default=8)

    args = parser.parse_args()
    # args.data_path = 'D:/datasets/CUB_200_2011_BBox/'
    args.new_H = 96
    args.new_W = 96
    args.grid_res = 8
    args.max_parts = 8
    args.outliers = 2
    args.plot_per_img = 10
    args.pck_alpha = 0.05
    args.cuda = torch.cuda.is_available()
    args.cuda = True
    args.img_path = os.path.join(args.data_path, 'images/')
    args.pair_path = args.data_path
    args.part_path = os.path.join(args.data_path, 'parts/')

    # 0: SM
    # 1: SMAC - Rayleigh Quotient
    # 2: SMAC - bi-directional eigen decomposition
    # 3: SMAC - forward eigen decompostion and backward KKT
    args.SMAC = 1
    args.model = 'GMN'
    # args.model = 'ImpGMN'

    args.n_epoch = 8
    args.nbatch = 8
    args.lr = 1e-4

    args.clip = 1e1
    args.show_images = True
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    args.all_train_iter_loss = []
    args.all_train_iter_pck = []
    args.all_test_iter_loss = []
    args.all_test_iter_pck = []

    return args


def _apply_loss(d, d_gt):
    # x = d - d_gt
    # eps = 1e-8
    # loss = torch.sum(torch.sqrt(torch.diagonal(torch.bmm(x, x.permute(0,2,1)), dim1=-2, dim2=-1) + eps), dim=1)
    # return loss
    n = d_gt.shape[-2]
    return torch.sum(torch.norm(d[:, :n, :] - d_gt, dim=-1))


def test(args, i_epoch, model, data_loader):
    total_loss = 0
    total_pck = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (img1, img2, P1, gt_P1, img_path1, img_path2, P2) in enumerate(data_loader):
            # print('sum img2 = ', torch.sum(img2))
            # print(args.trans_un_norm)
            # print(args.trans_un_norm(img2.clone()).min(), args.trans_un_norm(img2.clone()).max())
            start = time()
            img1 = img1.to(args.device).to(torch.float)
            P1 = P1.to(args.device)
            img2 = img2.to(args.device).to(torch.float)
            P2 = P2.to(args.device)
            gt_P1 = gt_P1.to(args.device)
            pred_P1 = model(img1, P1, img2, P2)
            loss = _apply_loss(pred_P1, gt_P1)
            total_loss += loss.item()
            args.all_test_iter_loss.append(loss.item())
            pck = batch_calc_pck(img1, pred_P1, gt_P1, alpha=args.pck_alpha)
            total_pck += pck
            args.all_test_iter_pck.append(pck)
            if batch_idx % 1 == 0:
                print('Test epoch: %d, batch: %d(%d), loss = %.4f, pck = %.4f, time(%.4f)' %
                      (i_epoch, batch_idx, len(data_loader), loss.item(), pck, time() - start))
            if args.show_images and batch_idx % args.plot_per_img == 0:
                im1 = args.trans_un_norm(img1[0]).cpu().permute(1, 2, 0)
                im2 = args.trans_un_norm(img2[0]).cpu().permute(1, 2, 0)
                show_features(im1, P1[0,:,:].squeeze(), im2, P2[0,:,:], pred_P1[0,:,:].squeeze(), gt_P1[0,:,:].squeeze(), 1)
                # show_features(img1[0,:,:,:].permute(1,2,0), P1.squeeze(),
                #               img2[0,:,:,:].permute(1,2,0), pred_P1.squeeze(), gt_P1.squeeze(), 10)
    return total_loss / float(batch_idx+1), total_pck / float(batch_idx+1)


def train(args, i_epoch, model, data_loader, optimizer):
    total_loss = 0
    total_pck = 0
    model.train()
    # for batch_idx, (img1, img2, P1, gt_P1, img_path1, img_path2, P2, A1, A2) in enumerate(data_loader):
    for batch_idx, (img1, img2, P1, gt_P1, img_path1, img_path2, P2) in enumerate(data_loader):
        check_nan(img1)
        check_nan(img2)
        check_nan(P1)
        check_nan(gt_P1)
        check_nan(P2)

        start = time()
        # print(args.trans_un_norm)
        # print(args.trans_un_norm(img2.clone()).min(), args.trans_un_norm(img2.clone()).max())
        img1 = img1.to(args.device).to(torch.float)
        P1 = P1.to(args.device)
        img2 = img2.to(args.device).to(torch.float)
        P2 = P2.to(args.device)
        gt_P1 = gt_P1.to(args.device)
        # A1.to(args.device)
        # A2.to(args.device)
        optimizer.zero_grad()

        # with torch.autograd.detect_anomaly():
        pred_P1 = model(img1, P1, img2, P2)
        loss = _apply_loss(pred_P1, gt_P1)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        args.all_train_iter_loss.append(loss.item())
        pck = batch_calc_pck(img1, pred_P1, gt_P1, alpha=args.pck_alpha)
        total_pck += pck
        args.all_train_iter_pck.append(pck)
        if batch_idx % 1 == 0:
            print('Train epoch: %d, batch: %d(%d), loss = %.4f, pck = %.4f, time(%.4f)' %
                  (i_epoch, batch_idx, len(data_loader), loss.item(), pck, time()-start))
            # print('Train epoch: ', i_epoch, ', batch: ', batch_idx, "(", len(data_loader), "), loss = ", loss.item(),
            #       ', pck = ', pck, '， time: ', time()-start)
        if args.show_images and batch_idx % args.plot_per_img == 0:
            # im1 = cv2.imread(img_path1[0]) / 255.
            # im2 = cv2.imread(img_path2[0]) / 255.
            im1 = args.trans_un_norm(img1[0]).cpu().permute(1,2,0)
            im2 = args.trans_un_norm(img2[0]).cpu().permute(1,2,0)
            show_features(im1, P1[0,:,:].squeeze(), im2, P2[0,:,:], pred_P1[0,:,:].squeeze(), gt_P1[0,:,:].squeeze(), 4)
    return total_loss / float(batch_idx + 1), total_pck / float(batch_idx + 1)


if __name__ == '__main__':
    # vis = visdom.Visdom()
    args = init_config()
    # pdb.set_trace()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    args.trans_un_norm = transforms.Compose([
        UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset = CUB5000_perm(args, transform=transform)

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=args.nbatch, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.nbatch, shuffle=True)

    # model = GMN(args)
    model = ImpGMN(args)
    model = model.to(args.device)
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -args.clip, args.clip))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_pck = 0

    for i_epoch in range(1, args.n_epoch+1):
        # test_loss, test_pck = test(args, i_epoch, model, test_dataloader)
        train_loss, train_pck = train(args, i_epoch, model, train_dataloader, optimizer)
        print("############  Avg_train_loss = ", train_loss, "Avg_train_pck = ", train_pck, "############")
        test_loss , test_pck  = test (args, i_epoch, model, test_dataloader)
        print("############  Avg_test_loss = ", test_loss, "Avg_test_pck = ", test_pck, "############")
        if test_pck > best_pck:
            best_pck = test_pck
            torch.save(model.state_dict(), args.save_path)




















