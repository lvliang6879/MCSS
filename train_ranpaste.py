from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from model.models import CPS_Network
from utils.utils import *
from utils.att import *
import argparse
import numpy as np
import os
import torch
from model.models.SEUNet import *
import utils.mask_gen as mask_gen

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

from torch.nn import CrossEntropyLoss, DataParallel
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.multiprocessing
from torch.cuda.amp import autocast
from torch.cuda.amp import grad_scaler
from copy import deepcopy

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["CUDA_VISIBLE_DEVICES"] = '4, 5'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


seed = 1234
set_random_seed(seed)

DATASET = 'MER'  # ['DFC22', 'iSAID', 'MER', 'MSL', 'Vaihingen', 'GID-15']
SPLIT = '1-8'  # ['1-4', '1-8', '100', '300']
DFC22_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/DFC22/'
iSAID_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/iSAID/'
MER_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/MER/'
MSL_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/MSL/'
Vaihingen_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/Vaihingen/WCSL_crop/Vaihingen/'
# Vaihingen_DATASET_PATH ='/data1/users/lvliang/project_123/WSCL-main/WSCL-main/dataset/splits/Vaihingen/1-8/save/SEResUNet/20230828_170334_seed1234/'

GID15_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/GID-15/'

LAMBDA = 0.5
TRAIN_MODE = 'RanPaste'  # ['sup_only', 'WSCL', 'fixmatch', 'fixmatch_SGM', 'dualmatch', 'multimatch', 'multimatch_SGM', 'dualmatch_SGM', 'train_SGM']
# AUG_PROCESS_MODE = 'SACM'  # ['SC', 'DS', 'DC', 'SDC', 'SACM']
AUG_PROCESS_MODE= ''
THRESHOLD_MODE = ''  # ['adaptive', 'single', 'adaptive_decay']
# THRESHOLD_MODE = ''
# SINGLE_THRESHOLD = 1.0
SINGLE_THRESHOLD = ''
FINAL_THSHOLD = ''
# SGM ='SGM'
SGM = ''
# CLASS_MIXNUM = 2
CLASS_MIXNUM = ''
# SGM_LAMDA = 0.7
SGM_LAMDA = ''

PERCENT = 20
NUM_CLASSES = {'DFC22': 12, 'iSAID': 15, 'MER': 9, 'MSL': 9, 'Vaihingen': 5, 'GID-15': 15}


def parse_args():
    parser = argparse.ArgumentParser(description='WSCL Framework')

    # basic settings
    parser.add_argument('--data-root', type=str, default=MER_DATASET_PATH)
    parser.add_argument('--dataset', type=str, choices=['GID-15', 'iSAID', 'DFC22', 'MER', 'MSL', 'Vaihingen'],
                        default=DATASET)
    parser.add_argument('--lamb', type=int, default=LAMBDA,
                        help='the trade-off weight to balance the supervised loss and the unsupervised loss')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--threshold_mode', type=str, choices=['adaptive', 'single', 'adaptive_decay'],
                        default=THRESHOLD_MODE)
    parser.add_argument('--single_threshold', default=SINGLE_THRESHOLD)
    parser.add_argument('--final_threshold', default=FINAL_THSHOLD)
    parser.add_argument('---SGM_lamda', default=SGM_LAMDA)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet50')
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2', 'ResUNet'],
                        default='ResUNet')
    parser.add_argument('--train_mode', type=str,
                        choices=['RanPaste', 'WSCL', 'fixmatch', 'fixmatch_SGM', 'dualmatch', 'multimatch',
                                 'multimatch_SGM', 'dualmatch_SGM'],
                        default=TRAIN_MODE)
    parser.add_argument('--use_SGM', default=SGM)
    parser.add_argument('--SACM', default=AUG_PROCESS_MODE)
    parser.add_argument('--classmix_num', default=CLASS_MIXNUM)
    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str,
                        default='./dataset/splits/' + DATASET + '/' + SPLIT + '/labeled.txt')
    parser.add_argument('--unlabeled-id-path', type=str,
                        default='./dataset/splits/' + DATASET + '/' + SPLIT + '/unlabeled.txt')
    # parser.add_argument('--save-path', type=str, default='./exp/' + DATASET + '/' + SPLIT + '_' + SGM+ '_'+ str(SGM_LAMDA) + str(PERCENT) + '_' + str(LAMBDA) + '/' + AUG_PROCESS_MODE)
    parser.add_argument('--save-path', type=str, default='./exp/' + DATASET + '/' + SPLIT + '_' + str(
        LAMBDA) + '/' + TRAIN_MODE + '/' + THRESHOLD_MODE + '_' +
                                                         str(SINGLE_THRESHOLD) + '_to_' + str(
        FINAL_THSHOLD) + '/' + AUG_PROCESS_MODE + '_' + str(CLASS_MIXNUM) + '/' + SGM)
    parser.add_argument('--percent', type=float, default=PERCENT, help='0~100, the low-entropy percent r')
    args = parser.parse_args()
    return args


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main(args):
    create_path(args.save_path)

    criterion = CrossEntropyLoss(ignore_index=255)

    valset = SemiDataset(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(valset, batch_size=8, shuffle=False, pin_memory=False, num_workers=args.num_workers,
                           drop_last=False)

    temp_datset_sup = SemiDataset(args.dataset, args.data_root, 'train_l', args.crop_size, args.labeled_id_path)
    tmp_trainloader_l = DataLoader(temp_datset_sup, batch_size=int(args.batch_size / 2), shuffle=True,
                                   pin_memory=False, num_workers=args.num_workers, drop_last=True)
    l_sup = len(tmp_trainloader_l)
    print("l_sup:", l_sup)
    del temp_datset_sup, tmp_trainloader_l

    trainset_u = SemiDataset(args.dataset, args.data_root, 'train_u', args.crop_size, args.unlabeled_id_path)
    trainset_l = SemiDataset(args.dataset, args.data_root, 'train_l', args.crop_size, args.labeled_id_path,
                             nsample=len(trainset_u.ids))

    trainloader_u = DataLoader(trainset_u, batch_size=int(args.batch_size / 2), shuffle=True,
                               pin_memory=False, num_workers=args.num_workers, drop_last=True)
    # trainloader_u_1 = DataLoader(trainset_u, batch_size=int(args.batch_size / 2), shuffle=True,
    #                            pin_memory=False, num_workers=args.num_workers, drop_last=True)
    trainloader_l = DataLoader(trainset_l, batch_size=int(args.batch_size / 2), shuffle=True,
                               pin_memory=False, num_workers=args.num_workers, drop_last=True)

    # net, optimizer_0, optimizer_1, lr_scheduler_0, lr_scheduler_1 = init_basic_elems(args, trainloader_u)

    net, net_ema, optimizer, _, lr_scheduler, _  = init_basic_elems(args, trainloader_u)
    d_net, d_net_ema, m_optimizer, _, lr_scheduler, _  = init_basic_elems(args, trainloader_u)
    # print('\nParams: %.1fM' % count_params(net))

    global_thresh = 1 / NUM_CLASSES[args.dataset]
    class_conf = torch.ones(NUM_CLASSES[args.dataset]) * global_thresh

    if args.train_mode == 'sup_only':
        train_sup(net, trainloader_l, trainloader_u, valloader, criterion, optimizer, lr_scheduler, args)
    elif args.train_mode == 'fixmatch':
        train_fixmatch(net, trainloader_l, trainloader_u, valloader, criterion, optimizer, lr_scheduler, args,
                       global_thresh, class_conf)
    elif args.train_mode == 'RanPaste':
        train_RanPaste(net, net_ema, trainloader_l, trainloader_u, valloader, criterion, optimizer, lr_scheduler, args,
                       global_thresh, class_conf)

    elif args.train_mode == 'icnet':
        train_icnet(net, net_ema, d_net, d_net_ema, trainloader_l,
        trainloader_u, valloader, criterion, optimizer, m_optimizer, lr_scheduler, args)

    elif args.train_mode == 'classhyper':
        train_classhyper(net, trainloader_l, trainloader_u, valloader,
                             criterion, optimizer_0, optimizer_1, lr_scheduler_0, lr_scheduler_1, args)

def create_model(ema=False):

    model = se_resnext50_32x4d(num_classes=NUM_CLASSES[args.dataset], pretrained=None)
    pretrained_dict = torch.load(
        '/data1/users/lvliang/project_123/ClassHyPer-master/ClassHyPer-master/examples/save/se_resnext50_32x4d-a260b3a4.pth')
    my_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in my_dict}
    my_dict.update(pretrained_dict)
    model.load_state_dict(my_dict)
    # model = LANet.LANet(in_channels=3, num_classes=classes)
    # model = deeplab.deeplabv3plus_resnet50(num_classes=classes, output_stride=8, pretrained_backbone=True)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def init_basic_elems(args, trainloader_u):
    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2,
                 'ResUNet': se_resnext50_32x4d}
    if args.model == 'ResUNet':
        # net = model_zoo[args.model](NUM_CLASSES[args.dataset], None)
        # pretrained_dict = torch.load(
        #     '/data1/users/lvliang/project_123/ClassHyPer-master/ClassHyPer-master/examples/save/se_resnext50_32x4d-a260b3a4.pth')
        # my_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in my_dict}
        # my_dict.update(pretrained_dict)
        # net.load_state_dict(my_dict)
        net = create_model()
        net_ema = deepcopy(net)
        for p in net_ema.parameters():
            p.requires_grad_(False)

    # optimizer = optim.AdamW([
    #     # {'params': model.backbone.parameters(), 'lr': args.lr},
    #     {'params': [param for name, param in net.named_parameters()
    #                 if 'backbone' not in name],
    #      'lr': args.lr * 1.0}],
    #     lr=args.lr,
    #     betas=(0.9, 0.999),
    #     eps=1e-08,
    #     weight_decay=2e-4,
    #     amsgrad=False
    # )
    optimizer_0 = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                            lr=args.lr,
                            betas=(0.9, 0.999),
                            eps=1e-08,
                            weight_decay=2e-4,
                            amsgrad=False
                            )
    optimizer_1 = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                              lr=args.lr,
                              betas=(0.9, 0.999),
                              eps=1e-08,
                              weight_decay=2e-4,
                              amsgrad=False
                              )

    lr_scheduler_0 = optim.lr_scheduler.OneCycleLR(optimizer_0, max_lr=0.0001 * 6,
                                                 steps_per_epoch=len(trainloader_u),
                                                 epochs=args.epochs,
                                                 div_factor=6)
    lr_scheduler_1 = optim.lr_scheduler.OneCycleLR(optimizer_0, max_lr=0.0001 * 6,
                                                   steps_per_epoch=len(trainloader_u),
                                                   epochs=args.epochs,
                                                   div_factor=6)

    net = DataParallel(net).cuda()
    net_ema = DataParallel(net_ema).cuda()
    # model = model.cuda()
    return net, net_ema, optimizer_0, optimizer_1, lr_scheduler_0, lr_scheduler_1


def train_classhyper(net, trainloader_l, trainloader_u_0, valloader,
                     criterion, optimizer_0, optimizer_1, lr_scheduler_0, lr_scheduler_1, args):


    scaler = grad_scaler.GradScaler()


    # train_dataloader = iter(trainloader_l)
    # unsupervised_dataloader_0 = iter(trainloader_u_0)
    # unsupervised_dataloader_1 = iter(trainloader_u_1)
    # tbar = tqdm(zip(trainloader_l, trainloader_u_0), total=len(trainloader_l))
    total_loss, total_loss_l, total_loss_u = 0.0, 0.0, 0.0
    previous_best = 0

    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.9f\t\t\t\t\t previous best = %.2f" %
              (epoch, optimizer_0.param_groups[0]["lr"], previous_best))

        total_loss, total_loss_l, total_loss_u = 0.0, 0.0, 0.0

        tbar = tqdm(zip(trainloader_l, trainloader_u_0), total=len(trainloader_u_0))
        # for i, ((train_minibatch), (unsup_minibatch_0), (unsup_minibatch_1)) in enumerate(tbar):
        for i, ((img, mask, _, _), (img_u_w, img2_u_w, cutmix_box, img_u_s1, img_u_s2, _, _, _)) in enumerate(tbar):
            # train_minibatch = train_dataloader.__next__()
            imgs = img.cuda()
            gts = mask.cuda()

            optimizer_0.zero_grad()
            optimizer_1.zero_grad()
            cps_loss = 0.0
            if epoch >= 30:
                unsup_imgs_0 = img_u_w.cuda()
                unsup_imgs_1 = img2_u_w.cuda()

                with torch.no_grad():
                    net.eval()
                    # Estimate the pseudo-label with branch#1 & supervise branch#2
                    logits_u0_tea_1 = net(unsup_imgs_0, 1)
                    prob_u0_tea_1 = torch.sigmoid(logits_u0_tea_1).detach()

                    logits_u1_tea_1 = net(unsup_imgs_1, 1)
                    prob_u1_tea_1 = torch.sigmoid(logits_u1_tea_1).detach()

                    # Estimate the pseudo-label with branch#2 & supervise branch#1
                    logits_u0_tea_2 = net(unsup_imgs_0, 2)
                    prob_u0_tea_2 = torch.sigmoid(logits_u0_tea_2).detach()

                    logits_u1_tea_2 = net(unsup_imgs_1, 2)
                    prob_u1_tea_2 = torch.sigmoid(logits_u1_tea_2).detach()
                    ps_u1_tea_label_1 = torch.argmax(prob_u1_tea_1, dim=1)

                batch_mix_masks = torch.zeros_like(ps_u1_tea_label_1)
                for img_i in range(unsup_imgs_0.shape[0]):
                    classes = torch.unique(ps_u1_tea_label_1[img_i], sorted=True)
                    nclasses = classes.shape[0]
                    if nclasses > 2:
                        classes = classes[torch.Tensor(
                            np.random.choice(nclasses, int((nclasses - nclasses % 2) / 2), replace=False)).long()]
                    elif nclasses == 2:
                        classes = classes[1].unsqueeze(0)
                    elif nclasses == 1:
                        continue
                    batch_mix_masks[img_i] = mask_gen.generate_class_mask(ps_u1_tea_label_1[img_i], classes)

                batch_mix_masks = batch_mix_masks.unsqueeze(1)
                unsup_imgs_mixed = unsup_imgs_0 * (1 - batch_mix_masks) + unsup_imgs_1 * batch_mix_masks

                # Mix teacher predictions using same mask
                # the mask pixels are either 1 or 0
                prob_cons_tea_1 = prob_u0_tea_1 * (1 - batch_mix_masks) + prob_u1_tea_1 * batch_mix_masks
                prob_cons_tea_2 = prob_u0_tea_2 * (1 - batch_mix_masks) + prob_u1_tea_2 * batch_mix_masks

                ps_label_1 = torch.argmax(prob_cons_tea_1, dim=1)
                ps_label_2 = torch.argmax(prob_cons_tea_2, dim=1)

            with autocast():
                net.train()
                if epoch >= 30:  # warmup
                    # Get student#1 prediction for mixed image
                    logits_cons_stu_1 = net(unsup_imgs_mixed, 1)

                    # Get student#2 prediction for mixed image
                    logits_cons_stu_2 = net(unsup_imgs_mixed, 2)

                    ps_label_1 = ps_label_1.long()
                    ps_label_2 = ps_label_2.long()

                    cps_loss = criterion(logits_cons_stu_1, ps_label_2) + criterion(logits_cons_stu_2, ps_label_1)

                # empirically set coefficient to 1.0
                cps_loss = cps_loss * 1.0

                # sup loss
                sup_logits_l = net(imgs, 1)
                sup_logits_r = net(imgs, 2)
                # sup_logits_l = net(imgs)

                gts = gts.long()

                loss_sup_l = criterion(sup_logits_l, gts)
                loss_sup_r = criterion(sup_logits_r, gts)

                loss = loss_sup_l + loss_sup_r + cps_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer_0)
                scaler.step(optimizer_1)
                scaler.update()
                torch.cuda.synchronize()

            lr_scheduler_0.step()
            lr_scheduler_1.step()
            total_loss += loss.item()
            total_loss_l += loss_sup_l.item() + loss_sup_r.item()
            if epoch >= 30:
                total_loss_u += cps_loss.item()
            tbar.set_description('Loss: %.3f, Loss_l: %.3f, Loss_u: %.3f' % (
                total_loss / (i + 1), total_loss_l / (i + 1), total_loss_u / (i + 1)))

        if (epoch + 1) % 1 == 0:
            metric = meanIOU(num_classes=NUM_CLASSES[args.dataset])
            metric_2 = meanIOU(num_classes=NUM_CLASSES[args.dataset])

            net.eval()
            # net_ema.eval()
            tbar = tqdm(valloader)

            with torch.no_grad():
                for img, mask, _ in tbar:
                    img = img.cuda()
                    pred = net(img, 1)
                    # pred = net(img)
                    pred_2 = net(img, 2)
                    pred = torch.argmax(pred, dim=1)
                    pred_2 = torch.argmax(pred_2, dim=1)
                    metric.add_batch(pred.cpu().numpy(), mask.numpy())
                    metric_2.add_batch(pred_2.cpu().numpy(), mask.numpy())
                    IOU, mIOU = metric.evaluate()
                    IOU_2, mIOU_2 = metric_2.evaluate()
                    tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

            mIOU *= 100.0
            IOU *= 100
            mIOU_2 *= 100.0
            IOU_2 *= 100
            print('IoU: {}  | MIoU: {}'.format(IOU, mIOU))
            print('IoU_2: {}  | MIoU_2: {}'.format(IOU_2, mIOU_2))
            if mIOU > previous_best:
                if previous_best != 0:
                    os.remove(
                        os.path.join(args.save_path, '%s_%s_%s_epoch_%d_%.2f.pth' %
                                     (args.model, args.backbone, args.train_mode, previous_epoch, previous_best)))
                previous_best = mIOU
                previous_epoch = epoch
                torch.save(net.module.state_dict(),
                           os.path.join(args.save_path, '%s_%s_%s_epoch_%d_%.2f.pth' %
                                        (args.model, args.backbone, args.train_mode, epoch, mIOU)))



def train_icnet(net, net_ema, d_net, d_net_ema, trainloader_l, trainloader_u,
             valloader, criterion, optimizer, m_optimizer, lr_scheduler, args):
    scaler = grad_scaler.GradScaler()

    iters = 0
    total_iters = len(trainloader_u) * args.epochs

    previous_best = 0.0
    previous_best_iou = 0
    previous_best_ema = 0.0
    previous_best_d = 0
    previous_best_ema_d = 0
    weight_u = args.lamb
    global_step = 0

    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.9f\t\t\t\t\t previous best = %.2f  %s" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best, previous_best_iou))

        net.train()
        d_net.train()

        total_loss, total_loss_l, total_loss_u = 0.0, 0.0, 0.0

        tbar = tqdm(zip(trainloader_l, trainloader_u), total=len(trainloader_u))
        for i, ((img, mask, _, _), (img_u_w, img2_u_w, cutmix_box, img_u_s1, img_u_s2, _, _, _)) in enumerate(tbar):
            img, mask = img.cuda(), mask.cuda()
            img_u_w, img2_u_w, cutmix_box, img_u_s1, img_u_s2 = img_u_w.cuda(), img2_u_w.cuda(), cutmix_box.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
            optimizer.zero_grad()
            num_lb, num_ulb = img.shape[0], img_u_w.shape[0]
            with autocast():
                masks_pred = net(img)
                bce_loss = criterion(masks_pred, mask)

                d_masks_pred = d_net(img)
                d_bce_loss = criterion(d_masks_pred, mask)
                sup_loss = bce_loss + d_bce_loss

                N = torch.zeros((img_u_s1.shape[2], img_u_s1.shape[3])).cuda()
                random_x1 = random.randint(0, int(img_u_s1.shape[2] / 2))
                random_y1 = random.randint(0, int(img_u_s1.shape[3] / 2))
                N[random_x1:(random_x1 + int(img_u_s1.shape[2] / 2)),
                random_y1:(random_y1 + int(img_u_s1.shape[2] / 2))] = 1.0

                if global_step % 2 == 0:
                    logit1 = net(img_u_s1)
                    # logit2 = d_net_ema(img_u_w)
                    with torch.no_grad():
                        # net_ema.eval()
                        preds_u_w = d_net_ema(torch.cat((img_u_w, img2_u_w)))
                        logit2, logit2_2 = preds_u_w.split([num_lb, num_lb])
                        soft_2 = logit2.softmax(dim=1)
                        pseudo_logit_2, pseudo_label_2 = soft_2.max(dim=1)
                        prob2_u_w = logit2_2.softmax(dim=1)
                        conf2_u_w, mask2_u_w = prob2_u_w.max(dim=1)
                        pseudo_label_2[cutmix_box == 1] = mask2_u_w[cutmix_box == 1]

                else:
                    logit1 = d_net(img_u_s1)
                    # logit2 = net_ema(ul_imgs2)
                    with torch.no_grad():
                        # net_ema.eval()
                        preds_u_w = net_ema(torch.cat((img_u_w, img2_u_w)))
                        logit2, logit2_2 = preds_u_w.split([num_lb, num_lb])
                        soft_2 = logit2.softmax(dim=1)
                        pseudo_logit_2, pseudo_label_2 = soft_2.max(dim=1)
                        prob2_u_w = logit2_2.softmax(dim=1)
                        conf2_u_w, mask2_u_w = prob2_u_w.max(dim=1)
                        pseudo_label_2[cutmix_box == 1] = mask2_u_w[cutmix_box == 1]

                logit_1 = logit1.detach()
                soft_1 = F.softmax(logit_1, dim=1)
                pseudo_logit_1, pseudo_label_1 = torch.max(soft_1, dim=1)

                logit_2 = logit2.detach()
                soft_2 = (F.softmax(logit_2, dim=1))  # + F.softmax(logit_1, dim=1)) / 2
                pseudo_logit_2, pseudo_label_2 = torch.max(soft_2, dim=1)

                del logit2,  # soft_2

                pseudo_loss = criterion(logit1, pseudo_label_2)

                loss = sup_loss + pseudo_loss  # + neg_pseudo_loss  # + unc_weight.mean()  # * 0.5

                # tbar.set_description('Loss_l: %.3f, Loss_u: %.3f' % (sup_loss.item(), pseudo_loss.item()))


                optimizer.zero_grad()
                m_optimizer.zero_grad()
                # loss.backward()

                # optimizer.step()
                # m_optimizer.step()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.step(m_optimizer)
                scaler.update()
                torch.cuda.synchronize()

                alpha = min(1 - 1 / (global_step + 1), 0.9)

                for ema_param, param in zip(net_ema.parameters(), net.parameters()):
                    ema_param.data = ema_param.data * alpha + (1 - alpha) * param.data

                for ema_param, param in zip(d_net_ema.parameters(), d_net.parameters()):
                    ema_param.data = ema_param.data * alpha + (1 - alpha) * param.data

                global_step += 1
                del logit1, pseudo_label_2, masks_pred  # , d_masks_
            lr_scheduler.step()
            total_loss += loss.item()
            total_loss_l += sup_loss.item()
            total_loss_u += pseudo_loss.item()

            tbar.set_description('Loss: %.3f, Loss_l: %.3f, Loss_u: %.3f' % (
                total_loss / (i + 1), total_loss_l / (i + 1), total_loss_u / (i + 1)))

        if (epoch + 1) % 1 == 0:
            metric = meanIOU(num_classes=NUM_CLASSES[args.dataset])
            metric_ema = meanIOU(num_classes=NUM_CLASSES[args.dataset])
            metric_d = meanIOU(num_classes=NUM_CLASSES[args.dataset])
            metric_ema_d = meanIOU(num_classes=NUM_CLASSES[args.dataset])

            net.eval()
            d_net.eval()
            tbar = tqdm(valloader)

            with torch.no_grad():
                for img, mask, _ in tbar:
                    img = img.cuda()
                    pred = net(img)
                    pred_ema = net_ema(img)
                    pred_d = d_net(img)
                    pred_ema_d = d_net_ema(img)
                    pred = torch.argmax(pred, dim=1)
                    pred_ema = torch.argmax(pred_ema, dim=1)
                    pred_d = torch.argmax(pred_d, dim=1)
                    pred_ema_d = torch.argmax(pred_ema_d, dim=1)
                    metric.add_batch(pred.cpu().numpy(), mask.numpy())
                    metric_ema.add_batch(pred_ema.cpu().numpy(), mask.numpy())
                    metric_d.add_batch(pred_d.cpu().numpy(), mask.numpy())
                    metric_ema_d.add_batch(pred_ema_d.cpu().numpy(), mask.numpy())
                    IOU, mIOU = metric.evaluate()
                    IOU_ema, mIOU_ema = metric_ema.evaluate()
                    IOU_d, mIOU_d = metric_d.evaluate()
                    IOU_ema_d, mIOU_ema_d = metric_ema_d.evaluate()

                    tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))
                    tbar.set_description('mIOU_ema: %.2f' % (mIOU_ema * 100.0))

            mIOU *= 100.0
            IOU *= 100
            mIOU_ema *= 100.0
            IOU_ema *= 100
            mIOU_d *= 100.0
            IOU_d *= 100
            mIOU_ema_d *= 100.0
            IOU_ema_d *= 100
            print('IoU: {}  | MIoU: {}'.format(IOU, mIOU))
            print('IOU_ema: {}  | MIOU_ema: {}'.format(IOU_ema, mIOU_ema))
            print('IoU_d: {}  | MIoU_d: {}'.format(IOU_d, mIOU_d))
            print('IOU_ema_d: {}  | MIOU_ema_d: {}'.format(IOU_ema_d, mIOU_ema_d))
            if mIOU > previous_best:
                if previous_best != 0:
                    os.remove(
                        # os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
                        os.path.join(args.save_path, '%s_%s_%s_epoch_%d_%.2f.pth' %
                                     (args.model, args.backbone, args.train_mode, previous_epoch, previous_best)))
                previous_best = mIOU
                previous_best_iou = IOU
                previous_epoch = epoch
                # torch.save(model.module.state_dict(),
                #            os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))
                torch.save(net.module.state_dict(),
                           os.path.join(args.save_path, '%s_%s_%s_epoch_%d_%.2f.pth' %
                                        (args.model, args.backbone, args.train_mode, epoch, mIOU)))

            if mIOU_ema > previous_best_ema:
                if previous_best_ema != 0:
                    os.remove(
                        # os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
                        os.path.join(args.save_path, '%s_%s_%s_ema_epoch_%d_%.2f.pth' %
                                     (args.model, args.backbone, args.train_mode, previous_ema_epoch, previous_best_ema)))
                previous_best_ema = mIOU_ema
                previous_ema_epoch = epoch
                # torch.save(model.module.state_dict(),
                #            os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))
                torch.save(net.module.state_dict(),
                           os.path.join(args.save_path, '%s_%s_%s_ema_epoch_%d_%.2f.pth' %
                                        (args.model, args.backbone, args.train_mode, epoch, mIOU_ema)))

            if mIOU_d > previous_best_d:
                if previous_best_d != 0:
                    os.remove(
                        # os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
                        os.path.join(args.save_path, '%s_%s_%s_d_epoch_%d_%.2f.pth' %
                                     (args.model, args.backbone, args.train_mode, previous_epoch_d, previous_best_d)))
                previous_best_d = mIOU_d
                previous_epoch_d = epoch
                # torch.save(model.module.state_dict(),
                #            os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))
                torch.save(net.module.state_dict(),
                           os.path.join(args.save_path, '%s_%s_%s_d_epoch_%d_%.2f.pth' %
                                        (args.model, args.backbone, args.train_mode, epoch, mIOU_d)))

            if mIOU_ema_d > previous_best_ema_d:
                if previous_best_ema_d != 0:
                    os.remove(
                        # os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
                        os.path.join(args.save_path, '%s_%s_%s_ema_d_epoch_%d_%.2f.pth' %
                                     (args.model, args.backbone, args.train_mode, previous_ema_d_epoch, previous_best_ema_d)))
                previous_best_ema_d = mIOU_ema_d
                previous_ema_d_epoch = epoch
                # torch.save(model.module.state_dict(),
                #            os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))
                torch.save(net.module.state_dict(),
                           os.path.join(args.save_path, '%s_%s_%s_ema_d_epoch_%d_%.2f.pth' %
                                        (args.model, args.backbone, args.train_mode, epoch, mIOU_ema_d)))



def train_RanPaste(net, net_ema, trainloader_l, trainloader_u, valloader, criterion, optimizer, lr_scheduler, args,
                   global_thresh, class_conf):
    scaler = grad_scaler.GradScaler()

    iters = 0
    total_iters = len(trainloader_u) * args.epochs

    previous_best = 0.0
    previous_best_iou = 0
    previous_best_ema = 0.0
    weight_u = args.lamb
    global_step = 0

    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.9f\t\t\t\t\t previous best = %.2f  %s" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best, previous_best_iou))

        total_loss, total_loss_l, total_loss_u = 0.0, 0.0, 0.0

        tbar = tqdm(zip(trainloader_l, trainloader_u), total=len(trainloader_u))
        for i, ((img, mask, _, _), (img_u_w, img2_u_w, cutmix_box, img_u_s1, img_u_s2, _, _, _)) in enumerate(tbar):
            img, mask = img.cuda(), mask.cuda()
            img_u_w, img2_u_w, cutmix_box, img_u_s1, img_u_s2 = img_u_w.cuda(), img2_u_w.cuda(), cutmix_box.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
            optimizer.zero_grad()
            num_lb, num_ulb = img.shape[0], img_u_w.shape[0]
            with autocast():

                net.train()
                # net_ema.train()
                # net_ema.eval()

                masks_pred = net(img)
                bce_loss = criterion(masks_pred, mask)

                pred_soft = F.softmax(masks_pred, dim=1)
                prob_out2 = torch.max(pred_soft, dim=1)
                thred_out = torch.lt(prob_out2[0], 0.9)  # < 0.9 comput cross loss
                bce_loss = bce_loss * thred_out.float()
                sup_loss = bce_loss.mean()
                # sup_loss = bce_loss

                #  unlabel_image
                random_x = []
                random_y = []
                rans_x = int(img_u_s1.shape[2] / 2)
                rans_y = int(img_u_s1.shape[2] / 2)
                cut_num = img_u_s1.shape[0]
                for cn in range(cut_num):
                    random_x1 = random.randint(0, rans_x)
                    random_x.append(random_x1)
                    random_y1 = random.randint(0, rans_y)
                    random_y.append(random_y1)

                for cn in range(cut_num):
                    img_u_s1[cn, :, random_x[cn]:(random_x[cn] + rans_x),
                    random_y[cn]:(random_y[cn] + rans_y)] = img[0, :, random_x[cn]:(random_x[cn] + rans_x),
                                                            random_y[cn]:(random_y[cn] + rans_y)]

                logit1 = net(img_u_s1)
                #去掉cutmix再试试
                with torch.no_grad():
                    # net_ema.eval()
                    preds_u_w = net_ema(torch.cat((img_u_w, img2_u_w)))
                    logit2, logit2_2 = preds_u_w.split([num_lb, num_lb])
                    soft_2 = logit2.softmax(dim=1)
                    pseudo_logit_2, pseudo_label_2 = soft_2.max(dim=1)
                    prob2_u_w = logit2_2.softmax(dim=1)
                    conf2_u_w, mask2_u_w = prob2_u_w.max(dim=1)
                    pseudo_label_2[cutmix_box == 1] = mask2_u_w[cutmix_box == 1]
                # net_ema.train()

                pseudo_w1 = torch.gt(pseudo_logit_2, 0.6).float()  # > mean_predict, comput loss

                for cno in range(cut_num):
                    pseudo_label_2[cno, random_x[cno]:(random_x[cno] + rans_x),
                    random_y[cno]:(random_y[cno] + rans_y)] = mask[0, random_x[cno]:(random_x[cno] + rans_x),
                                                              random_y[cno]:(random_y[cno] + rans_y)]

                pseudo_loss = criterion(logit1, pseudo_label_2)

                final_weight1 = pseudo_w1.float()
                pseudo_loss = pseudo_loss * final_weight1
                pseudo_loss = pseudo_loss.sum() / (final_weight1.sum() + 1e-6)
                # pseudo_loss = pseudo_loss.mean()

                # k_pse = sigmoid_rampup(epoch, 50)
                loss = sup_loss + pseudo_loss  # * 0.5


                optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()
                alpha = min(1 - 1 / (global_step + 1), 0.999)
                for ema_param, param in zip(net_ema.parameters(), net.parameters()):
                    ema_param.data = ema_param.data * alpha + (1 - alpha) * param.data
                net.zero_grad()
                global_step += 1
                del logit1, pseudo_label_2, masks_pred



            total_loss += loss.item()
            total_loss_l += sup_loss.item()
            total_loss_u += pseudo_loss.item()

            # iters += 1
            # lr = args.lr * (1 - iters / total_iters) ** 0.9
            # optimizer.param_groups[0]["lr"] = lr

            lr_scheduler.step()

            tbar.set_description('Loss: %.3f, Loss_l: %.3f, Loss_u: %.3f' % (
                total_loss / (i + 1), total_loss_l / (i + 1), total_loss_u / (i + 1)))

        if (epoch + 1) % 1 == 0:
            metric = meanIOU(num_classes=NUM_CLASSES[args.dataset])
            metric_ema = meanIOU(num_classes=NUM_CLASSES[args.dataset])

            net.eval()
            # net_ema.eval()
            tbar = tqdm(valloader)

            with torch.no_grad():
                for img, mask, _ in tbar:
                    img = img.cuda()
                    pred = net(img)
                    pred_ema = net_ema(img)
                    pred = torch.argmax(pred, dim=1)
                    pred_ema = torch.argmax(pred_ema, dim=1)
                    metric.add_batch(pred.cpu().numpy(), mask.numpy())
                    metric_ema.add_batch(pred_ema.cpu().numpy(), mask.numpy())
                    IOU, mIOU = metric.evaluate()
                    IOU_ema, mIOU_ema = metric_ema.evaluate()

                    tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))
                    tbar.set_description('mIOU_ema: %.2f' % (mIOU_ema * 100.0))



            mIOU *= 100.0
            IOU *= 100
            mIOU_ema *= 100.0
            IOU_ema *= 100
            print('IoU: {}  | MIoU: {}'.format(IOU, mIOU))
            print('IOU_ema: {}  | MIOU_ema: {}'.format(IOU_ema, mIOU_ema))
            if mIOU > previous_best:
                if previous_best != 0:
                    os.remove(
                        # os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
                        os.path.join(args.save_path, '%s_%s_%s_epoch_%d_%.2f.pth' %
                                     (args.model, args.backbone, args.train_mode, previous_epoch, previous_best)))
                previous_best = mIOU
                previous_best_iou = IOU
                previous_epoch = epoch
                # torch.save(model.module.state_dict(),
                #            os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))
                torch.save(net.module.state_dict(),
                           os.path.join(args.save_path, '%s_%s_%s_epoch_%d_%.2f.pth' %
                                        (args.model, args.backbone, args.train_mode, epoch, mIOU)))

            if mIOU_ema > previous_best_ema:
                if previous_best_ema != 0:
                    os.remove(
                        # os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
                        os.path.join(args.save_path, '%s_%s_%s_ema_epoch_%d_%.2f.pth' %
                                     (args.model, args.backbone, args.train_mode, previous_ema_epoch, previous_best_ema)))
                previous_best_ema = mIOU_ema
                previous_ema_epoch = epoch
                # torch.save(model.module.state_dict(),
                #            os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))
                torch.save(net.module.state_dict(),
                           os.path.join(args.save_path, '%s_%s_%s_ema_epoch_%d_%.2f.pth' %
                                        (args.model, args.backbone, args.train_mode, epoch, mIOU_ema)))

        net.train()
        net_ema.train()


def train_fixmatch(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, lr_scheduler, args,
                   global_thresh, class_conf):
    scaler = grad_scaler.GradScaler()

    iters = 0
    total_iters = len(trainloader_u) * args.epochs

    previous_best = 0.0
    previous_best_iou = 0.0
    weight_u = args.lamb

    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.9f\t\t\t\t\t previous best = %.2f  %s" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best, previous_best_iou))

        total_loss, total_loss_l, total_loss_u = 0.0, 0.0, 0.0

        tbar = tqdm(zip(trainloader_l, trainloader_u), total=len(trainloader_u))
        for i, ((img, mask, _, _), (img_u_w, img2_u_w, cutmix_box, img_u_s1, img_u_s2, _, _, _)) in enumerate(tbar):
            img, mask = img.cuda(), mask.cuda()
            img_u_w, img2_u_w, cutmix_box, img_u_s1, img_u_s2 = img_u_w.cuda(), img2_u_w.cuda(), cutmix_box.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
            optimizer.zero_grad()
            num_lb, num_ulb = img.shape[0], img_u_w.shape[0]
            with autocast():
                with torch.no_grad():
                    model.eval()
                    preds_u_w = model(torch.cat((img_u_w, img2_u_w)))
                    pred_u_w, pred2_u_w = preds_u_w.split([num_lb, num_lb])
                    prob_u_w = pred_u_w.softmax(dim=1)
                    conf_u_w, mask_u_w = prob_u_w.max(dim=1)
                    prob2_u_w = pred2_u_w.softmax(dim=1)
                    conf2_u_w, mask2_u_w = prob2_u_w.max(dim=1)
                    mask_u_w[cutmix_box == 1] = mask2_u_w[cutmix_box == 1]

                model.train()
                preds = model(torch.cat((img, img_u_s1)))
                pred, pred_u_s = preds.split([num_lb, num_ulb])

                pred = model(img)

                loss_l = criterion(pred, mask)
                loss_u = criterion(pred_u_s, mask_u_w)

                loss_u = loss_u * (conf_u_w >= args.single_threshold)
                loss_u = torch.mean(loss_u)

                loss = (loss_l + weight_u * loss_u) / 2

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()

            total_loss += loss.item()
            total_loss_l += loss_l.item()
            total_loss_u += loss_u.item()

            # iters += 1
            # lr = args.lr * (1 - iters / total_iters) ** 0.9
            # optimizer.param_groups[0]["lr"] = lr

            lr_scheduler.step()

            tbar.set_description('Loss: %.3f, Loss_l: %.3f, Loss_u: %.3f' % (
                total_loss / (i + 1), total_loss_l / (i + 1), total_loss_u / (i + 1)))

        if (epoch + 1) % 1 == 0:
            metric = meanIOU(num_classes=NUM_CLASSES[args.dataset])

            model.eval()
            tbar = tqdm(valloader)

            with torch.no_grad():
                for img, mask, _ in tbar:
                    img = img.cuda()
                    pred = model(img)
                    pred = torch.argmax(pred, dim=1)

                    metric.add_batch(pred.cpu().numpy(), mask.numpy())
                    IOU, mIOU = metric.evaluate()

                    tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

            mIOU *= 100.0
            IOU *= 100
            print('IoU: {}  | MIoU: {}'.format(IOU, mIOU))
            if mIOU > previous_best:
                if previous_best != 0:
                    os.remove(
                        # os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
                        os.path.join(args.save_path, '%s_%s_%s_%s_%.2f_weight_%.2f_epoch_%d_%.2f.pth' %
                                     (args.model, args.backbone, args.train_mode, args.threshold_mode,
                                      args.single_threshold, weight_u, previous_epoch, previous_best)))
                previous_best = mIOU
                previous_best_iou = IOU
                previous_epoch = epoch
                # torch.save(model.module.state_dict(),
                #            os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))
                torch.save(model.module.state_dict(),
                           os.path.join(args.save_path, '%s_%s_%s_%s_%.2f_weight_%.2f_epoch_%d_%.2f.pth' %
                                        (args.model, args.backbone, args.train_mode, args.threshold_mode,
                                         args.single_threshold, weight_u, epoch, mIOU)))


def train_sup(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, lr_scheduler, args):
    scaler = grad_scaler.GradScaler()

    iters = 0
    total_iters = len(trainloader_u) * args.epochs

    previous_best = 0.0
    previous_best_iou = 0.0
    weight_u = args.lamb

    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.9f\t\t\t\t\t previous best = %.2f  %s" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best, previous_best_iou))

        total_loss, total_loss_l, total_loss_u = 0.0, 0.0, 0.0

        tbar = tqdm(zip(trainloader_l, trainloader_u), total=len(trainloader_u))
        # for i, ((img, mask,_,_), (img_u_w, img_u_s1, img_u_s2,_,_,_)) in enumerate(tbar):
        for i, ((img, mask, _, _), (img_u_w, _, _, img_u_s1, img_u_s2, _, _, _)) in enumerate(tbar):
            img, mask = img.cuda(), mask.cuda()
            img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
            optimizer.zero_grad()
            with autocast():
                model.train()

                num_lb, num_ulb = img.shape[0], img_u_w.shape[0]
                pred = model(img)
                loss_l = criterion(pred, mask)
                loss = loss_l
                # print("iters {} loss_l {}:".format(i+1, loss_l))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # torch.cuda.synchronize()

            total_loss += loss.item()
            total_loss_l += loss_l.item()
            iters += 1

            # iters += 1
            # lr = args.lr * (1 - iters / total_iters) ** 0.9
            # optimizer.param_groups[0]["lr"] = lr

            lr_scheduler.step()
            tbar.set_description('Loss: %.3f, Loss_l: %.3f, Loss_u: %.3f' % (
            total_loss / (i + 1), total_loss_l / (i + 1), total_loss_u / (i + 1)))

        if (epoch + 1) % 1 == 0:
            metric = meanIOU(num_classes=NUM_CLASSES[args.dataset])

            model.eval()
            tbar = tqdm(valloader)

            with torch.no_grad():
                for img, mask, _ in tbar:
                    img = img.cuda()
                    pred = model(img)
                    pred = torch.argmax(pred, dim=1)

                    metric.add_batch(pred.cpu().numpy(), mask.numpy())
                    IOU, mIOU = metric.evaluate()

                    tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

            mIOU *= 100.0
            IOU *= 100
            print('IoU: {}  | MIoU: {}'.format(IOU, mIOU))
            if mIOU > previous_best:
                if previous_best != 0:
                    os.remove(
                        os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
                previous_best = mIOU
                previous_best_iou = IOU
                # torch.save(model.module.state_dict(),
                torch.save(model.state_dict(),
                           os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))


if __name__ == '__main__':
    args = parse_args()
    if args.epochs is None:
        args.epochs = {'GID-15': 100, 'iSAID': 100, 'MER': 100, 'MSL': 100, 'Vaihingen': 100, 'DFC22': 100}[
            args.dataset]
    if args.lr is None:
        args.lr = {'GID-15': 0.001, 'iSAID': 0.001, 'MER': 0.001, 'MSL': 0.001,
                   'Vaihingen': 0.001, 'DFC22': 0.001}[args.dataset] / 16 * args.batch_size
    if args.crop_size is None:
        args.crop_size = {'GID-15': 320, 'iSAID': 320, 'MER': 320, 'MSL': 320, 'Vaihingen': 320, 'DFC22': 320}[
            args.dataset]
    if args.data_root is None:
        args.data_root = {'GID-15': GID15_DATASET_PATH,
                          'iSAID': iSAID_DATASET_PATH,
                          'MER': MER_DATASET_PATH,
                          'MSL': MSL_DATASET_PATH,
                          'Vaihingen': Vaihingen_DATASET_PATH,
                          'DFC22': DFC22_DATASET_PATH}[args.dataset]

    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name}: {arg_value}")

    main(args)
