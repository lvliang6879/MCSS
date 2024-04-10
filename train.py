from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from utils.utils import *
from utils.att import *
import argparse
import numpy as np
import os
import torch
from model.models.SEUNet import *
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

from torch.nn import CrossEntropyLoss, DataParallel, KLDivLoss, MSELoss
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.multiprocessing
from torch.cuda.amp import autocast
from torch.cuda.amp import grad_scaler


torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 3'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


        
seed = 1234
set_random_seed(seed)

DATASET = 'DFC22'     # ['DFC22', 'iSAID', 'MER', 'MSL', 'Vaihingen', 'GID-15']
SPLIT = '1-8'     # ['1-4', '1-8', '100', '300']
DFC22_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/DFC22/'
iSAID_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/iSAID/'
MER_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/MER/'
MSL_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/MSL/'
Vaihingen_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/Vaihingen/WCSL_crop/Vaihingen/'
# Vaihingen_DATASET_PATH ='/data1/users/lvliang/project_123/WSCL-main/WSCL-main/dataset/splits/Vaihingen/1-8/save/SEResUNet/20230828_170334_seed1234/'

GID15_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/GID-15/'

LAMBDA = 0.5
TRAIN_MODE = 'fixmatch'  #['sup_only', 'WSCL', 'fixmatch', 'fixmatch_SGM', 'dualmatch', 'multimatch', 'multimatch_SGM', 'dualmatch_SGM', 'train_SGM']
# AUG_PROCESS_MODE = 'SACM'   # ['SC', 'DS', 'DC', 'SDC', 'SACM']
AUG_PROCESS_MODE= ''
THRESHOLD_MODE = 'single'    # ['adaptive', 'single', 'adaptive_decay']
# THRESHOLD_MODE = ''
# SINGLE_THRESHOLD = 1.0
SINGLE_THRESHOLD = 0.95
FINAL_THSHOLD = 0.8
# SGM ='SGM'
SGM=''
# CLASS_MIXNUM = 3
CLASS_MIXNUM = ''
SGM_LAMDA = 0.5
# SGM_LAMDA = ''

PERCENT = 20
NUM_CLASSES = {'DFC22': 12, 'iSAID': 15, 'MER': 9, 'MSL': 9, 'Vaihingen': 5, 'GID-15': 15}

def parse_args():
    parser = argparse.ArgumentParser(description='WSCL Framework')

    # basic settings
    parser.add_argument('--data-root', type=str, default=DFC22_DATASET_PATH)
    parser.add_argument('--dataset', type=str, choices=['GID-15', 'iSAID', 'DFC22', 'MER', 'MSL', 'Vaihingen'], default=DATASET)
    parser.add_argument('--lamb', type=int, default=LAMBDA, help='the trade-off weight to balance the supervised loss and the unsupervised loss')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--threshold_mode', type=str, choices=[ 'adaptive', 'single', 'adaptive_decay'], default=THRESHOLD_MODE)
    parser.add_argument('--single_threshold', default=SINGLE_THRESHOLD)
    parser.add_argument('--final_threshold', default=FINAL_THSHOLD)
    parser.add_argument('---SGM_lamda', default=SGM_LAMDA)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet50')
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2', 'ResUNet'],
                        default='ResUNet')
    parser.add_argument('--train_mode', type=str, choices=['sup_only', 'WSCL', 'fixmatch', 'fixmatch_SGM', 'dualmatch', 'multimatch', 'multimatch_SGM', 'dualmatch_SGM'],
                        default=TRAIN_MODE)
    parser.add_argument('--use_SGM', default=SGM)
    parser.add_argument('--SACM', default=AUG_PROCESS_MODE)
    parser.add_argument('--classmix_num', default=CLASS_MIXNUM)
    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, default='./dataset/splits/' + DATASET + '/' + SPLIT + '/labeled.txt')
    parser.add_argument('--unlabeled-id-path', type=str, default='./dataset/splits/' + DATASET + '/' + SPLIT + '/unlabeled.txt')
    # parser.add_argument('--save-path', type=str, default='./exp/' + DATASET + '/' + SPLIT + '_' + SGM+ '_'+ str(SGM_LAMDA) + str(PERCENT) + '_' + str(LAMBDA) + '/' + AUG_PROCESS_MODE)
    parser.add_argument('--save-path', type=str,  default='./exp/' + DATASET + '/' + SPLIT + '_' + str(LAMBDA) + '/' + TRAIN_MODE + '/' +  THRESHOLD_MODE +'_' +
                        str(SINGLE_THRESHOLD) + '_to_' + str(FINAL_THSHOLD) + '/' + AUG_PROCESS_MODE + '_' + str(CLASS_MIXNUM) + '/' + SGM)
    parser.add_argument('--percent', type=float, default=PERCENT, help='0~100, the low-entropy percent r')
    args = parser.parse_args()
    return args


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main(args):
    create_path(args.save_path)

    criterion = CrossEntropyLoss(ignore_index=255)
    criterion_kl = KLDivLoss(reduction="none")
    criterion_mse = MSELoss()

    valset = SemiDataset(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(valset, batch_size=8, shuffle=False, pin_memory=False, num_workers=args.num_workers, drop_last=False)

    temp_datset_sup = SemiDataset(args.dataset, args.data_root, 'train_l', args.crop_size, args.labeled_id_path)
    tmp_trainloader_l = DataLoader(temp_datset_sup, batch_size=int(args.batch_size / 2), shuffle=True,
                               pin_memory=False, num_workers=args.num_workers, drop_last=True)
    l_sup = len(tmp_trainloader_l)
    print("l_sup:", l_sup)
    del temp_datset_sup, tmp_trainloader_l

    trainset_u = SemiDataset(args.dataset, args.data_root, 'train_u', args.crop_size, args.unlabeled_id_path)
    trainset_l = SemiDataset(args.dataset, args.data_root, 'train_l', args.crop_size, args.labeled_id_path, nsample=len(trainset_u.ids))
    # trainset_l = SemiDataset(args.dataset, args.data_root, 'train_l', args.crop_size, args.labeled_id_path)
    trainloader_u = DataLoader(trainset_u, batch_size=int(args.batch_size / 2), shuffle=True,
                               pin_memory=False, num_workers=args.num_workers, drop_last=True)
    trainloader_l = DataLoader(trainset_l, batch_size=int(args.batch_size / 2), shuffle=True,
                               pin_memory=False, num_workers=args.num_workers, drop_last=True)

    # trainloader_l = DataLoader(trainset_l, batch_size=16, shuffle=True,
    #                            pin_memory=False, num_workers=args.num_workers, drop_last=True)
    model, optimizer, lr_scheduler = init_basic_elems(args, trainloader_u)
    print('\nParams: %.1fM' % count_params(model))

    global_thresh = 1 / NUM_CLASSES[args.dataset]
    class_conf = torch.ones(NUM_CLASSES[args.dataset]) * global_thresh

    if args.train_mode == 'sup_only':
        train_sup(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, lr_scheduler, args)
    elif args.train_mode == 'WSCL':
        train_WSCL(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, lr_scheduler, args)
    # elif args.train_mode == 'fixmatch':
    #     train_fixmatch(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, lr_scheduler, args, global_thresh, class_conf)
    elif args.train_mode == 'dualmatch':
        train_dualmatch(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, lr_scheduler, args, l_sup)
    elif args.train_mode == 'dualmatch_SGM':
        train_dualmatch_SGM(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, lr_scheduler, args, l_sup)
    elif args.train_mode == 'multimatch':
        train_multimatch(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, lr_scheduler, args)
    elif args.train_mode == 'fixmatch_SGM':
        train_fixmatch_SGM(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, lr_scheduler,  args, global_thresh, class_conf, l_sup)
    elif args.train_mode == 'multimatch_SGM':
        train_multimatchh_SGM(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, lr_scheduler,  args, global_thresh, class_conf, l_sup)
    elif args.train_mode == 'train_SGM':
        SGM_pre(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, lr_scheduler, args, l_sup)
    elif args.train_mode == 'train_prototype':
        train_prototype(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, lr_scheduler, args, l_sup)
    elif args.train_mode == 'fixmatch':
        train_fixmatch(model, trainloader_l, trainloader_u, valloader, criterion, criterion_kl, criterion_mse, optimizer, lr_scheduler, args, global_thresh, class_conf)
    elif args.train_mode == 'train_llClassmatch':
        train_llClassmatch(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, lr_scheduler, args, global_thresh, class_conf)
    elif args.train_mode == 'RanPaste':
        train_RanPaste(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, lr_scheduler, args, global_thresh, class_conf)




def init_basic_elems(args, trainloader_u):
    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2,
                 'ResUNet': se_resnext50_32x4d}
    if args.model == 'ResUNet':
        model = model_zoo[args.model](NUM_CLASSES[args.dataset], None)
        pretrained_dict = torch.load(
            '/data1/users/lvliang/project_123/ClassHyPer-master/ClassHyPer-master/examples/save/se_resnext50_32x4d-a260b3a4.pth')
        my_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in my_dict}
        my_dict.update(pretrained_dict)
        model.load_state_dict(my_dict)
    else:
        model = model_zoo[args.model](args.backbone, NUM_CLASSES[args.dataset])
    head_lr_multiple = 1.0
    if args.model == 'deeplabv2':
        assert args.backbone == 'resnet101'
        head_lr_multiple = 1.0



    # optimizer = optim.SGD([
    #                  # {'params': model.backbone.parameters(), 'lr': args.lr},
    #                  {'params': [param for name, param in model.named_parameters()
    #                              if 'backbone' not in name],
    #                   'lr': args.lr * head_lr_multiple}],
    #                 lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.Adam([
    #     {'params': model.backbone.parameters(), 'lr': args.lr},
        # {'params': [param for name, param in model.named_parameters()
        #             if 'backbone' not in name],
        #  'lr': args.lr * head_lr_multiple}],
        # lr=args.lr,  weight_decay=1e-4)
    optimizer = optim.AdamW([
           # {'params': model.backbone.parameters(), 'lr': args.lr},
           {'params': [param for name, param in model.named_parameters()
                      if 'backbone' not in name],
           'lr': args.lr * head_lr_multiple}],
                              lr=args.lr,
                              betas=(0.9, 0.999),
                              eps=1e-08,
                              weight_decay=2e-4,
                              amsgrad=False
                              )
    lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0001 * 6,
                                                     steps_per_epoch=len(trainloader_u),
                                                     epochs=args.epochs,
                                                     div_factor=6)


    model = DataParallel(model).cuda()
    # model = model.cuda()
    return model, optimizer, lr_scheduler



def train_multimatchh_SGM(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, lr_scheduler, args, global_thresh, class_conf, l_sup):
    scaler = grad_scaler.GradScaler()

    iters = 0
    total_iters = len(trainloader_u) * args.epochs

    previous_best = 0.0
    previous_best_iou = 0.0
    weight_u = args.lamb


    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.9f\t\t\t\t\t previous best = %.2f  %s" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best, previous_best_iou))

        total_loss, total_loss_l, total_loss_u_s1, total_loss_u_s2, total_loss_u_s3 = 0.0, 0.0, 0.0, 0., 0.0
        multi_scale_class_memory = torch.zeros((4, 1, NUM_CLASSES[args.dataset], 128)).cuda()
        # multi_scale_class_memory = None
        tbar = tqdm(zip(trainloader_l, trainloader_u), total=len(trainloader_u))

        for i, ((img, mask), (img_u_w, img_u_s1, img_u_s2, img_u_s3, img_u_s4, img_u_s5)) in enumerate(tbar):
            img, mask = img.cuda(), mask.cuda()
            img_u_w, img_u_s1, img_u_s2, img_u_s3, img_u_s4, img_u_s5 = img_u_w.cuda(), \
            img_u_s1.cuda(), img_u_s2.cuda(), img_u_s3.cuda(), img_u_s4.cuda(), img_u_s5.cuda()
            optimizer.zero_grad()
            with autocast():
                model.train()
                # _, deep_features = model(img)
                if args.use_SGM:
                    if i <= l_sup:
                        multi_scale_class_memory = store_class_memory_ema\
                        (args, deep_features, mask, NUM_CLASSES[args.dataset], multi_scale_class_memory, i)
                else:
                    multi_scale_class_memory = None

                with torch.no_grad():
                    model.eval()
                    pred_u_w = model(img_u_w, multi_scale_class_memory)
                    prob_u_w = pred_u_w.softmax(dim=1)
                    conf_u_w, mask_u_w = prob_u_w.max(dim=1)

                model.train()
                num_lb, num_ulb = img.shape[0], img_u_w.shape[0]

                preds = model(torch.cat((img, img_u_s1, img_u_s2, img_u_s3)), multi_scale_class_memory)

                pred, pred_u_s1, pred_u_s2, pred_u_s3 = preds.split([num_lb, num_ulb, num_ulb, num_ulb])

                loss_l = criterion(pred, mask)
                loss_u_s1 = criterion(pred_u_s1, mask_u_w)
                loss_u_s2 = criterion(pred_u_s2, mask_u_w)
                loss_u_s3 = criterion(pred_u_s3, mask_u_w)

                global_thresh = args.global_lamda * global_thresh + (1 - args.global_lamda) * conf_u_w.mean()
                for class_idx in range(NUM_CLASSES[args.dataset]):
                    class_conf_new = 0 if len(conf_u_w[mask_u_w == class_idx]) == 0 else \
                        (conf_u_w[mask_u_w == class_idx]).sum() / len(conf_u_w[mask_u_w == class_idx])
                    class_conf[class_idx] = args.class_lamda * class_conf[class_idx] + (1 - args.class_lamda) * class_conf_new

                # if args.threshold_mode == 'single':

                if global_thresh <= 0.87:
                    loss_u_s1 = loss_u_s1 * (conf_u_w >= args.single_threshold)
                    loss_u_s2 = loss_u_s2 * (conf_u_w >= args.single_threshold)
                    loss_u_s3 = loss_u_s3 * (conf_u_w >= args.single_threshold)
                    loss_u_s1 = torch.mean(loss_u_s1)
                    loss_u_s2 = torch.mean(loss_u_s2)
                    loss_u_s3 = torch.mean(loss_u_s3)
                # elif args.threshold_mode == 'adaptive_threshold':
                else:
                    class_thresh = global_thresh.cuda() * class_conf.cuda() / class_conf.max()
                    for class_idx in range(NUM_CLASSES[args.dataset]):
                        conf_u_w[mask_u_w == class_idx] = conf_u_w[mask_u_w == class_idx] * \
                        (conf_u_w[mask_u_w == class_idx] > class_thresh[class_idx])
                    loss_u_s1 = loss_u_s1 * (conf_u_w > 0)
                    loss_u_s2 = loss_u_s2 * (conf_u_w > 0)
                    loss_u_s3 = loss_u_s3 * (conf_u_w > 0)
                    loss_u_s1 = torch.mean(loss_u_s1)
                    loss_u_s2 = torch.mean(loss_u_s2)
                    loss_u_s3 = torch.mean(loss_u_s3)
                # elif args.threshold_mode == 'adaptive_reweight':
                #     re_weight = torch.zeros(NUM_CLASSES[args.dataset])
                #     for class_idx in range(self.config.nb_classes):
                #         mask = (conf_u_w1[p_label_map == class_idx] < class_thresh[class_idx])
                #         loss_slice1 = loss_u_s_ce_w1[p_label_map == class_idx].clone()
                #         # if epoch <= self.config.warmup_period:
                #         weight = class_thresh[class_idx] if class_thresh[class_idx] < 0.5 else (1 - class_thresh[class_idx])
                #     #     # else:
                #     #     #     weight = class_thresh[class_idx] if class_thresh[class_idx] < 0.88 else (gama[class_idx]/class_thresh[class_idx])
                #         loss_slice1[mask] *= weight
                #         loss_u_s_ce_w1[p_label_map == class_idx] = loss_slice1
                #     # #
                #         loss_slice2 = loss_u_s_ce_w2[p_label_map == class_idx].clone()
                #         loss_slice2[mask] *= weight
                #         loss_u_s_ce_w2[p_label_map == class_idx] = loss_slice2
                #
                #         loss_slice3 = loss_u_s_ce_w3[p_label_map == class_idx].clone()
                #         loss_slice3[mask] *= weight
                #         loss_u_s_ce_w3[p_label_map == class_idx] = loss_slice3
                #
                #         re_weight[class_idx] = weight

                if i == len(trainloader_u) - 1:
                    print("epoch {} iter {}, global_thresh: {} \n".format(epoch, i, global_thresh))
                    print("epoch {} iter {}, class_conf: {} \n".format(epoch, i, class_conf))
                    # print("epoch {} iter {}, class_thresh: {} \n".format(epoch, i, class_thresh))
                    loss_u_s_ce = criterion(pred_u_s1, mask_u_w)
                    print("epoch {} iter {}, loss_u_s1_ce: {} \n".format(epoch, i, loss_u_s_ce))
                    print("epoch {} iter {}, loss_u_s1_ce_weight: {} \n".format(epoch, i, loss_u_s1))
                    # print("epoch {} iter {}, loss_u_s: {} \n".format(epoch, i, loss_u))

                loss = (loss_l + weight_u * loss_u_s1 + weight_u * loss_u_s2 + weight_u * loss_u_s2) / 4

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()

            total_loss += loss.item()
            total_loss_l += loss_l.item()
            total_loss_u_s1 += loss_u_s1.item()
            total_loss_u_s2 += loss_u_s2.item()
            total_loss_u_s3 += loss_u_s3.item()

            # iters += 1
            # lr = args.lr * (1 - iters / total_iters) ** 0.9
            # optimizer.param_groups[0]["lr"] = lr

            lr_scheduler.step()

            tbar.set_description('Loss: %.3f, Loss_l: %.3f, Loss_u_s1: %.3f, Loss_u_s2: %.3f,  Loss_u_s3: %.3f' % (
                total_loss / (i + 1), total_loss_l / (i + 1), total_loss_u_s1 / (i + 1), total_loss_u_s2 / (i + 1),
                total_loss_u_s3 / (i + 1)))
            # tbar.set_description('Loss: %.3f, Loss_l: %.3f, Loss_u_s1: %.3f, Loss_u_s2: %.3f' % (
            #     total_loss / (i + 1), total_loss_l / (i + 1), total_loss_u_s1 / (i + 1), total_loss_u_s2 / (i + 1)))

        if (epoch + 1) % 1 == 0:
            metric = meanIOU(num_classes=NUM_CLASSES[args.dataset])

            model.eval()
            tbar = tqdm(valloader)

            with torch.no_grad():
                for img, mask, _ in tbar:
                    img = img.cuda()
                    pred = model(img, multi_scale_class_memory)
                    pred = torch.argmax(pred, dim=1)

                    metric.add_batch(pred.cpu().numpy(), mask.numpy())
                    IOU, mIOU = metric.evaluate()

                    tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

            mIOU *= 100.0
            IOU *= 100
            print('IoU: {}  | MIoU: {}'.format(IOU, mIOU))
            if mIOU > previous_best:
                # if previous_best != 0:
                #     os.remove(
                #         os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
                previous_best = mIOU
                previous_best_iou = IOU
                if args.use_SGM:
                    torch.save(model.module.state_dict(),
                    # torch.save(model.state_dict(),
                    #            os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))
                               os.path.join(args.save_path, '%s_%s_%s_%s_%.4f_SGM_%.4f_epoch_%d_%.2f.pth' %
                            (args.model, args.backbone, args.train_mode, args.threshold_mode, args.single_threshold, args.SGM_lamda, epoch, mIOU)))
                else:
                    torch.save(model.module.state_dict(),
                               # torch.save(model.state_dict(),
                               #            os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))
                     os.path.join(args.save_path, '%s_%s_%s_%s_%.4f_epoch_%d_%.2f_(2).pth' %
                    (args.model, args.backbone, args.train_mode, args.threshold_mode, args.single_threshold, epoch, mIOU)))

def train_multimatch(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, lr_scheduler, args):
    scaler = grad_scaler.GradScaler()

    iters = 0
    total_iters = len(trainloader_u) * args.epochs

    previous_best = 0.0
    previous_best_iou = 0.0
    weight_u = args.lamb

    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.9f\t\t\t\t\t previous best = %.2f  %s" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best, previous_best_iou))

        threshold = args.single_threshold
        if args.threshold_mode == 'adaptive_decay':
            threshold = calculate_decay_threshold(args.single_threshold, args.final_threshold, 0, args.epochs, epoch)
            print(f"\nepoch {epoch}/End epoch {args.epochs}, Epoch {epoch}: Initial threshold = "
                  f"{args.single_threshold:.4f}, Decayed threshold = {threshold:.4f}")

        total_loss, total_loss_l, total_loss_u_s1, total_loss_u_s2, total_loss_u_s3 = 0.0, 0.0, 0.0, 0., 0.0

        tbar = tqdm(zip(trainloader_l, trainloader_u), total=len(trainloader_u))
        # for i, ((img, mask), (img_u_w, img_u_s1, img_u_s2, img_u_s3, img_u_s4, img_u_s5)) in enumerate(tbar):
        #     img, mask = img.cuda(), mask.cuda()
        #     img_u_w, img_u_s1, img_u_s2, img_u_s3, img_u_s4, img_u_s5 = \
        #         img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda(), img_u_s3.cuda(), img_u_s4.cuda(), img_u_s5.cuda()
        #     optimizer.zero_grad()
        #     with autocast():
        #         with torch.no_grad():
        #             model.eval()
        #             pred_u_w = model(img_u_w)
        #             prob_u_w = pred_u_w.softmax(dim=1)
        #             conf_u_w, mask_u_w = prob_u_w.max(dim=1)
        for i, ((img, mask,_,_), (img_u_w, img2_u_w, cutmix_box, img_u_s1, img_u_s2, img_u_s3, img_u_s4, img_u_s5)) in enumerate(tbar):
            img, mask = img.cuda(), mask.cuda()
            img_u_w, img2_u_w, cutmix_box, img_u_s1, img_u_s2, img_u_s3, img_u_s4, img_u_s5 = \
            img_u_w.cuda(), img2_u_w.cuda(), cutmix_box.cuda(), img_u_s1.cuda(), img_u_s2.cuda(), img_u_s3.cuda(), img_u_s4.cuda(), img_u_s5.cuda()
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

                num_lb, num_ulb = img.shape[0], img_u_w.shape[0]
                # pred = model(img)
                # preds = model(torch.cat((img, img_u_s1, img_u_s2, img_u_s3)))
                # pred, pred_u_s1, pred_u_s2, pred_u_s3 = preds.split([num_lb, num_ulb, num_ulb, num_ulb])

                preds = model(torch.cat((img, img_u_s1, img_u_s2, img_u_s3)))
                pred, pred_u_s1, pred_u_s2, pred_u_s3 = preds.split([num_lb, num_lb, num_ulb, num_lb])

                loss_l = criterion(pred, mask)
                loss_u_s1 = criterion(pred_u_s1, mask_u_w)
                loss_u_s2 = criterion(pred_u_s2, mask_u_w)
                loss_u_s3 = criterion(pred_u_s3, mask_u_w)
                # loss_u_s4 = criterion(pred_u_s4, mask_u_w)
                # loss_u_s5 = criterion(pred_u_s5, mask_u_w)

                # if global_conf <= max_value:
                #     single_threshold = args.single_threshold
                # else:
                #     single_threshold = calculate_decay_rate(initial_value, final_value, start_epoch, end_epoch)

                loss_u_s1 = loss_u_s1 * (conf_u_w >= threshold)
                loss_u_s2 = loss_u_s2 * (conf_u_w >= threshold)
                loss_u_s3 = loss_u_s3 * (conf_u_w >= threshold)
                # loss_u_s4 = loss_u_s4 * (conf_u_w >= threshold)
                # loss_u_s5 = loss_u_s5 * (conf_u_w >= threshold)
                loss_u_s1 = torch.mean(loss_u_s1)
                loss_u_s2 = torch.mean(loss_u_s2)
                loss_u_s3 = torch.mean(loss_u_s3)
                # loss_u_s4 = torch.mean(loss_u_s4)
                # loss_u_s5 = torch.mean(loss_u_s5)

                loss = (loss_l + weight_u * loss_u_s1 + weight_u * loss_u_s2 +
                weight_u * loss_u_s3 + weight_u ) / 4

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()

            total_loss += loss.item()
            total_loss_l += loss_l.item()
            total_loss_u_s1 += loss_u_s1.item()
            total_loss_u_s2 += loss_u_s2.item()
            total_loss_u_s3 += loss_u_s3.item()

            iters += 1

            lr_scheduler.step()

            # iters += 1
            # lr = args.lr * (1 - iters / total_iters) ** 0.9
            # optimizer.param_groups[0]["lr"] = lr

            tbar.set_description('Loss: %.3f, Loss_l: %.3f, Loss_u_s1: %.3f, Loss_u_s2: %.3f,  Loss_u_s3: %.3f' % (
                total_loss / (i + 1), total_loss_l / (i + 1), total_loss_u_s1 / (i + 1), total_loss_u_s2 / (i + 1),
                total_loss_u_s3 / (i + 1)))

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
            # if mIOU > previous_best:
            #     # if previous_best != 0:
            #     #     os.remove(
            #     #         os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
            #     previous_best = mIOU
            #     previous_best_iou = IOU
            #     # torch.save(model.module.state_dict(),
            #     #            os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))
            #     torch.save(model.module.state_dict(),
            #                # torch.save(model.state_dict(),
            #                #            os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))
            #                os.path.join(args.save_path, '%s_%s_%s_%s_%.4f_epoch_%d_%.2f.pth' %
            #                             (args.model, args.backbone, args.train_mode, args.threshold_mode,
            #                              args.single_threshold, epoch, mIOU)))
            if mIOU > previous_best:

                if previous_best != 0:
                    os.remove(os.path.join(args.save_path, '%s_%s_%s_%s_%.2f_weight_%.2f_epoch_%d_%.2f.pth' %
                    (args.model, args.backbone, args.train_mode, args.threshold_mode, args.single_threshold, weight_u, previous_epoch, previous_best)))
                previous_best = mIOU
                previous_best_iou = IOU
                previous_epoch = epoch
                torch.save(model.module.state_dict(), os.path.join(args.save_path, '%s_%s_%s_%s_%.2f_weight_%.2f_epoch_%d_%.2f.pth' %
                (args.model, args.backbone, args.train_mode, args.threshold_mode, args.single_threshold, weight_u, epoch, mIOU)))


def train_MCSS(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, lr_scheduler, args, l_sup):
    scaler = grad_scaler.GradScaler()

    iters = 0
    total_iters = len(trainloader_u) * args.epochs

    previous_best = 0.0
    previous_best_iou = 0.0
    IOU = torch.ones(NUM_CLASSES[args.dataset]).cuda() * 99
    weight_u = args.lamb
    start_epoch = None

    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.9f\t\t\t\t\t previous best = %.2f  %s" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best, previous_best_iou))

        threshold = args.single_threshold
        if args.threshold_mode == 'adaptive_decay':
            threshold = calculate_decay_threshold(args.single_threshold, args.final_threshold, 0, args.epochs, epoch)
            print(f"\nStart epoch {start_epoch}/End epoch {args.epochs}, Epoch {epoch}: Initial threshold = "
                  f"{args.single_threshold:.4f}, Decayed threshold = {threshold:.4f}")
        # print('weights = 1 - IOU / 100: ', 1 - IOU / 100)

        total_loss, total_loss_l, total_loss_l_p, total_loss_u_s1, total_loss_p_u_s1, total_loss_u_s2, total_loss_p_u_s2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        multi_scale_class_memory = torch.zeros((4, 1, NUM_CLASSES[args.dataset], 128)).cuda()
        parral_multi_scale_class_memory = torch.zeros((2, 4, 1, NUM_CLASSES[args.dataset], 128)).cuda()

        tbar = tqdm(zip(trainloader_l, trainloader_u), total=len(trainloader_u))

        for i, ((img, mask, img2, mask2), (img_u_w, img2_u_w, cutmix_box, img_u_s1, img_u_s2, _, _, _)) in enumerate(tbar):
            img, mask, img2, mask2 = img.cuda(), mask.cuda(), img2.cuda(), mask2.cuda()
            img_u_w, img2_u_w, cutmix_box, img_u_s1, img_u_s2 = img_u_w.cuda(), img2_u_w.cuda(), cutmix_box.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
            optimizer.zero_grad()
            num_lb, num_ulb = img.shape[0], img_u_w.shape[0]


            if args.SACM  == 'SACM':
                class_indices = torch.arange(0, NUM_CLASSES[args.dataset])
                # 从class_indices中根据class_conf的权重随机采样3个类别
                selected_classes = random.choices(class_indices, weights=(1 - IOU/100), k=args.classmix_num)
                img_u_w, img2_u_w, img_u_s1, img_u_s2, classmix_matric = \
                self_adaptive_class_mix(selected_classes, img_u_w, img2_u_w, img_u_s1, img_u_s2, img2[0], mask2[0])

            with autocast():
                model.eval()
                _, deep_features = model(img)
                if args.use_SGM == 'SGM':
                    if i <= l_sup:
                        multi_scale_class_memory = store_class_memory_ema \
                        (args, deep_features, mask, NUM_CLASSES[args.dataset], multi_scale_class_memory, i)
                        parral_multi_scale_class_memory[0] = multi_scale_class_memory
                        parral_multi_scale_class_memory[1] = multi_scale_class_memory
                    # store_class_memory()
                else:
                    multi_scale_class_memory = None

                with torch.no_grad():
                    model.eval()
                    preds_u_w, _ = model(torch.cat((img_u_w, img2_u_w)), parral_multi_scale_class_memory)
                    pred_u_w, pred2_u_w = preds_u_w.split([num_lb, num_lb])
                    prob_u_w = pred_u_w.softmax(dim=1)
                    conf_u_w, mask_u_w = prob_u_w.max(dim=1)
                    prob2_u_w = pred2_u_w.softmax(dim=1)
                    conf2_u_w, mask2_u_w = prob2_u_w.max(dim=1)
                    mask_u_w[cutmix_box == 1] = mask2_u_w[cutmix_box == 1]
                    if args.SACM  == 'SACM' and classmix_matric != None:
                        mask_u_w[:, classmix_matric == 1] = mask2[0][classmix_matric == 1]
                        # print("selected class:{}\n, selected_pixel_num:{}\n".format(selected_classes, (classmix_matric==1).sum()))

                model.train()

                num_lb, num_ulb = img.shape[0], img_u_w.shape[0]
                preds, prototype_preds = model(torch.cat((img, img_u_s1, img_u_s2)), None)
                pred, pred_u_s1, pred_u_s2 = preds.split([num_lb, num_ulb, num_lb])

                # prototype_pred, prototype_u_s1, prototype_u_s2 = prototype_preds.split([num_lb, num_ulb, num_lb])
                # prototype2_pred, prototype2_u_s1, prototype2_u_s2 = prototype_preds[1].split([num_lb, num_ulb, num_lb])

                loss_l = criterion(pred, mask)
                # loss_l_p = criterion(prototype_pred, mask)
                # loss_l_p2 = criterion(prototype2_pred, mask)

                loss_u_s1 = criterion(pred_u_s1, mask_u_w)
                # loss_u_s1_p = criterion(prototype_u_s1, mask_u_w)
                # loss_u_s1_p2 = criterion(prototype2_u_s1, mask_u_w)

                loss_u_s2 = criterion(pred_u_s2, mask_u_w)

                # loss_u_s2_p = criterion(prototype_u_s2, mask_u_w)
                # loss_u_s2_p2 = criterion(prototype2_u_s2, mask_u_w)


                loss_u_s1 = loss_u_s1 * (conf_u_w >= threshold)
                loss_u_s2 = loss_u_s2 * (conf_u_w >= threshold)
                loss_u_s1 = torch.mean(loss_u_s1)
                loss_u_s2 = torch.mean(loss_u_s2)

                # loss_u_s1_p = loss_u_s1_p * (conf_u_w >= threshold)
                # loss_u_s1_p2 = loss_u_s1_p2 * (conf_u_w >= threshold)

                # loss_u_s2_p = loss_u_s2_p * (conf_u_w >= threshold)
                # loss_u_s2_p2 = loss_u_s2_p2 * (conf_u_w >= threshold)

                # loss_u_s1_p = torch.mean(loss_u_s1_p)
                # loss_u_s2_p = torch.mean(loss_u_s2_p)


                # loss_u_s2_p1 = torch.mean(loss_u_s2_p1)
                # loss_u_s2_p2 = torch.mean(loss_u_s2_p2)


                # loss_l_all = (loss_l + loss_l_p) / 2
                # loss_u_s1_all = (weight_u * loss_u_s1 + weight_u * loss_u_s1_p) / 2
                # loss_u_s2_all = (weight_u * loss_u_s2 + weight_u * loss_u_s2_p) / 2

                loss = (loss_l  + weight_u * loss_u_s1 + weight_u * loss_u_s2) / 3
                # loss = loss_l + loss_u_s1_all + loss_u_s2_all

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()

            total_loss += loss.item()
            total_loss_l += loss_l.item()
            total_loss_u_s1 += loss_u_s1.item()
            total_loss_u_s2 += loss_u_s2.item()

            # total_loss_l_p += loss_l_all.item()
            # total_loss_p_u_s1 += loss_u_s1_all.item()
            # total_loss_p_u_s2 += loss_u_s2_all.item()

            iters += 1

            lr_scheduler.step()

            tbar.set_description('Loss: %.3f, Loss_l: %.3f, Loss_u_s1: %.3f, Loss_u_s2: %.3f' % (
            total_loss / (i + 1), loss_l / (i + 1), loss_u_s1 / (i + 1), loss_u_s2 / (i + 1)))

            # print('Loss: %.3f, Loss_l: %.3f, Loss_u_s1: %.3f, Loss_u_s2: %.3f\n' % (total_loss / (i + 1), total_loss_l / (i + 1), total_loss_u_s1 / (i + 1), total_loss_u_s2 / (i + 1)))

            # tbar.set_description('Loss_l_p: %.3f, loss_u_s1_p: %.3f, loss_u_s2_p: %.3f' % (
            #       total_loss_l_p / (i + 1), total_loss_p_u_s1 / (i + 1), total_loss_p_u_s2 / (i + 1)))

            # print('Loss_l_p: %.3f, loss_u_s1_p: %.3f, loss_u_s2_p: %.3f' % (
            #       total_loss_l_p / (i + 1), total_loss_p_u_s1 / (i + 1), total_loss_p_u_s2 / (i + 1)))

        if (epoch + 1) % 1 == 0:
            metric = meanIOU(num_classes=NUM_CLASSES[args.dataset])
            metric_proto = meanIOU(num_classes=NUM_CLASSES[args.dataset])

            model.eval()
            tbar = tqdm(valloader)

            with torch.no_grad():
                for img, mask, _ in tbar:
                    img = img.cuda()
                    pred, proto_preds = model(img, parral_multi_scale_class_memory)
                    pred = torch.argmax(pred, dim=1)
                    proto_pred = torch.argmax(proto_preds, dim=1)

                    metric.add_batch(pred.cpu().numpy(), mask.numpy())
                    IOU, mIOU = metric.evaluate()
                    tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))
                    metric_proto.add_batch(proto_pred.cpu().numpy(), mask.numpy())
                    proto_IOU, proto_mIOU = metric_proto.evaluate()
                    tbar.set_description('proto_mIOU: %.2f' % (proto_mIOU * 100.0))


            mIOU *= 100.0
            IOU *= 100
            proto_mIOU *= 100.0
            proto_IOU *= 100
            print('IoU: {}  | MIoU: {}'.format(IOU, mIOU))
            print('proto_IOU: {}  | proto_mIOU: {}'.format(proto_IOU, proto_mIOU))
            if mIOU > previous_best:
                state = {'state_dict': model.module.state_dict(), 'class_momory': parral_multi_scale_class_memory}
                # file_name =  '%s_%s_%s_%s_%.2f_to_%.2f_weight_%.2f_epoch_%d_%.2f.pth' %
                # (args.model, args.backbone, args.train_mode, args.threshold_mode, args.single_threshold, args.final_threshold, weight_u, epoch, mIOU)
                if previous_best != 0:
                    os.remove(os.path.join(args.save_path,'%s_%s_%s_%.2f_%s_%.2f_to_%.2f_weight_%.2f_epoch_%d_%.2f.pth' %
                   (args.model, args.backbone, args.train_mode, args.SGM_lamda, args.threshold_mode, args.single_threshold,
                    args.final_threshold, weight_u, previous_epoch, previous_best)))
                previous_best = mIOU
                previous_best_iou = IOU
                previous_epoch = epoch
                torch.save(state, os.path.join(args.save_path, '%s_%s_%s_%.2f_%s_%.2f_to_%.2f_weight_%.2f_epoch_%d_%.2f.pth' %
                (args.model, args.backbone, args.train_mode, args.SGM_lamda, args.threshold_mode, args.single_threshold, args.final_threshold, weight_u, epoch, mIOU)))



def train_dualmatch(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, lr_scheduler, args, l_sup):
    scaler = grad_scaler.GradScaler()

    iters = 0
    total_iters = len(trainloader_u) * args.epochs

    previous_best = 0.0
    previous_best_iou = 0.0
    IOU = torch.ones(NUM_CLASSES[args.dataset]).cuda() * 99
    weight_u = args.lamb

    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.9f\t\t\t\t\t previous best = %.2f  %s" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best, previous_best_iou))

        threshold = args.single_threshold
        if args.threshold_mode == 'adaptive_decay':
            threshold = calculate_decay_threshold(args.single_threshold, args.final_threshold, 0, args.epochs, epoch)
            print(f"\nepoch {epoch}/End epoch {args.epochs}, Epoch {epoch}: Initial threshold = "
                  f"{args.single_threshold:.4f}, Decayed threshold = {threshold:.4f}")
        # print('weights = 1 - IOU / 100: ', 1 - IOU / 100)

        total_loss, total_loss_l, total_loss_l_p, total_loss_u_s1, total_loss_p_u_s1, total_loss_u_s2, total_loss_p_u_s2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        tbar = tqdm(zip(trainloader_l, trainloader_u), total=len(trainloader_u))

        for i, ((img, mask, img2, mask2), (img_u_w, img2_u_w, cutmix_box, img_u_s1, img_u_s2, _, _, _)) in enumerate(tbar):
            img, mask, img2, mask2 = img.cuda(), mask.cuda(), img2.cuda(), mask2.cuda()
            img_u_w, img2_u_w, cutmix_box, img_u_s1, img_u_s2 = img_u_w.cuda(), img2_u_w.cuda(), cutmix_box.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
            optimizer.zero_grad()
            num_lb, num_ulb = img.shape[0], img_u_w.shape[0]


            if args.SACM == 'SACM':
                class_indices = torch.arange(0, NUM_CLASSES[args.dataset])
                # 从class_indices中根据class_conf的权重随机采样3个类别
                selected_classes = random.choices(class_indices, weights=(1 - IOU/100), k=args.classmix_num)
                img_u_w, img2_u_w, img_u_s1, img_u_s2, classmix_matric = \
                self_adaptive_class_mix(selected_classes, img_u_w, img2_u_w, img_u_s1, img_u_s2, img2[0], mask2[0])

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
                    if args.SACM == 'SACM' and classmix_matric != None:
                        mask_u_w[:, classmix_matric == 1] = mask2[0][classmix_matric == 1]
                        # print("selected class:{}\n, selected_pixel_num:{}\n".format(selected_classes, (classmix_matric==1).sum()))

                model.train()

                num_lb, num_ulb = img.shape[0], img_u_w.shape[0]
                preds = model(torch.cat((img, img_u_s1, img_u_s2)))
                pred, pred_u_s1, pred_u_s2 = preds.split([num_lb, num_ulb, num_lb])


                loss_l = criterion(pred, mask)

                loss_u_s1 = criterion(pred_u_s1, mask_u_w)

                loss_u_s2 = criterion(pred_u_s2, mask_u_w)

                loss_u_s1 = loss_u_s1 * (conf_u_w >= threshold)
                loss_u_s2 = loss_u_s2 * (conf_u_w >= threshold)
                loss_u_s1 = torch.mean(loss_u_s1)
                loss_u_s2 = torch.mean(loss_u_s2)

                loss = loss_l + weight_u * loss_u_s1 + weight_u * loss_u_s2

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()

            total_loss += loss.item()
            total_loss_l += loss_l.item()
            total_loss_u_s1 += loss_u_s1.item()
            total_loss_u_s2 += loss_u_s2.item()

            total_loss_l_p += loss_l.item()
            total_loss_p_u_s1 += loss_u_s1.item()
            total_loss_p_u_s2 += loss_u_s2.item()

            iters += 1

            lr_scheduler.step()

            tbar.set_description('Loss: %.3f, Loss_l: %.3f, Loss_u_s1: %.3f, Loss_u_s2: %.3f' % (
            total_loss / (i + 1), total_loss_l / (i + 1), total_loss_u_s1 / (i + 1), total_loss_u_s2 / (i + 1)))


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
                    os.remove(os.path.join(args.save_path, '%s_%s_%s_%s_%.2f_to_%.2f_weight_%.2f_epoch_%d_%.2f.pth' %
                   (args.model, args.backbone, args.train_mode, args.threshold_mode, args.single_threshold,
                    args.final_threshold, weight_u, previous_epoch, previous_best)))
                previous_best = mIOU
                previous_best_iou = IOU
                previous_epoch = epoch
                torch.save(model.module.state_dict(), os.path.join(args.save_path, '%s_%s_%s_%s_%.2f_to_%.2f_weight_%.2f_epoch_%d_%.2f.pth' %
                (args.model, args.backbone, args.train_mode, args.threshold_mode, args.single_threshold, args.final_threshold, weight_u, epoch, mIOU)))











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
            tbar.set_description('Loss: %.3f, Loss_l: %.3f, Loss_u: %.3f' % (total_loss / (i + 1), total_loss_l / (i + 1), total_loss_u / (i + 1)))

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
                    os.remove(os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
                previous_best = mIOU
                previous_best_iou = IOU
                # torch.save(model.module.state_dict(),
                torch.save(model.state_dict(),
                           os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))



if __name__ == '__main__':
    args = parse_args()
    if args.epochs is None:
        args.epochs = {'GID-15': 100, 'iSAID': 100, 'MER': 100, 'MSL': 100, 'Vaihingen': 100, 'DFC22': 100}[args.dataset]
    if args.lr is None:
        args.lr = {'GID-15': 0.001, 'iSAID': 0.001, 'MER': 0.001, 'MSL': 0.001,
                   'Vaihingen': 0.001, 'DFC22': 0.001}[args.dataset] / 16 * args.batch_size
    if args.crop_size is None:
        args.crop_size = {'GID-15': 320, 'iSAID': 320, 'MER': 320, 'MSL': 320, 'Vaihingen': 320, 'DFC22': 320}[args.dataset]
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
