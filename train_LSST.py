
from utils import count_params, meanIOU, color_map
import cv2
import argparse
from copy import deepcopy
import timeit
import datetime
import random

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

from torch.nn import CrossEntropyLoss, DataParallel
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.multiprocessing
from torch.cuda.amp import autocast
from torch.cuda.amp import grad_scaler


start = timeit.default_timer()
start_writeable = datetime.datetime.now().strftime('%m-%d_%H-%M')

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    # if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
        
seed = 1234
set_random_seed(seed)

MODE = None

DATASET = 'DFC22'     # ['DFC22', 'iSAID', 'MER', 'MSL', 'Vaihingen', 'GID-15']
SPLIT = '1-8'     # ['1-4', '1-8', '100', '300']
DFC22_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/DFC22/'
iSAID_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/iSAID/'
MER_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/MER/'
MSL_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/MSL/'
Vaihingen_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/Vaihingen/WCSL_crop/Vaihingen/'
# Vaihingen_DATASET_PATH ='/data1/users/lvliang/project_123/WSCL-main/WSCL-main/dataset/splits/Vaihingen/1-8/save/SEResUNet/20230828_170334_seed1234/'

GID15_DATASET_PATH = '/data1/users/lvliang/project_123/dataset/GID-15/'

RATIO = 0.2
NUM_CLASSES = {'GID-15': 15, 'iSAID': 15, 'DFC22': 12, 'MER': 9, 'MSL': 9, 'Vaihingen': 5}

def parse_args():
    parser = argparse.ArgumentParser(description='LSST Framework')

    # basic settings
    parser.add_argument('--data-root', type=str, default=DFC22_DATASET_PATH)
    parser.add_argument('--dataset', type=str, choices=['GID-15', 'iSAID', 'DFC22', 'MER', 'MSL', 'Vaihingen'], default=DATASET)
    parser.add_argument('--ratio', type=float, default=RATIO, help='0-1')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet50')
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2', 'ResUNet'],
                        default='ResUNet')
    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, default='./dataset/splits/' + DATASET + '/' + SPLIT + '/labeled.txt')
    parser.add_argument('--unlabeled-id-path', type=str, default='./dataset/splits/' + DATASET + '/' + SPLIT + '/unlabeled.txt')
    parser.add_argument('--pseudo-mask-path', type=str, default='./output/' + DATASET + '/' + SPLIT + '_' + str(RATIO) + '/pseudo_masks')
    parser.add_argument('--save-path', type=str, default='./output/' + DATASET + '/' + SPLIT + '_' + str(RATIO) + '/models')

    args = parser.parse_args()
    return args


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main(args):
    create_path(args.save_path)
    create_path(args.pseudo_mask_path)

    criterion = CrossEntropyLoss(ignore_index=255)

    valset = SemiDataset(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, drop_last=False)

    print('\n================> Total stage 1/3: Supervised training on labeled images (SupOnly)')

    global MODE
    MODE = 'train'

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size, args.labeled_id_path)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=8, drop_last=True)

    model, optimizer = init_basic_elems(args)
    print('\nParams: %.1fM' % count_params(model))
    best_model = train(model, trainloader, valloader, criterion, optimizer, args)

    """
        Adaptive Pseudo-Labeling
    """
    print('\n\n\n================> Total stage 2/3: Adaptive Pseudo labeling all unlabeled images')

    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    sparse_label(best_model, dataloader, args)

    print('\n\n\n================> Total stage 3/3: Re-training on labeled and unlabeled images')

    MODE = 'semi_train'

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                           args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=8, drop_last=True)

    model, optimizer = init_basic_elems(args)
    train(model, trainloader, valloader, criterion, optimizer, args)

    end = timeit.default_timer()
    print('Total time: ' + str(end - start) + ' seconds')


def init_basic_elems(args):
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

    return model, optimizer


def train(model, trainloader, valloader, criterion, optimizer, args):
    iters = 0
    total_iters = len(trainloader) * args.epochs

    previous_best = 0.0
    previous_best_iou = 0.0

    global MODE
    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f  %s" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best, previous_best_iou))

        model.train()
        total_loss = 0.0
        tbar = tqdm(trainloader)

        for i, (img, mask) in enumerate(tbar):
            img, mask = img.cuda(), mask.cuda()

            pred = model(img)
            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters += 1
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0

            tbar.set_description('Loss: %.3f' % (total_loss / (i + 1)))

        if (epoch + 1) % 10 == 0:
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
                torch.save(model.module.state_dict(),
                           os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))

                best_model = deepcopy(model)

    return best_model


def sparse_label(model, dataloader, args):
    model.eval()
    tbar = tqdm(dataloader)

    with torch.no_grad():
        for img, _, id in tbar:
            img = img.cuda()
            pred = model(img)
            soft_max_output, hard_output = pred.max(dim=1)
            for j in range(soft_max_output.shape[0]):
                soft, hard = soft_max_output[j].cpu().numpy(), hard_output[j].cpu().numpy()
                need = []
                for c in range(NUM_CLASSES[args.dataset]):
                    soft_clone, hard_clone = deepcopy(soft), deepcopy(hard)
                    need.append(ratio_sample(hard_clone, soft_clone, args.ratio, c))
                need = np.min(np.array(need), axis=0)

                pred = Image.fromarray(need.astype(np.uint8), mode='P')
                pred.save('%s/%s' % (args.pseudo_mask_path, os.path.basename(id[j].split(' ')[1])))


def ratio_sample(hard_out, soft_max_out, ratio, s_class):
    single_h = hard_out
    single_s = soft_max_out
    h_index = (single_h != s_class)
    single_h[h_index] = 255
    single_s[h_index] = 0
    all = sorted(soft_max_out[(hard_out == s_class)], reverse=True)
    num = len(all)
    need_num = int(num * ratio + 0.5)
    if need_num != 0:
        adaptive_threshold = all[need_num - 1]
        mask = (single_s >= adaptive_threshold)
        index = (mask == False)
        single_h[index] = 255
    else:
        single_h[(single_h != 255)] = 255

    return single_h
               
             
if __name__ == '__main__':
    args = parse_args()
    if args.epochs is None:
        args.epochs = {'GID-15': 50, 'iSAID': 50, 'MER': 50, 'MSL': 50, 'Vaihingen': 50, 'DFC22': 50}[args.dataset]
    if args.lr is None:
        args.lr = {'GID-15': 0.001, 'iSAID': 0.001, 'MER': 0.001, 'MSL': 0.001,
                   'Vaihingen': 0.001, 'DFC22': 0.001}[args.dataset] / 16 * args.batch_size
    if args.crop_size is None:
        args.crop_size = {'GID-15': 321, 'iSAID': 321, 'MER': 321, 'MSL': 321, 'Vaihingen': 321, 'DFC22': 321}[args.dataset]
    if args.data_root is None:
        args.data_root = {'GID-15': GID15_DATASET_PATH,
                          'iSAID': iSAID_DATASET_PATH,
                          'MER': MER_DATASET_PATH,
                          'MSL': MSL_DATASET_PATH,
                          'Vaihingen': Vaihingen_DATASET_PATH,
                          'DFC22': DFC22_DATASET_PATH}[args.dataset]

    print(args)

    main(args)
