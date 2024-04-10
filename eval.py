import numpy as np
import torch
import torch.nn.functional as F
# from utils import rgb_to_labels, color_list, rgb_to_mask, predict_img, pointrend_predict_img
from sklearn.metrics import confusion_matrix, cohen_kappa_score, recall_score, accuracy_score
import torch.nn as nn
import os
from PIL import Image
# from pointrend.sampling_points import *

# f1_score()



def train_kappa(pre, val_masks):
    cm2lbl = color_list()
    imgs_masks = []

    kappa_cof = 0
    # epoch_kappa = 0
    true_index = []
    pre_index = []
    for j in range(len(pre)):
        predict = torch.argmax(pre[j], dim=0).long()
        predict = predict.cpu().detach().numpy()
        data = val_masks[j].astype('int64')
        idx = (data[0, :, :] * 256 + data[1, :, :]) * 256 + data[2, :, :]
        idx = idx.astype('int32')
        true = cm2lbl[idx]
        true = true.astype('int64')
        true_index.append(true)
        pre_index.append(predict)
        # conf_matrix = confusion_matrix(true.reshape(-1),predict.reshape(-1))
        # kappa = cohen_kappa_score(true.reshape(-1),predict.reshape(-1))
        # epoch_kappa += kappa
    kappa = cohen_kappa_score(np.array(true_index).reshape(-1), np.array(pre_index).reshape(-1))
    # kappa_mean = epoch_kappa/len(pre)
    # kappa_cof += kappa
    # num = i
    # kappa_cof_mean = kappa_cof/(num+1)
    # kappa /= len(pre)

    # return np.array(kappa_cof).mean()
    # epoch_kappa += kappa
    return kappa



def out_to_rgb(out_index):

    colormap = np.array([[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]])
    # colormap = np.array([[0, 0, 0], [0, 127, 127], [0, 0, 127], [0, 127, 255], [0, 0, 63], [0, 100, 155], [0, 63, 127], [0, 63, 255], [0, 127, 191],
    # [0, 0, 255], [0, 63, 0], [0, 63, 63], [0, 0, 191], [0, 63, 191], [0, 191, 127], [0, 127, 63]])
    rgb_img = colormap[out_index]
    return rgb_img

def train_visualization(x_batch, y_batch, edge_batch, masks_pred, edge_pre, epoch):
    """
    :param x_batch:
    :param y_batch:
    :param masks_pred:
    :param edge_pre:
    :return:
    """
    x_batch = x_batch.cpu().numpy()
    y_batch = y_batch.cpu().numpy()
    edge_batch = edge_batch.cpu().numpy()
    for i in range(len(x_batch)):
        img = (x_batch[i] * 255).astype(np.uint8)
        img = img.transpose(1, 2, 0)
        label = (y_batch[i]).astype(np.uint8)

        # label = np.expand_dims(label, axis=2)
        # label = np.repeat(label, 3, axis=2)
        label = label.transpose(1, 2, 0)


        edge_label = (edge_batch[i] * 255).astype(np.uint8)
        edge_label = np.expand_dims(edge_label, axis=2)
        edge_label = np.repeat(edge_label, 3, axis=2)

        model_seg = torch.argmax(masks_pred[i], dim=0).long()
        model_seg = model_seg.cpu().detach().numpy()
        # model_seg = (model_seg * 255).astype(np.uint8)
        # model_seg = np.expand_dims(model_seg, axis=2)
        # model_seg = np.repeat(model_seg, 3, axis=2)
        model_seg = out_to_rgb(model_seg).astype(np.uint8)

        model_edge = torch.softmax(edge_pre[i], dim=0)
        model_edge = torch.argmax(model_edge, dim=0).long()
        model_edge = model_edge.cpu().detach().numpy()
        model_edge = (model_edge * 255).astype(np.uint8)
        model_edge = np.expand_dims(model_edge, axis=2)
        model_edge = np.repeat(model_edge, 3, axis=2)


        image = np.concatenate([img, label, model_seg, edge_label, model_edge], axis=1)
        image = Image.fromarray(image)
        image.save('predict_out/train_visualization/epoch{}_{}.jpg'.format(epoch, i))



def SemiSupervised_visualization(sup_batch, y_batch, sup_pred, unsup_batch, unsup_pred, unsupAug_batch1,
                                 unsupAug_pred1, unsupAug_batch2, unsupAug_pred2, unsup_y, epoch):
    """
    :param x_batch:
    :param y_batch:
    :param masks_pred:
    :param edge_pre:
    :return:
    """
    sup_batch = sup_batch.cpu().numpy()
    y_batch = y_batch.cpu().numpy()
    unsup_batch = unsup_batch.cpu().numpy()
    unsupAug_batch1 = unsupAug_batch1.cpu().numpy()
    unsupAug_batch2 = unsupAug_batch2.cpu().numpy()
    unsup_y = unsup_y.cpu().numpy()

    for i in range(len(sup_batch)):
        sup_img = (sup_batch[i] * 255).astype(np.uint8)
        sup_img = sup_img.transpose(1, 2, 0)
        label = out_to_rgb((y_batch[i])).astype(np.uint8)
        # label = np.expand_dims(label, axis=2)
        # label = np.repeat(label, 3, axis=2)
        # label = label.transpose(1, 2, 0)


        unsup_label =  out_to_rgb(unsup_y[i]).astype(np.uint8)
        # unsup_label = unsup_label.transpose(1, 2, 0)

        sup_seg = torch.argmax(sup_pred[i], dim=0).long()
        sup_seg = sup_seg.cpu().detach().numpy()
        # model_seg = (model_seg * 255).astype(np.uint8)
        # model_seg = np.expand_dims(model_seg, axis=2)
        # model_seg = np.repeat(model_seg, 3, axis=2)
        sup_seg = out_to_rgb(sup_seg).astype(np.uint8)

        unsup_img = (unsup_batch[i] * 255).astype(np.uint8)
        unsup_img = unsup_img.transpose(1, 2, 0)

        unsup_seg = torch.argmax(unsup_pred[i], dim=0).long()
        unsup_seg = unsup_seg.cpu().detach().numpy()
        unsup_seg = out_to_rgb(unsup_seg).astype(np.uint8)

        unsupAug_img1 = (unsupAug_batch1[i] * 255).astype(np.uint8)
        unsupAug_img1 = unsupAug_img1.transpose(1, 2, 0)

        unsupAug_seg1 = torch.argmax(unsupAug_pred1[i], dim=0).long()
        unsupAug_seg1 = unsupAug_seg1.cpu().detach().numpy()
        unsupAug_seg1 = out_to_rgb(unsupAug_seg1).astype(np.uint8)

        unsupAug_img2 = (unsupAug_batch2[i] * 255).astype(np.uint8)
        unsupAug_img2 = unsupAug_img2.transpose(1, 2, 0)

        unsupAug_seg2 = torch.argmax(unsupAug_pred2[i], dim=0).long()
        unsupAug_seg2 = unsupAug_seg2.cpu().detach().numpy()
        unsupAug_seg2 = out_to_rgb(unsupAug_seg2).astype(np.uint8)


        image = np.concatenate([sup_img, label, sup_seg, unsup_img, unsup_label,
                                unsup_seg, unsupAug_img1, unsupAug_seg1, unsupAug_img2,unsupAug_seg2], axis=1)
        image = Image.fromarray(image)
        image.save('predict_out/semisupervised_visualization/epoch{}_{}.jpg'.format(epoch, i))



def cutmix_visualization(sup_batch, y_batch, sup_pred, unsup_x1, unsup_y1,
                         unsup_pred1, unsup_x2, unsup_y2, unsup_pred2,
                         mixed_image_s1, mask_mix, pre_mix_s1, mixed_image_s2, pre_mix_s2, epoch):
    """
    :param x_batch:
    :param y_batch:
    :param masks_pred:
    :param edge_pre:
    :return:
    """
    sup_batch = sup_batch.cpu().numpy()
    y_batch = y_batch.cpu().numpy()
    unsup_x1 = unsup_x1.cpu().numpy()
    unsup_y1 = unsup_y1.cpu().numpy()
    unsup_x2 = unsup_x2.cpu().numpy()
    unsup_y2 = unsup_y2.cpu().numpy()
    mixed_image_s1 = mixed_image_s1.cpu().numpy()
    mixed_image_s2 = mixed_image_s2.cpu().numpy()

    for i in range(len(sup_batch)):
        sup_img = (sup_batch[i] * 255).astype(np.uint8)
        sup_img = sup_img.transpose(1, 2, 0)
        sup_label = out_to_rgb((y_batch[i])).astype(np.uint8)
        sup_seg = torch.argmax(sup_pred[i], dim=0).long()
        sup_seg = sup_seg.cpu().detach().numpy()
        sup_seg = out_to_rgb(sup_seg).astype(np.uint8)

        unsup_img1 = (unsup_x1[i] * 255).astype(np.uint8)
        unsup_img1 = unsup_img1.transpose(1, 2, 0)
        unsup_label1 = out_to_rgb((unsup_y1[i])).astype(np.uint8)

        unsup_seg1 = torch.argmax(unsup_pred1[i], dim=0).long()
        unsup_seg1 = unsup_seg1.cpu().detach().numpy()
        unsup_seg1 = out_to_rgb(unsup_seg1).astype(np.uint8)

        unsup_img2 = (unsup_x2[i] * 255).astype(np.uint8)
        unsup_img2 = unsup_img2.transpose(1, 2, 0)
        unsup_label2 = out_to_rgb((unsup_y2[i])).astype(np.uint8)

        unsup_seg2 = torch.argmax(unsup_pred2[i], dim=0).long()
        unsup_seg2 = unsup_seg2.cpu().detach().numpy()
        unsup_seg2 = out_to_rgb(unsup_seg2).astype(np.uint8)

        mixed_image1 = (mixed_image_s1[i] * 255).astype(np.uint8)
        mixed_image1 = mixed_image1.transpose(1, 2, 0)

        mask_mix_seg = mask_mix[i].cpu().detach().numpy()
        mask_mix_seg = out_to_rgb(mask_mix_seg).astype(np.uint8)


        mix_seg1 = torch.argmax(pre_mix_s1[i], dim=0).long()
        mix_seg1 = mix_seg1.cpu().detach().numpy()
        mix_seg1 = out_to_rgb(mix_seg1).astype(np.uint8)

        mixed_image2 = (mixed_image_s2[i] * 255).astype(np.uint8)
        mixed_image2 = mixed_image2.transpose(1, 2, 0)

        mix_seg2 = torch.argmax(pre_mix_s2[i], dim=0).long()
        mix_seg2 = mix_seg2.cpu().detach().numpy()
        mix_seg2 = out_to_rgb(mix_seg2).astype(np.uint8)

        image = np.concatenate([sup_img, sup_label, sup_seg, unsup_img1, unsup_label1,
                                unsup_seg1, unsup_img2, unsup_label2, unsup_seg2,
                                mixed_image1, mask_mix_seg, mix_seg1, mixed_image2, mix_seg2], axis=1)
        image = Image.fromarray(image)
        image.save('predict_out/semisupervised_visualization/epoch{}_{}.jpg'.format(epoch, i))










def evluation(net, imgs, labels):
    all_kappa = []
    img_class_f1 = []
    all_overall_accu = []
    all_iou = []
    for i in range(len(imgs)):
        test_img = imgs[i]
        # test_label = labels[i].transpose(2, 0, 1)
        test_label = np.expand_dims(labels[i], axis=0)
        h = test_img.shape[0]
        w = test_img.shape[1]
        x = 0
        y = 0
        size = 1024
        step = 999
        # size = 512
        # step = 512
        net.cuda()
        output_index = torch.zeros(6, h, w)
        while (y <= int(h / step)):
            while (x <= int(w / step)):
                sub_input = test_img[ min(y * step, h - size): min(y * step + size, h),
                            min(x * step, w - size):min(x * step + size, w)]
                mask = predict_img(net=net, full_img=sub_input, use_gpu=True)
                output_index[:, min(y * step, h - size): min(y * step + size, h), min(x * step, w - size):min(x * step + size, w)] += mask.cpu().squeeze()
                x += 1
            x = 0
            y += 1
        predict = torch.argmax(output_index, dim=0).long()
        predict = predict.detach().numpy()
        # index = np.squeeze(rgb_to_labels(test_label))
        index = test_label
        kappa, class_f1, overall_accuracy,iou = Metric(index, predict)
        all_kappa.append(kappa)
        img_class_f1.append(class_f1)
        all_overall_accu.append(overall_accuracy)
        all_iou.append(iou)
    return np.array(all_kappa), np.array(img_class_f1),\
        np.array(all_overall_accu), np.array(all_iou)



def pointrend_evluation(segnet, rendnet, imgs, labels):
    all_kappa = []
    img_class_f1 = []
    all_overall_accu = []
    all_pt_kappa = []
    all_pt_OA = []
    for i in range(len(imgs)):
        test_img = imgs[i]
        test_label = labels[i].transpose(2, 0, 1)
        test_label = np.expand_dims(test_label, axis=0)
        h = test_img.shape[0]
        w = test_img.shape[1]
        x = 0
        y = 0
        size = 1024
        step = 999
        output_index = torch.zeros(6, h, w)
        points_label = []
        points_pre = []
        index = rgb_to_labels(test_label)

        while (y <= int(h / step)):
            while (x <= int(w / step)):

                sub_input = test_img[min(y * step, h - size): min(y * step + size, h),
                            min(x * step, w - size):min(x * step + size, w)]

                sub_label = index[:, min(y * step, h - size): min(y * step + size, h),
                            min(x * step, w - size):min(x * step + size, w)]

                mask = predict_img(net=segnet, full_img=sub_input, use_gpu=True)
                point_out, points, _ = pointrend_predict_img(rendnet, sub_input, mask["refine"], mask["coarse"])

                gt_points = point_sample(
                    torch.from_numpy(sub_label).float().unsqueeze(dim=1),
                    points.cpu(),
                    mode="nearest",
                ).squeeze_(1).long()

                points_pre.append(torch.argmax(point_out, dim=1).squeeze().cpu().numpy())
                points_label.append(gt_points.squeeze().cpu().numpy())

                output_index[:, min(y * step, h - size): min(y * step + size, h), min(x * step, w - size):min(x * step + size, w)] += mask["coarse"].cpu().squeeze()
                x += 1
            x = 0
            y += 1
        predict = torch.argmax(output_index, dim=0).long()
        predict = predict.detach().numpy()

        # print("test points gt: ", points_label)
        # print("test points pre: ", points_pre)
        kappa, class_f1, overall_accuracy,_ = Metric(index, predict)
        pt_kappa, _, pt_OA, _ = Metric(np.array(points_label), np.array(points_pre))

        all_kappa.append(kappa)
        img_class_f1.append(class_f1)
        all_overall_accu.append(overall_accuracy)
        all_pt_kappa.append(pt_kappa)
        all_pt_OA.append(pt_OA)

    return np.array(all_kappa), np.array(img_class_f1), np.array(all_overall_accu), np.array(all_pt_kappa), np.array(all_pt_OA)


def MT_evluation(net, ema_net, imgs, labels):
    all_kappa = []
    img_class_f1 = []
    all_overall_accu = []

    all_kappa_ema = []
    all_overall_accu_ema = []
    net = net.eval()
    ema_net = ema_net.eval()
    for i in range(len(imgs)):
        test_img = imgs[i]
        test_label = labels[i].transpose(2, 0, 1)
        test_label = np.expand_dims(test_label, axis=0)
        h = test_img.shape[0]
        w = test_img.shape[1]
        x = 0
        y = 0
        size = 1024
        step = 999
        # size = 512
        # step = 512

        # for para in net.parameters():
        #     print(para)
        #     break
        #
        # for emapara in ema_net.parameters():
        #     print(emapara)
        #     break


        output_index = torch.zeros(6, h, w)
        output_index_ema = torch.zeros(6, h, w)
        while (y <= int(h / step)):
            while (x <= int(w / step)):
                sub_input = test_img[ min(y * step, h - size): min(y * step + size, h),
                            min(x * step, w - size):min(x * step + size, w)]
                mask_net = predict_img(net=net, full_img=sub_input, use_gpu=True)
                mask_emanet = predict_img(net=ema_net, full_img=sub_input, use_gpu=True)
                output_index[:, min(y * step, h - size): min(y * step + size, h), min(x * step, w - size):min(x * step + size, w)] += mask_net
                output_index_ema[:, min(y * step, h - size): min(y * step + size, h), min(x * step, w - size):min(x * step + size, w)] += mask_emanet
                x += 1
            x = 0
            y += 1
        predict = torch.argmax(output_index, dim=0).long()
        predict = predict.detach().numpy()
        index = np.squeeze(rgb_to_labels(test_label))

        predict_ema = torch.argmax(output_index_ema, dim=0).long()
        predict_ema = predict_ema.detach().numpy()

        kappa, class_f1, overall_accuracy,_ = Metric(index, predict)
        kappa_ema , class_f1_ema , overall_accuracy_ema , _ = Metric(index, predict_ema)

        all_kappa.append(kappa)
        img_class_f1.append(class_f1)
        all_overall_accu.append(overall_accuracy)

        all_kappa_ema.append(kappa_ema)
        all_overall_accu_ema.append(overall_accuracy_ema)


    return np.array(all_kappa), np.array(img_class_f1), np.array(all_overall_accu), np.array(all_kappa_ema), np.array(all_overall_accu_ema)


def Metric(label, predict):
    label = label.reshape(-1)
    predict = predict.reshape(-1)
    p1 = 0
    total_tp = 0
    class_f1 = []
    overall_tp = []
    class_iou = []
    for i in range(6):
        single_predict = (predict == i)
        single_label = (label == i)
        tp = np.sum(np.logical_and(single_predict, single_label))
        union = np.sum(np.logical_or(single_predict, single_label))
        tpfp = np.sum(single_predict)
        tpfn = np.sum(single_label)
        precision = tp / tpfp if tpfp > 0 else 0
        recall = tp / tpfn if tpfn > 0 else 0
        f1 = 2 * (recall * precision) / (recall + precision) if(precision != 0 or recall !=0) else 0
        m = tpfp * tpfn
        p1 += m
        total_tp += tp
        iou_score = tp / union if union > 0 else 0
        class_f1.append(f1)
        overall_tp.append(tp)
        class_iou.append(iou_score)
    class_sum = len(label)
    pe = p1 / pow(class_sum, 2)
    p0 = total_tp / class_sum
    kappa_score = (p0 - pe) / (1 - pe)

    overall_accuracy = np.array(overall_tp).sum() / class_sum
    return kappa_score, np.array(class_f1), overall_accuracy, np.array(class_iou)
