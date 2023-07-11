# -*- coding: utf-8 -*-
# @Time    : 2022/3/17 14:24
# @Author  : wenzhang
# @File    : func_utils.py
import os.path as osp
import os
import numpy as np
import random
import torch as tr
import torch.nn as nn
from scipy.spatial.distance import cdist
from utils.utils import lr_scheduler


def obtain_label_decision(loader, netF, netC):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs, labels = data[0].cuda(), data[1]
            feas = netF(inputs.float())
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = tr.cat((all_fea, feas.float().cpu()), 0)
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)

    all_fea = tr.cat((all_fea, tr.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / tr.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])

    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)

    for round in range(1):  # SSL
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])

    return initc, all_fea


def obtain_label_shot(loader, netF, netC):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netF(inputs)
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = tr.cat((all_fea, feas.float().cpu()), 0)
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    # print(all_output.shape)
    # ent = tr.sum(-all_output * tr.log(all_output + args.epsilon), dim=1)
    # unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = tr.max(all_output, 1)

    accuracy = tr.sum(tr.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    all_fea = tr.cat((all_fea, tr.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / tr.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])

    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > 0)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], 'cosine')
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]
    # acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    # log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    # print(log_str+'\n')

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc[labelset], 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    # print(log_str + '\n')

    return pred_label.astype('int'), dd


def update_decision(dset_loaders, netF_list, netC_list, netG_list, optimizer, info_loss, args):
    max_iter = len(dset_loaders["target"])
    num_src = len(args.src)

    iter_num = 0
    while iter_num < max_iter:
        iter_target = iter(dset_loaders["target"])
        inputs_target, _, tar_idx = iter_target.next()
        if inputs_target.size(0) == 1:
            continue
        inputs_target = inputs_target.cuda()

        # 每10个epoch才进行一次pseudo labels增强
        # 这样改仅从75.54到75.61，变化很小
        interval_iter = 10
        if iter_num % interval_iter == 0 and args.cls_par > 0:
            initc = []
            all_feas = []
            for i in range(num_src):
                netF_list[i].eval()
                temp1, temp2 = obtain_label_decision(dset_loaders['Target'], netF_list[i], netC_list[i])
                temp1 = tr.from_numpy(temp1).cuda()
                temp2 = tr.from_numpy(temp2).cuda()
                initc.append(temp1)
                all_feas.append(temp2)
                netF_list[i].train()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        ###################################################################################
        # output, domain weight, weighted output
        if args.use_weight:
            weights_all = tr.ones(inputs_target.shape[0], len(args.src))
            tmp_output = tr.zeros(len(args.src), inputs_target.shape[0], args.class_num)
            for i in range(len(args.src)):
                tmp_output[i] = netC_list[i](netF_list[i](inputs_target))
                weights_all[:, i] = netG_list[i](tmp_output[i]).squeeze()
            z = tr.sum(weights_all, dim=1) + 1e-16
            weights_all = tr.transpose(tr.transpose(weights_all, 0, 1) / z, 0, 1)
            weights_domain = tr.sum(weights_all, dim=0) / tr.sum(weights_all)
            domain_weight = weights_domain.reshape([1, num_src, 1]).cuda()
        else:
            domain_weight = tr.Tensor([1 / num_src] * num_src).reshape([1, num_src, 1]).cuda()
        weights_domain = np.round(tr.squeeze(domain_weight, 0).t().flatten().cpu().detach(), 3)
        # print(type(domain_weight), type(weights_domain))  # [1, 3, 1], [3]

        outputs_all = tr.cat([netC_list[i](netF_list[i](inputs_target)).unsqueeze(1) for i in range(num_src)], 1).cuda()
        preds = tr.softmax(outputs_all, dim=2)
        outputs_all_w = (preds * domain_weight).sum(dim=1).cuda()
        # print(outputs_all.shape, preds.shape, domain_weight.shape, outputs_all_w.shape)
        # [4, 8, 4], [4, 8, 4], [1, 8, 1], [4, 4]
        ###################################################################################

        # self pseudo label loss
        if args.cls_par > 0:
            initc_ = tr.zeros(initc[0].size()).cuda()
            temp = all_feas[0]
            all_feas_ = tr.zeros(temp[tar_idx, :].size()).cuda()
            for i in range(num_src):
                initc_ = initc_ + weights_domain[i] * initc[i].float()
                src_fea = all_feas[i]
                all_feas_ = all_feas_ + weights_domain[i] * src_fea[tar_idx, :]
            dd = tr.cdist(all_feas_.float(), initc_.float(), p=2)
            pred_label = dd.argmin(dim=1)
            pred = pred_label.int().long()
            clf_loss = nn.CrossEntropyLoss()(outputs_all_w, pred)
        else:
            clf_loss = tr.tensor(0.0).cuda()

        # raw decision
        im_loss = info_loss(outputs_all_w, args.epsilon)
        loss_all = args.cls_par * clf_loss + args.ent_par * im_loss

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()


def update_shot_ens(dset_loaders, netF, netC, optimizer, info_loss, args):
    # 分别对每个源模型和Xt训练，获得迁移之后的模型，然后集成
    max_iter = len(dset_loaders["target"])

    iter_num = 0
    while iter_num < max_iter:
        iter_target = iter(dset_loaders["target"])
        inputs_target, _, tar_idx = iter_target.next()
        if inputs_target.size(0) == 1:
            continue
        inputs_target = inputs_target.cuda()

        # 每10个epoch才进行一次pseudo labels增强
        interval_iter = 10
        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            mem_label, dd = obtain_label_shot(dset_loaders['Target'], netF, netC)
            mem_label = tr.from_numpy(mem_label).cuda()
            netF.train()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        feas_target = netF(inputs_target.cuda())
        outputs_target = netC(feas_target)

        # 构建 shot loss
        if args.cls_par > 0:  # 控制需不需要加自监督loss，默认0.3
            pred = mem_label[tar_idx].long()
            clf_loss = nn.CrossEntropyLoss()(outputs_target, pred)
        else:
            clf_loss = tr.tensor(0.0).cuda()

        # IM loss on the weighted output
        im_loss = info_loss(outputs_target, args.epsilon)
        loss_all = args.cls_par * clf_loss + args.ent_par * im_loss

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()


def knowledge_vote(preds_softmax, confidence_gate, num_classes):
    max_p, max_p_class = preds_softmax.max(2)
    max_conf, _ = max_p.max(1)
    max_p_mask = (max_p > confidence_gate).float().cuda()
    preds_vote = tr.zeros(preds_softmax.size(0), preds_softmax.size(2)).cuda()
    for batch_idx, (p, p_class, p_mask) in enumerate(zip(max_p, max_p_class, max_p_mask)):
        if tr.sum(p_mask) > 0:
            p = p * p_mask
        for source_idx, source_class in enumerate(p_class):
            preds_vote[batch_idx, source_class] += p[source_idx]
    _, preds_vote = preds_vote.max(1)
    preds_vote = tr.zeros(preds_vote.size(0), num_classes).cuda().scatter_(1, preds_vote.view(-1, 1), 1)

    return preds_vote

