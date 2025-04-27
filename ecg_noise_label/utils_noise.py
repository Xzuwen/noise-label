from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.AverageMeter import AverageMeter
from utils.criterion import *
import time
import sys
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


################################# MOIT #############################################

## Input interpolation functions
def mix_data_lab(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    lam = max(lam, 1 - lam)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, index, lam


## Masks creation

## Unsupervised mask for batch and memory (note that memory also contains the current mini-batch)

def unsupervised_masks_estimation(args, xbm, mix_index1, mix_index2, epoch, bsz, device):
    labelsUnsup = torch.arange(bsz).long().unsqueeze(1).to(
        device)  # If no labels used, label is the index in mini-batch
    maskUnsup_batch = torch.eye(bsz, dtype=torch.float32).to(device)
    maskUnsup_batch = maskUnsup_batch.repeat(2, 2)
    maskUnsup_batch[torch.eye(2 * bsz) == 1] = 0  ##remove self-contrast case

    if args.xbm_use == 1 and epoch > args.xbm_begin:
        ## Extend mask to consider xbm_memory features (all zeros except for the last features stored that contain the augmented view in the memory
        maskUnsup_mem = torch.zeros((2 * bsz, xbm.K)).float().to(
            device)  ##Mini-batch samples with memory samples (add columns)

        ##Re-use measkUnsup_batch to copy it in the memory (in the righ place) and find the augmented views (without gradients)

        if xbm.ptr == 0:
            maskUnsup_mem[:, -2 * bsz:] = maskUnsup_batch
        else:
            maskUnsup_mem[:, xbm.ptr - (2 * bsz):xbm.ptr] = maskUnsup_batch

    else:
        maskUnsup_mem = []

    ######################### Mixup additional mask: unsupervised term ######################
    ## With no labels (labelUnsup is just the index in the mini-batch, i.e. different for each sample)
    quad1_unsup = torch.eq(labelsUnsup[mix_index1],
                           labelsUnsup.t()).float()  ##Minor label in 1st mini-batch part equal to mayor label in the first mini-batch part (note that mayor label of 1st and 2nd is the same as we force the original image to always be the dominant)
    quad2_unsup = torch.eq(labelsUnsup[mix_index1],
                           labelsUnsup.t()).float()  ##Minor label in 1st mini-batch part equal to mayor label in the second mini-batch part
    quad3_unsup = torch.eq(labelsUnsup[mix_index2],
                           labelsUnsup.t()).float()  ##Minor label in 2nd mini-batch part equal to mayor label in the first mini-batch part
    quad4_unsup = torch.eq(labelsUnsup[mix_index2],
                           labelsUnsup.t()).float()  ##Minor label in 2nd mini-batch part equal to mayor label in the second mini-batch part

    mask2_a_unsup = torch.cat((quad1_unsup, quad2_unsup), dim=1)
    mask2_b_unsup = torch.cat((quad3_unsup, quad4_unsup), dim=1)
    mask2Unsup_batch = torch.cat((mask2_a_unsup, mask2_b_unsup), dim=0)

    ## Make sure diagonal is zero (i.e. not taking as positive my own sample)
    mask2Unsup_batch[torch.eye(2 * bsz) == 1] = 0

    if args.xbm_use == 1 and epoch > args.xbm_begin:
        ## Extend mask to consider xbm_memory features (will be zeros excpet the positions for the augmented views for the second mixup term)
        mask2Unsup_mem = torch.zeros((2 * bsz, xbm.K)).float().to(
            device)  ##Mini-batch samples with memory samples (add columns)

        if xbm.ptr == 0:
            mask2Unsup_mem[:, -2 * bsz:] = mask2Unsup_batch
        else:
            mask2Unsup_mem[:, xbm.ptr - (2 * bsz):xbm.ptr] = mask2Unsup_batch

    else:
        mask2Unsup_mem = []

    return maskUnsup_batch, maskUnsup_mem, mask2Unsup_batch, mask2Unsup_mem


def supervised_masks_estimation(args, labels, xbm, xbm_labels, mix_index1, mix_index2, epoch, bsz, device):
    ###################### Supervised mask excluding augmented view ###############################
    labels = labels.contiguous().view(-1, 1)

    if labels.shape[0] != bsz:
        raise ValueError('Num of labels does not match num of features')

    ##Create mask without diagonal to avoid augmented view, i.e. this is supervised mask
    maskSup_batch = torch.eq(labels, labels.t()).float() - torch.eye(bsz, dtype=torch.float32).to(device)
    maskSup_batch = maskSup_batch.repeat(2, 2)
    maskSup_batch[torch.eye(2 * bsz) == 1] = 0  ##remove self-contrast case

    if args.xbm_use == 1 and epoch > args.xbm_begin:
        ## Extend mask to consider xbm_memory features
        maskSup_mem = torch.eq(labels, xbm_labels.t()).float().repeat(2,
                                                                      1)  ##Mini-batch samples with memory samples (add columns)

        if xbm.ptr == 0:
            maskSup_mem[:, -2 * bsz:] = maskSup_batch
        else:
            maskSup_mem[:, xbm.ptr - (2 * bsz):xbm.ptr] = maskSup_batch

    else:
        maskSup_mem = []

    ######################### Mixup additional mask: supervised term ######################
    ## With labels
    quad1_sup = torch.eq(labels[mix_index1],
                         labels.t()).float()  ##Minor label in 1st mini-batch part equal to mayor label in the first mini-batch part (note that mayor label of 1st and 2nd is the same as we force the original image to always be the mayor/dominant)
    quad2_sup = torch.eq(labels[mix_index1],
                         labels.t()).float()  ##Minor label in 1st mini-batch part equal to mayor label in the second mini-batch part
    quad3_sup = torch.eq(labels[mix_index2],
                         labels.t()).float()  ##Minor label in 2nd mini-batch part equal to mayor label in the first mini-batch part
    quad4_sup = torch.eq(labels[mix_index2],
                         labels.t()).float()  ##Minor label in 2nd mini-batch part equal to mayor label in the second mini-batch part

    mask2_a_sup = torch.cat((quad1_sup, quad2_sup), dim=1)
    mask2_b_sup = torch.cat((quad3_sup, quad4_sup), dim=1)
    mask2Sup_batch = torch.cat((mask2_a_sup, mask2_b_sup), dim=0)

    ## Make sure diagonal is zero (i.e. not taking as positive my own sample)
    mask2Sup_batch[torch.eye(2 * bsz) == 1] = 0

    if args.xbm_use == 1 and epoch > args.xbm_begin:
        ## Extend mask to consider xbm_memory features. Here we consider that the label for images is the minor one, i.e. labels[mix_index1], labels[mix_index2] and xbm_labels_mix
        ## Here we don't repeat the columns part as in maskSup because the minor label is different for the first and second part of the mini-batch (different mixup shuffling for each mini-batch part)
        maskExtended_sup3_1 = torch.eq(labels[mix_index1],
                                       xbm_labels.t()).float()  ##Mini-batch samples with memory samples (add columns)
        maskExtended_sup3_2 = torch.eq(labels[mix_index2],
                                       xbm_labels.t()).float()  ##Mini-batch samples with memory samples (add columns)
        mask2Sup_mem = torch.cat((maskExtended_sup3_1, maskExtended_sup3_2), dim=0)

        if xbm.ptr == 0:
            mask2Sup_mem[:, -2 * bsz:] = mask2Sup_batch

        else:
            mask2Sup_mem[:, xbm.ptr - (2 * bsz):xbm.ptr] = mask2Sup_batch

    else:
        mask2Sup_mem = []

    return maskSup_batch, maskSup_mem, mask2Sup_batch, mask2Sup_mem


#### Losses

def InterpolatedContrastiveLearning_loss(args, pairwise_comp, maskSup, mask2Sup, maskUnsup, mask2Unsup, logits_mask,
                                         lam1, lam2, bsz, epoch, device):
    logits = torch.div(pairwise_comp, args.batch_t)

    exp_logits = torch.exp(logits) * logits_mask  # remove diagonal

    if args.aprox == 1:  # 相当于求log-softmax,得到log_prob
        log_prob = logits - torch.log(exp_logits.sum(1,
                                                     keepdim=True))  ## Approximation for numerical stability taken from supervised contrastive learning
    else:
        log_prob = torch.log(torch.exp(logits) + 1e-10) - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

    exp_logits2 = torch.exp(logits) * logits_mask  # remove diagonal

    if args.aprox == 1:
        log_prob2 = logits - torch.log(exp_logits2.sum(1,
                                                       keepdim=True))  ## Approximation for numerical stability taken from supervised contrastive learning
    else:
        log_prob2 = torch.log(torch.exp(logits) + 1e-10) - torch.log(exp_logits2.sum(1, keepdim=True) + 1e-10)

    # compute mean of log-likelihood over positive (weight individual loss terms with mixing coefficients)

    mean_log_prob_pos_sup = (maskSup * log_prob).sum(1) / (maskSup.sum(1) + maskUnsup.sum(1))  # 256
    mean_log_prob_pos_unsup = (maskUnsup * log_prob).sum(1) / (maskSup.sum(1) + maskUnsup.sum(1))  # 256
    ## Second mixup term log-probs 指每个扩充样本的第二个mixup项，而非第二个扩充样本
    mean_log_prob_pos2_sup = (mask2Sup * log_prob2).sum(1) / (mask2Sup.sum(1) + mask2Unsup.sum(1))  # 256
    mean_log_prob_pos2_unsup = (mask2Unsup * log_prob2).sum(1) / (mask2Sup.sum(1) + mask2Unsup.sum(1))  # 256
    ## Weight first and second mixup term (both data views) with the corresponding mixing weight

    ## First mixup term. First mini-batch part. Unsupervised + supervised loss separated
    loss1a = -lam1 * mean_log_prob_pos_unsup[:int(len(mean_log_prob_pos_unsup) / 2)] - lam1 * mean_log_prob_pos_sup[
                                                                                              :int(
                                                                                                  len(mean_log_prob_pos_sup) / 2)]
    ## First mixup term. Second mini-batch part. Unsupervised + supervised loss separated
    loss1b = -lam2 * mean_log_prob_pos_unsup[int(len(mean_log_prob_pos_unsup) / 2):] - lam2 * mean_log_prob_pos_sup[
                                                                                              int(len(
                                                                                                  mean_log_prob_pos_sup) / 2):]
    ## All losses for first mixup term
    loss1 = torch.cat((loss1a, loss1b))
    ## Second mixup term. First mini-batch part. Unsupervised + supervised loss separated
    loss2a = -(1.0 - lam1) * mean_log_prob_pos2_unsup[:int(len(mean_log_prob_pos2_unsup) / 2)] - (
                1.0 - lam1) * mean_log_prob_pos2_sup[:int(len(mean_log_prob_pos2_sup) / 2)]
    ## Second mixup term. Second mini-batch part. Unsupervised + supervised loss separated
    loss2b = -(1.0 - lam2) * mean_log_prob_pos2_unsup[int(len(mean_log_prob_pos2_unsup) / 2):] - (
                1.0 - lam2) * mean_log_prob_pos2_sup[int(len(mean_log_prob_pos2_sup) / 2):]
    ## All losses secondfor first mixup term
    loss2 = torch.cat((loss2a, loss2b))  # 128+128=256

    ## Final loss (summation of mixup terms after weighting)
    loss = loss1 + loss2

    loss = loss.view(2, bsz).mean()

    return loss


## Semi-supervised learning

def ClassificationLoss(args, predsA, predsB, predsNoDA, y_a1, y_b1, y_a2, y_b2, mix_index1, mix_index2, lam1, lam2,
                       criterionCE, agreement_measure, epoch, device):
    preds = torch.cat((predsA, predsB), dim=0)

    targets_1 = torch.cat((y_a1, y_a2), dim=0)
    targets_2 = torch.cat((y_b1, y_b2), dim=0)
    mix_index = torch.cat((mix_index1, mix_index2), dim=0)

    ones_vec = torch.ones((predsA.size()[0],)).float().to(device)
    lam_vec = torch.cat((lam1 * ones_vec, lam2 * ones_vec), dim=0).to(device)

    if args.PredictiveCorrection == 0 or epoch <= args.startLabelCorrection:
        loss = lam_vec * criterionCE(preds, targets_1) + (1 - lam_vec) * criterionCE(preds, targets_2)
        loss = loss.mean()


    elif args.PredictiveCorrection == 1 and epoch > args.startLabelCorrection:
        agreement_measure = torch.cat((agreement_measure, agreement_measure), dim=0).to(device)
        lossLabeled = agreement_measure * (
                lam_vec * criterionCE(preds, targets_1) + (1 - lam_vec) * criterionCE(preds, targets_2))
        lossLabeled = lossLabeled.mean()

        ## Pseudo-labeling
        prob = F.softmax(predsNoDA, dim=1)
        prob = torch.cat((prob, prob), dim=0)
        z1 = prob.clone().detach()
        z2 = z1[mix_index, :]
        preds_logSoft = F.log_softmax(preds)

        loss_x1_pred_vec = lam_vec * (1 - agreement_measure) * (-torch.sum(z1 * preds_logSoft, dim=1))  ##Soft
        loss_x1_pred = torch.sum(loss_x1_pred_vec) / len(loss_x1_pred_vec)

        loss_x2_pred_vec = (1 - lam_vec) * (1 - agreement_measure[mix_index]) * (
            -torch.sum(z2 * preds_logSoft, dim=1))  ##Soft
        loss_x2_pred = torch.sum(loss_x2_pred_vec) / len(loss_x2_pred_vec)

        lossUnlabeled = loss_x1_pred + loss_x2_pred

        loss = lossLabeled + lossUnlabeled

    return loss


def train_MOIT(args, model, device, train_loader, optimizer, epoch, xbm, agreement, result_path, noise_rate):
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    counter = 1

    criterionCE = torch.nn.CrossEntropyLoss(reduction="none")

    sample_features = torch.zeros((len(train_loader.dataset), 128), dtype=torch.float32).to(device)

    for batch_idx, (img1, img2, img_noDA, labels, _, index, _, clean_labels) in enumerate(train_loader):

        if epoch > 1:
            agreement_measure = agreement[index]
        else:
            agreement_measure = []
        img1, img2, img_noDA, labels, index = img1.to(device), img2.to(device), img_noDA.to(device), labels.to(
            device), index.to(device)

        ##Interpolated inputs
        img1, y_a1, y_b1, mix_index1, lam1 = mix_data_lab(img1, labels, args.alpha, device)
        img2, y_a2, y_b2, mix_index2, lam2 = mix_data_lab(img2, labels, args.alpha, device)

        bsz = img1.shape[0]

        ## Set grads to 0.
        model.zero_grad()

        ## Backbone forward pass

        predsA, embedA = model(img1)
        predsB, embedB = model(img2)

        ## Forward pass free of DA
        predsNoDA, noDA_features = model(img_noDA)

        ## Compute classification loss (returned individual per-sample loss)

        ## Remove this preds from graph
        predsNoDA = predsNoDA.detach()

        lossClassif = ClassificationLoss(args, predsA, predsB, predsNoDA, y_a1, y_b1, y_a2, y_b2, mix_index1,
                                         mix_index2, lam1, lam2, criterionCE, agreement_measure, epoch, device)

        ############# Update memory ##############
        if args.xbm_use == 1:
            xbm.enqueue_dequeue(torch.cat((embedA.detach(), embedB.detach()), dim=0),
                                torch.cat((labels.detach().squeeze(), labels.detach().squeeze()), dim=0))

        ############# Get features from memory ##############
        if args.xbm_use == 1 and epoch > args.xbm_begin:
            xbm_feats, xbm_labels = xbm.get()
            xbm_labels = xbm_labels.unsqueeze(1)
        else:
            xbm_feats, xbm_labels = [], []
        #####################################################

        ###################### Unsupervised mask with augmented view ###############################
        maskUnsup_batch, maskUnsup_mem, mask2Unsup_batch, mask2Unsup_mem = unsupervised_masks_estimation(args, xbm,
                                                                                                         mix_index1,
                                                                                                         mix_index2,
                                                                                                         epoch, bsz,
                                                                                                         device)
        ############################################################################################

        ## Contrastive learning
        embeds_batch = torch.cat([embedA, embedB], dim=0)
        pairwise_comp_batch = torch.matmul(embeds_batch, embeds_batch.t())

        if args.xbm_use == 1 and epoch > args.xbm_begin:
            embeds_mem = torch.cat([embedA, embedB, xbm_feats], dim=0)
            pairwise_comp_mem = torch.matmul(embeds_mem[:2 * bsz],
                                             embeds_mem[2 * bsz:].t())  ##Compare mini-batch with memory
            ######################################################################

        ###################### Supervised mask excluding augmented view ###############################
        maskSup_batch, maskSup_mem, mask2Sup_batch, mask2Sup_mem = \
            supervised_masks_estimation(args, labels, xbm, xbm_labels, mix_index1, mix_index2, epoch, bsz, device)
        ############################################################################################

        # Mask-out self-contrast cases
        logits_mask_batch = (torch.ones_like(maskSup_batch) - torch.eye(2 * bsz).to(
            device))  ## Negatives mask, i.e. all except self-contrast sample

        loss = InterpolatedContrastiveLearning_loss(args, pairwise_comp_batch, maskSup_batch, mask2Sup_batch,
                                                    maskUnsup_batch, mask2Unsup_batch, logits_mask_batch, lam1, lam2,
                                                    bsz, epoch, device)

        if args.xbm_use == 1 and epoch > args.xbm_begin:

            logits_mask_mem = torch.ones_like(maskSup_mem)  ## Negatives mask, i.e. all except self-contrast sample

            if xbm.ptr == 0:
                logits_mask_mem[:, -2 * bsz:] = logits_mask_batch
            else:
                logits_mask_mem[:, xbm.ptr - (2 * bsz):xbm.ptr] = logits_mask_batch

            loss_mem = InterpolatedContrastiveLearning_loss(args, pairwise_comp_mem, maskSup_mem, mask2Sup_mem,
                                                            maskUnsup_mem, mask2Unsup_mem, logits_mask_mem, lam1, lam2,
                                                            bsz, epoch, device)

            loss = loss + loss_mem

        loss = loss + lossClassif

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy_v2(predsNoDA, labels, top=[1, 5])
        top1.update(prec1.item(), img1.size(0))
        top5.update(prec5.item(), img1.size(0))
        train_loss.update(loss.item(), img1.size(0))

        sample_features[index] = noDA_features.detach()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if counter % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Learning rate: {:.6f}'.format(
                epoch, counter * len(img1), len(train_loader.dataset),
                       100. * counter / len(train_loader), loss.item(),
                optimizer.param_groups[0]['lr']))
        counter = counter + 1

    torch.save(sample_features, result_path + f'/noise_rate_{noise_rate}/features/sample_features_epoch{epoch}.pth')

    return train_loss.avg, top1.avg, top5.avg, batch_time.sum


##############################################################################################

############################### MOIT + ###############################

def criterionMixBoot(args, preds, predsNoDA, targets_1, targets_2, mix_index, lam, criterionCE, epoch, device):
    lam_vec = lam * torch.ones((preds.size()[0],)).float().to(device)

    if args.PredictiveCorrection == 0 or epoch <= args.startLabelCorrection:
        loss = lam_vec * criterionCE(preds, targets_1) + (1 - lam_vec) * criterionCE(preds, targets_2)

    elif args.PredictiveCorrection == 1 and epoch > args.startLabelCorrection:

        ## Hard boot
        output_x1 = F.log_softmax(predsNoDA, dim=1)
        output_x2 = output_x1[mix_index, :]
        z1 = torch.max(output_x1, dim=1)[1]
        z2 = torch.max(output_x2, dim=1)[1]

        B = 0.2

        loss_x1_vec = lam_vec * (1 - B) * criterionCE(preds, targets_1)
        loss_x1 = torch.sum(loss_x1_vec) / len(loss_x1_vec)

        loss_x1_pred_vec = lam_vec * B * criterionCE(preds, z1)
        loss_x1_pred = torch.sum(loss_x1_pred_vec) / len(loss_x1_pred_vec)

        loss_x2_vec = (1 - lam_vec) * (1 - B) * criterionCE(preds, targets_2)
        loss_x2 = torch.sum(loss_x2_vec) / len(loss_x2_vec)

        loss_x2_pred_vec = (1 - lam_vec) * B * criterionCE(preds, z2)
        loss_x2_pred = torch.sum(loss_x2_pred_vec) / len(loss_x2_pred_vec)

        loss = loss_x1 + loss_x1_pred + loss_x2 + loss_x2_pred

    return loss


def train_mixupBoot(args, model, device, train_loader, optimizer, epoch):
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    counter = 1

    criterionCE = torch.nn.CrossEntropyLoss(reduction="none")

    for batch_idx, (img1, _, img_noDA, labels, _, index, _, _) in enumerate(train_loader):

        img1, img_noDA, labels, index = img1.to(device), img_noDA.to(device), labels.to(device), index.to(device)

        img1, y_a1, y_b1, mix_index1, lam1 = mix_data_lab(img1, labels, args.alpha, device)

        model.zero_grad()

        predsA, _ = model(img1)

        ## Forward pass free of DA
        predsNoDA, _ = model(img_noDA)
        ## Remove this preds from graph
        predsNoDA = predsNoDA.detach()

        ## Compute classification loss (returned individual per-sample loss)
        lossClassif = criterionMixBoot(args, predsA, predsNoDA, y_a1, y_b1, mix_index1, lam1, criterionCE, epoch,
                                       device)

        ## Average loss after saving it per-sample
        loss = lossClassif.mean()

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy_v2(predsNoDA, labels, top=[1, 5])
        train_loss.update(loss.item(), img1.size(0))
        top1.update(prec1.item(), img1.size(0))
        top5.update(prec5.item(), img1.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if counter % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Learning rate: {:.6f}'.format(
                epoch, counter * len(img1), len(train_loader.dataset),
                       100. * counter / len(train_loader), loss.item(),
                optimizer.param_groups[0]['lr']))
        counter = counter + 1

    return train_loss.avg, top1.avg, top5.avg, batch_time.sum


###############################################################################################


#################################### EVALUATION ###############################################


def test_eval(args, model, device, test_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch = []
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            if args.method == "MOIT":
                output, _ = model(data)
            else:
                output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx + 1) * args.test_batch_size))

    test_loss /= len(test_loader.dataset)
    print('\nTest set prediction branch: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    loss_per_epoch = np.average(loss_per_batch)
    acc_val_per_epoch = np.array(100. * correct / len(test_loader.dataset))

    return (loss_per_epoch, acc_val_per_epoch)


'''
def LabelNoiseDetection(args, net, device, trainloader, testloader, sigma, epoch, result_path, noise_rate):

    net.eval()

    cls_time = AverageMeter()
    end = time.time()

    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).to(device)
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.targets).to(device)

    C = trainLabels.max() + 1

    ## Get train features
    transform_bak = trainloader.dataset.transform
    trainloader.dataset.transform = testloader.dataset.transform
    temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=0)
    trainFeatures = torch.load(result_path + f'/noise_rate_{noise_rate}/features/sample_features_epoch{epoch}.pth').to(
        device)
    trainFeatures = trainFeatures.t()
    # trainFeatures = torch.rand(len(trainloader.dataset), args.low_dim).t().to(device)
    # for batch_idx, (inputs, _, _, noisyLabels, _, index, _, targets) in enumerate(temploader):
    #     inputs = inputs.to(device)
    #     batchSize = inputs.size(0)
    #
    #     if args.method == "MOIT":
    #         _, features = net(inputs)
    #     else:
    #         features = net(inputs)
    #
    #     trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = features.data.t()

    trainLabels = torch.LongTensor(temploader.dataset.clean_labels).to(device)
    trainNoisyLabels = torch.LongTensor(temploader.dataset.targets).to(device)
    train_new_labels = torch.LongTensor(temploader.dataset.targets).to(device)


    discrepancy_measure1 = torch.zeros((len(temploader.dataset.targets),)).to(device)
    discrepancy_measure2 = torch.zeros((len(temploader.dataset.targets),)).to(device)
    agreement_measure = torch.zeros((len(temploader.dataset.targets),))#.to(device)

    ## Weighted k-nn correction

    with torch.no_grad():
        retrieval_one_hot_train = torch.zeros(args.k_val, C).to(device)

        for batch_idx, (inputs, _, _, targets, _, index, _, _) in enumerate(temploader):
            targets = targets.to(device)
            batchSize = inputs.size(0)

            features = trainFeatures.t()[index]

            dist = torch.mm(features, trainFeatures)
            dist[torch.arange(dist.size()[0]), index] = -1 ##Self-contrast set to -1

            yd, yi = dist.topk(args.k_val, dim=1, largest=True, sorted=True) ## Top-K similar scores and corresponding indexes
            candidates = trainNoisyLabels.view(1, -1).expand(batchSize, -1) ##Replicate the labels per row to select
            retrieval = torch.gather(candidates, 1, yi) ## From replicated labels get those of the top-K neighbours using the index yi (from top-k operation)

            retrieval_one_hot_train.resize_(batchSize * args.k_val, C).zero_()
            ## Generate the K*batchSize one-hot encodings from neighboring labels ("retrieval"), i.e. each row in retrieval
            # (set of neighbouring labels) is turned into a one-hot encoding
            retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_() ## Apply temperature to scores
            yd_transform[...] = 1.0 ##To avoid using similarities
            probs_corrected = torch.sum(torch.mul(retrieval_one_hot_train.view(batchSize, -1, C), yd_transform.view(batchSize, -1, 1)), 1)
            probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]

            prob_temp = probs_norm[torch.arange(0, batchSize), targets]
            prob_temp[prob_temp <= 1e-2] = 1e-2
            prob_temp[prob_temp > (1 - 1e-2)] = 1 - 1e-2
            discrepancy_measure1[index] = -torch.log(prob_temp)

            if args.discrepancy_corrected == 0:
                agreement_measure[index.data.cpu()] = (torch.max(probs_norm, dim=1)[1]==targets).float().data.cpu()


            _, predictions_corrected = probs_corrected.sort(1, True)
            new_labels = predictions_corrected[:, 0]

            train_new_labels[index] = new_labels

            cls_time.update(time.time() - end)


    tran_new_labels2 = train_new_labels.clone()
    with torch.no_grad():
        retrieval_one_hot_train = torch.zeros(args.k_val, C).to(device)

        for batch_idx, (inputs, _, _, targets, _, index, _, clean_labels) in enumerate(temploader):

            targets = targets.to(device)
            batchSize = inputs.size(0)

            features = trainFeatures.t()[index]

            dist = torch.mm(features, trainFeatures)
            dist[torch.arange(dist.size()[0]), index] = -1  ##Self-contrast set to -1

            yd, yi = dist.topk(args.k_val, dim=1, largest=True, sorted=True)  ## Top-K similar scores and corresponding indexes
            candidates = tran_new_labels2.view(1, -1).expand(batchSize, -1)  ##Replicate the labels per row to select
            retrieval = torch.gather(candidates, 1, yi)  ## From replicated labels get those of the top-K neighbours using the index yi (from top-k operation)

            retrieval_one_hot_train.resize_(batchSize * args.k_val, C).zero_()
            ## Generate the K*batchSize one-hot encodings from neighboring labels ("retrieval"), i.e. each row in retrieval
            # (set of neighbouring labels) is turned into a one-hot encoding
            retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()  ## Apply temperature to scores
            yd_transform[...] = 1.0  ##To avoid using similarities only counts
            probs_corrected = torch.sum(
                torch.mul(retrieval_one_hot_train.view(batchSize, -1, C), yd_transform.view(batchSize, -1, 1)), 1)

            probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]


            prob_temp = probs_norm[torch.arange(0, batchSize), targets]
            prob_temp[prob_temp<=1e-2] = 1e-2
            prob_temp[prob_temp > (1-1e-2)] = 1-1e-2

            discrepancy_measure2[index] = -torch.log(prob_temp)

            if args.discrepancy_corrected == 1:
                agreement_measure[index.data.cpu()] = (torch.max(probs_norm, dim=1)[1]==targets).float().data.cpu()


            cls_time.update(time.time() - end)


    #### Set balanced criterion for noise detection
    if args.balance_crit == "max" or args.balance_crit =="min" or args.balance_crit =="median":
        #agreement_measure_balanced = torch.zeros((len(temploader.dataset.targets),)).to(device)
        num_clean_per_class = torch.zeros(args.num_classes)
        for i in range(args.num_classes):
            idx_class = temploader.dataset.targets==i
            idx_class = torch.from_numpy(idx_class.astype("float")) == 1.0
            num_clean_per_class[i] = torch.sum(agreement_measure[idx_class])

        if args.balance_crit =="max":
            num_samples2select_class = torch.max(num_clean_per_class)
        elif args.balance_crit =="min":
            num_samples2select_class = torch.min(num_clean_per_class)
        elif args.balance_crit =="median":
            num_samples2select_class = torch.median(num_clean_per_class)

        agreement_measure = torch.zeros((len(temploader.dataset.targets),)).to(device)

        for i in range(args.num_classes):
            idx_class = temploader.dataset.targets==i
            samplesPerClass = idx_class.sum()
            idx_class = torch.from_numpy(idx_class.astype("float"))# == 1.0
            idx_class = (idx_class==1.0).nonzero().squeeze()
            if args.discrepancy_corrected == 0:
                discrepancy_class = discrepancy_measure1[idx_class]
            else:
                discrepancy_class = discrepancy_measure2[idx_class]

            if num_samples2select_class>=samplesPerClass:
                k_corrected = samplesPerClass
            else:
                k_corrected = num_samples2select_class

            top_clean_class_relative_idx = torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=False)[1]

            ##Agreement measure sets to 1 those samples detected as clean
            agreement_measure[idx_class[top_clean_class_relative_idx]] = 1.0

    trainloader.dataset.transform = transform_bak

    clean_idx = (temploader.dataset.noisy_labels == temploader.dataset.clean_labels).astype(float)

    right_num = torch.sum((torch.from_numpy(clean_idx) == 1.0) & (agreement_measure == 1.0))
    wrong_num = torch.sum((torch.from_numpy(clean_idx) != 1.0) & (agreement_measure == 1.0))

    precision = right_num / torch.sum(agreement_measure == 1.0) * 100.0
    error_rate = wrong_num / torch.sum(agreement_measure == 1.0) * 100.0
    acc = right_num / np.sum(clean_idx == 1.0) * 100.0

    print('precision: %.3f%% (%d/%d)' % (precision, right_num, torch.sum(agreement_measure == 1.0)))
    print('error_rate: %.3f%% (%d/%d)' % (error_rate, wrong_num, torch.sum(agreement_measure == 1.0)))
    print('acc: %.3f%% (%d/%d)' % (acc, right_num, np.sum(clean_idx == 1.0)))

    return discrepancy_measure1, discrepancy_measure2, agreement_measure, precision, error_rate, acc
'''


def Label_Detection(ep, net, noise_rate, k_nn, trainloader, sigma, result_path, device):
    net.eval()
    cls_time = AverageMeter()
    end_time = time.time()

    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).to(device)
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.noisy_labels).to(device)

    C = trainLabels.max() + 1  # 类别数目

    ## Get train features
    temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=128, shuffle=False)

    trainFeatures = torch.load(result_path + f'/noise_rate_{noise_rate}/features/sample_features_epoch{ep}.pth').to(
        device)

    trainNoisyLabels = torch.LongTensor(temploader.dataset.noisy_labels).to(device)
    train_new_labels = torch.LongTensor(temploader.dataset.noisy_labels).to(device)

    agreement_measure = torch.zeros((len(temploader.dataset.noisy_labels),))  #.to(device)

    ## Weighted k-nn correction
    batch_size = 128
    batch_num = (trainFeatures.shape[0] + batch_size - 1) // batch_size

    with torch.no_grad():
        retrieval_one_hot_train = torch.zeros(k_nn, C).to(device)
        for i in range(batch_num):
            start = i * batch_size
            end = min((i + 1) * batch_size, trainFeatures.shape[0])

            feature_batch = trainFeatures[start:end]
            norm_batch = torch.norm(feature_batch, p=2, dim=1, keepdim=True)
            norms = torch.norm(trainFeatures, p=2, dim=1, keepdim=True)
            dot_product = torch.mm(feature_batch, trainFeatures.T)
            sim_batch = dot_product / (norm_batch * norms.T)

            targets_batch = trainNoisyLabels[start:end]
            batchSize = sim_batch.shape[0]

            yd, yi = sim_batch.topk(k_nn, dim=1, largest=True,
                                    sorted=True)  # Top-K similar scores and corresponding indexes

            candidates = trainNoisyLabels.view(1, -1).expand(batchSize, -1)  ##Replicate the labels per row to select

            retrieval = torch.gather(candidates, 1, yi)  ## 从candidates中选取元素，沿着dim=1,索引为yi

            # 获取knn的标签

            ## 获取knn的标签的one-hot编码
            retrieval_one_hot_train.resize_(batchSize * k_nn, C).zero_()
            retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)  # 将1填入 retrieval_one_hot_train 张量的指定位置

            yd_transform = yd.clone().div_(sigma).exp_()  ## Apply temperature to scores
            yd_transform[...] = 1.0  ##To avoid using similarities
            ## 获得修正后的软标签
            probs_corrected_soft = torch.sum(retrieval_one_hot_train.view(batchSize, -1, C), 1)
            probs_soft_norm = probs_corrected_soft / torch.sum(probs_corrected_soft, dim=1)[:, None]  # 将软标签归一化

            _, predictions_corrected = probs_corrected_soft.sort(dim=1, descending=True)
            new_labels = predictions_corrected[:, 0]

            agreement_measure[start:end] = (torch.max(probs_soft_norm, dim=1)[1] == targets_batch).float().data.cpu()

            train_new_labels[start:end] = new_labels

            cls_time.update(time.time() - end_time)

    clean_idx = (temploader.dataset.noisy_labels == temploader.dataset.clean_labels).astype(float)

    right_num = torch.sum((torch.from_numpy(clean_idx) == 1.0) & (agreement_measure == 1.0))
    wrong_num = torch.sum((torch.from_numpy(clean_idx) != 1.0) & (agreement_measure == 1.0))

    precision = right_num / torch.sum(agreement_measure == 1.0) * 100.0
    error_rate = wrong_num / torch.sum(agreement_measure == 1.0) * 100.0
    acc = right_num / np.sum(clean_idx == 1.0) * 100.0

    print('precision: %.3f%% (%d/%d)' % (precision, right_num, torch.sum(agreement_measure == 1.0)))
    print('error_rate: %.3f%% (%d/%d)' % (error_rate, wrong_num, torch.sum(agreement_measure == 1.0)))
    print('acc: %.3f%% (%d/%d)' % (acc, right_num, np.sum(clean_idx == 1.0)))

    return agreement_measure, precision, error_rate, acc


def LabelNoiseDetection(ep, net, noise_rate, k_nn, trainloader, discrepancy_corrected, balance_crit, sigma, result_path,
                        device):
    net.eval()
    cls_time = AverageMeter()
    end_time = time.time()

    trainLabels = torch.LongTensor(trainloader.dataset.noisy_labels).to(device)

    C = trainLabels.max() + 1  # 类别数目

    ## Get train features
    temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False)

    trainFeatures = torch.load(result_path + f'/noise_rate_{noise_rate}/features/sample_features_epoch{ep}.pth').to(
        device)

    trainNoisyLabels = torch.LongTensor(temploader.dataset.noisy_labels).to(device)
    train_new_labels = torch.LongTensor(temploader.dataset.noisy_labels).to(device)

    discrepancy_measure1 = torch.zeros((len(temploader.dataset.noisy_labels),)).to(device)
    discrepancy_measure2 = torch.zeros((len(temploader.dataset.noisy_labels),)).to(device)
    agreement_measure = torch.zeros((len(temploader.dataset.noisy_labels),))  #.to(device)

    ## Weighted k-nn correction
    batch_size = 128
    batch_num = (trainFeatures.shape[0] + batch_size - 1) // batch_size

    with torch.no_grad():
        retrieval_one_hot_train = torch.zeros(k_nn, C).to(device)
        for i in range(batch_num):
            start = i * batch_size
            end = min((i + 1) * batch_size, trainFeatures.shape[0])

            feature_batch = trainFeatures[start:end]
            sim_batch = torch.mm(feature_batch, trainFeatures.T)

            targets_batch = trainNoisyLabels[start:end]
            batchSize = sim_batch.shape[0]

            yd, yi = sim_batch.topk(k_nn, dim=1, largest=True,
                                    sorted=True)  ## Top-K similar scores and corresponding indexes
            candidates = trainNoisyLabels.view(1, -1).expand(batchSize, -1)  ##Replicate the labels per row to select
            # 获取knn的标签
            retrieval = torch.gather(candidates, 1, yi)  ## 从candidates中选取元素，沿着dim=1,索引为yi
            ## 获取knn的标签的one-hot编码
            retrieval_one_hot_train.resize_(batchSize * k_nn, C).zero_()
            retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)  # 将1填入 retrieval_one_hot_train 张量的指定位置

            yd_transform = yd.clone().div_(sigma).exp_()  ## Apply temperature to scores
            yd_transform[...] = 1.0  ##To avoid using similarities
            ## 获得修正后的软标签
            probs_corrected_soft = torch.sum(
                torch.mul(retrieval_one_hot_train.view(batchSize, -1, C), yd_transform.view(batchSize, -1, 1)), 1)
            probs_soft_norm = probs_corrected_soft / torch.sum(probs_corrected_soft, dim=1)[:, None]  # 将软标签归一化

            _, predictions_corrected = probs_corrected_soft.sort(dim=1, descending=True)
            new_labels = predictions_corrected[:, 0]

            prob_temp = probs_soft_norm[torch.arange(0, batchSize), targets_batch]  # 选出对应标签的预测概率值
            prob_temp[prob_temp <= 1e-2] = 1e-2
            prob_temp[prob_temp > (1 - 1e-2)] = 1 - 1e-2
            discrepancy_measure1[start:end] = -torch.log(prob_temp)  # 衡量newlabel和原label的符合程度

            if discrepancy_corrected == 0:
                agreement_measure[start:end] = (
                        torch.max(probs_soft_norm, dim=1)[1] == targets_batch).float().data.cpu()

            train_new_labels[start:end] = new_labels

            cls_time.update(time.time() - end_time)

    tran_new_labels2 = train_new_labels.clone()
    with torch.no_grad():
        retrieval_one_hot_train = torch.zeros(k_nn, C).to(device)

        for i in range(batch_num):
            start = i * batch_size
            end = min((i + 1) * batch_size, trainFeatures.shape[0])

            feature_batch = trainFeatures[start:end]
            sim_batch = torch.mm(feature_batch, trainFeatures.T)

            targets_batch = trainNoisyLabels[start:end]
            batchSize = sim_batch.shape[0]

            yd, yi = sim_batch.topk(k_nn, dim=1, largest=True,
                                    sorted=True)  ## Top-K similar scores and corresponding indexes
            candidates = tran_new_labels2.view(1, -1).expand(batchSize, -1)  ##Replicate the labels per row to select
            retrieval = torch.gather(candidates, 1, yi)  ## 从candidates中沿着dim=1选出yi对应的数值
            ## 生成k_nn标签的独热编码 batchsize * k_nn * 10
            retrieval_one_hot_train.resize_(batchSize * k_nn, C).zero_()
            retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)

            yd_transform = yd.clone().div_(sigma).exp_()  ## Apply temperature to scores
            yd_transform[...] = 1.0  ##To avoid using similarities only counts
            probs_corrected_soft = torch.sum(
                torch.mul(retrieval_one_hot_train.view(batchSize, -1, C), yd_transform.view(batchSize, -1, 1)), 1)

            probs_norm_soft = probs_corrected_soft / torch.sum(probs_corrected_soft, dim=1)[:, None]

            prob_temp = probs_norm_soft[torch.arange(0, batchSize), targets_batch]
            prob_temp[prob_temp <= 1e-2] = 1e-2
            prob_temp[prob_temp > (1 - 1e-2)] = 1 - 1e-2

            discrepancy_measure2[start:end] = -torch.log(prob_temp)  # 衡量newlabel和原label的符合程度

            if discrepancy_corrected == 1:
                agreement_measure[start:end] = (
                        torch.max(probs_norm_soft, dim=1)[1] == targets_batch).float().data.cpu()

            cls_time.update(time.time() - end_time)

    #### check acc
    clean_idx = (temploader.dataset.noisy_labels == temploader.dataset.clean_labels).astype(float)
    agreement_measure = agreement_measure.cpu()

    right_num = torch.sum((torch.from_numpy(clean_idx) == 1.0) & (agreement_measure == 1.0))
    wrong_num = torch.sum((torch.from_numpy(clean_idx) != 1.0) & (agreement_measure == 1.0))

    precision = right_num / torch.sum(agreement_measure == 1.0) * 100.0
    error_rate = wrong_num / torch.sum(agreement_measure == 1.0) * 100.0
    acc = right_num / np.sum(clean_idx == 1.0) * 100.0

    progress_bar('precision: %.3f%% (%d/%d)' % (precision, right_num, torch.sum(agreement_measure == 1.0)))
    progress_bar('error_rate: %.3f%% (%d/%d)' % (error_rate, wrong_num, torch.sum(agreement_measure == 1.0)))
    progress_bar('acc: %.3f%% (%d/%d)' % (acc, right_num, np.sum(clean_idx == 1.0)))

    return agreement_measure, precision, error_rate, acc


def draw_pic3(precision_ls, error_rate_ls, find_true_sample_acc_ls, result_folder, noise_rate):
    """
    绘制训练过程曲线
    """
    precision_ls = torch.Tensor(precision_ls).cpu()
    error_rate_ls = torch.Tensor(error_rate_ls).cpu()
    find_true_sample_acc_ls = torch.Tensor(find_true_sample_acc_ls).cpu()

    x1 = torch.arange(0, len(precision_ls))
    x2 = torch.arange(0, len(error_rate_ls))
    x3 = torch.arange(0, len(find_true_sample_acc_ls))

    plt.figure(2)

    line1, = plt.plot(x1, precision_ls, color='green', marker='d', linestyle='-', markersize=6, alpha=0.5, linewidth=3)
    line2, = plt.plot(x2, error_rate_ls, color='blue', marker='o', linestyle='-', markersize=6, alpha=0.5, linewidth=3)
    line3, = plt.plot(x3, find_true_sample_acc_ls, color='orange', marker='*', linestyle='-', markersize=6, alpha=0.5,
                      linewidth=3)

    plt.legend([line1, line2, line3], ["precision", "error_rate", "find_true_sample_acc"], loc='upper right')
    plt.title(u"find_curves", fontsize=14, color='k')
    plt.savefig(result_folder + f'/noise_rate_{noise_rate}/find_result_curves.svg', dpi=600)  # 保存图片
    plt.show()


term_width = 10
last_time = time.time()
begin_time = last_time
TOTAL_BAR_LENGTH = 65.


def progress_bar(msg=None):
    global last_time, begin_time
    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    sys.stdout.write('\n')


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def log(p='\n', f=None):
    if f is None:
        print(p)
    else:
        f.write(p + '\n')
