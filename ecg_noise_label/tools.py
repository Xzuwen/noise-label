from torch.utils.data import Dataset
from util import *


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

def supervised_masks_estimation(args, labels, xbm, xbm_labels, mix_index, epoch, bsz, device):
    ###################### Supervised mask excluding augmented view ###############################
    labels = labels.contiguous().view(-1, 1)

    if labels.shape[0] != bsz:
        raise ValueError('Num of labels does not match num of features')

    ##Create mask without diagonal to avoid augmented view, i.e. this is supervised mask
    maskSup_batch = torch.eq(labels, labels.t()).float() - torch.eye(bsz, dtype=torch.float32).to(device)

    if args.xbm_use == 1 and epoch > args.xbm_begin:
        ## Extend mask to consider xbm_memory features
        maskSup_mem = torch.eq(labels, xbm_labels.t()).float().to(device)

        if xbm.ptr == 0:
            maskSup_mem[:, -1 * bsz:] = maskSup_batch
        else:
            maskSup_mem[:, (xbm.ptr - bsz):xbm.ptr] = maskSup_batch

    else:
        maskSup_mem = []

    mask2Sup_batch = torch.eq(labels[mix_index], labels.t()).float() - torch.eye(bsz, dtype=torch.float32).to(device)

    if args.xbm_use == 1 and epoch > args.xbm_begin:
        ## Extend mask to consider xbm_memory features. Here we consider that the label for images is the minor one, i.e. labels[mix_index1], labels[mix_index2] and xbm_labels_mix
        ## Here we don't repeat the columns part as in maskSup because the minor label is different for the first and second part of the mini-batch (different mixup shuffling for each mini-batch part)
        mask2Sup_mem = torch.eq(labels[mix_index],
                                xbm_labels.t()).float()  ##Mini-batch samples with memory samples (add columns)

        if xbm.ptr == 0:
            mask2Sup_mem[:, -1 * bsz:] = mask2Sup_batch

        else:
            mask2Sup_mem[:, (xbm.ptr - bsz):xbm.ptr] = mask2Sup_batch

    else:
        mask2Sup_mem = []

    return maskSup_batch, maskSup_mem, mask2Sup_batch, mask2Sup_mem


#### Losses
def InterpolatedContrastiveLearning_loss(args, pairwise_comp, lam, maskSup, mask2Sup, logits_mask):
    logits = torch.div(pairwise_comp, args.batch_t)

    exp_logits = torch.exp(logits) * logits_mask  # remove diagonal
    if args.aprox == 1:  # 相当于求log-softmax,得到log_prob
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  ## Approximation for numerical stability taken from supervised contrastive learning
    else:
        log_prob = torch.log(torch.exp(logits) + 1e-10) - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)


    exp_logits2 = torch.exp(logits) * logits_mask  # remove diagonal
    if args.aprox == 1:
        log_prob2 = logits - torch.log(exp_logits2.sum(1, keepdim=True))  ## Approximation for numerical stability taken from supervised contrastive learning
    else:
        log_prob2 = torch.log(torch.exp(logits) + 1e-10) - torch.log(exp_logits2.sum(1, keepdim=True) + 1e-10)


    mean_log_prob_pos_sup = (maskSup * log_prob).sum(1) / log_prob.shape[1]
    mean_log_prob_pos2_sup = (mask2Sup * log_prob2).sum(1) / log_prob2.shape[1]
    lossa = - lam * mean_log_prob_pos_sup
    lossb = -(1.0-lam) * mean_log_prob_pos2_sup
    loss = torch.cat((lossa, lossb))
    loss = loss.mean()

    return loss


## Semi-supervised learning
def ClassificationLoss(args, preds, preds_noMix, y_a, y_b, lam, mix_index, criterionCE, agreement_batch, epoch, device):
    if args.PredictiveCorrection == 0 or epoch <= args.startLabelCorrection:
        loss = lam * criterionCE(preds, y_a.long()) + (1 - lam) * criterionCE(preds, y_b.long())
        loss = loss.mean()


    elif args.PredictiveCorrection == 1 and epoch > args.startLabelCorrection:
        agreement_measure = agreement_batch.to(device)
        lossLabeled = agreement_measure * (
                lam * criterionCE(preds, y_a.long()) + (1 - lam) * criterionCE(preds, y_b.long()))
        lossLabeled = lossLabeled.mean()

        ## Pseudo-labeling
        prob = F.softmax(preds_noMix, dim=1)
        z1 = prob.clone().detach()  # (128, 2)
        z2 = z1[mix_index, :]  # (128, 2)

        preds_logSoft = F.log_softmax(preds, dim=1)  # (128, 2)

        loss_x1_pred_vec = lam * (1 - agreement_measure) * (-torch.sum(z1 * preds_logSoft, dim=1))  ##Soft
        loss_x1_pred = torch.sum(loss_x1_pred_vec) / len(loss_x1_pred_vec)

        loss_x2_pred_vec = (1 - lam) * (1 - agreement_measure[mix_index]) * (-torch.sum(z2 * preds_logSoft, dim=1))  ##Soft
        loss_x2_pred = torch.sum(loss_x2_pred_vec) / len(loss_x2_pred_vec)

        lossUnlabeled = loss_x1_pred + loss_x2_pred

        loss = lossLabeled + lossUnlabeled

    return loss


def train(args, xbm, epoch, data_loader, model, optimizer, agreement, result_path, device):
    correct = 0.
    total = 0.
    total_loss = 0.
    agreement = agreement.float().to(device)
    criterionCE = torch.nn.CrossEntropyLoss(reduction='none')

    sample_features = torch.zeros((len(data_loader.dataset), 128), dtype=torch.float32).to(device)

    model.train()

    for index, data, clean_label, noise_label in data_loader:
        data, noise_label, index = data.to(device), noise_label.to(device), index.to(device)

        signal, y_a, y_b, mix_index, lam = mix_data_lab(data, noise_label, args.alpha, device)

        model.zero_grad()
        bsz = signal.shape[0]

        preds, embed = model(signal)
        preds_noMix, embed_NoMix = model(data)
        sample_features[index] = embed.detach()
        agreement_batch = agreement[index]

        classify_loss = ClassificationLoss(args, preds, preds_noMix, y_a, y_b, lam, mix_index, criterionCE, agreement_batch, epoch,
                                           device)
        ############# Update memory ##############
        if args.xbm_use == 1:
            xbm.enqueue_dequeue(embed.detach(), noise_label.detach().squeeze())

        ############# Get features from memory ##############
        if args.xbm_use == 1 and epoch > args.xbm_begin:
            xbm_feats, xbm_labels = xbm.get()
            xbm_labels = xbm_labels.unsqueeze(1)
        else:
            xbm_feats, xbm_labels = [], []
        #####################################################
        pairwise_comp_batch = torch.matmul(embed, embed.t())

        if args.xbm_use == 1 and epoch > args.xbm_begin:
            pairwise_comp_mem = torch.matmul(embed, xbm_feats.t())  # Compare mini-batch with memory

        maskSup_batch, maskSup_mem, mask2Sup_batch, mask2Sup_mem = supervised_masks_estimation(args, noise_label, xbm, xbm_labels, mix_index, epoch, bsz, device)

        # Mask-out self-contrast cases
        logits_mask_batch = (torch.ones_like(maskSup_batch) - torch.eye(bsz).to(device))

        inter_loss = InterpolatedContrastiveLearning_loss(args, pairwise_comp_batch, lam, maskSup_batch, mask2Sup_batch, logits_mask_batch)

        if args.xbm_use == 1 and epoch > args.xbm_begin:
            logits_mask_mem = torch.ones_like(maskSup_mem)  ## Negatives mask, i.e. all except self-contrast sample

            if xbm.ptr == 0:
                logits_mask_mem[:, -1 * bsz:] = logits_mask_batch
            else:
                logits_mask_mem[:, xbm.ptr - (1 * bsz):xbm.ptr] = logits_mask_batch

            inter_loss_mem = InterpolatedContrastiveLearning_loss(args, pairwise_comp_mem, lam, maskSup_mem, mask2Sup_mem, logits_mask_mem)

            inter_loss = inter_loss + inter_loss_mem

        loss = inter_loss + classify_loss
        loss.backward()
        optimizer.step()

        probs = F.softmax(preds, dim=1)
        pred = torch.argmax(probs, 1)
        total_loss += loss.mean().item()
        total += noise_label.size(0)
        correct += (pred == noise_label).sum().item()

    with torch.no_grad():
        train_acc = 100. * float(correct) / float(total)
        train_loss = total_loss / len(data_loader)
        progress_bar('train-Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss, train_acc, correct, total))

    path = result_path + f'/features/'
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(sample_features, path + f'/sample_features_epoch{epoch}.pth')

    return train_acc, train_loss


def test(loader, model, criterion, device):
    correct = 0.
    total = 0.
    total_loss = 0.
    model.eval()

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logit, feat = model(batch_x)
        loss = criterion(logit, batch_y.long())
        total_loss += loss.mean().item()
        probs = F.softmax(logit, dim=1)
        pred = torch.argmax(probs, 1)
        total += batch_y.size(0)
        correct += (pred == batch_y).sum().item()

    acc = 100. * float(correct) / float(total)
    test_loss = total_loss / len(loader)
    progress_bar('test-Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss, acc, correct, total))

    return acc, test_loss


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def LabelNoiseDetection_Mean(ep, net, dataset, k_nn, trainloader, discrepancy_corrected, sigma, result_path, device):
    net.eval()
    cls_time = AverageMeter()
    end_time = time.time()

    trainLabels = torch.LongTensor(trainloader.dataset.noise_label.long()).to(device)

    C = trainLabels.max() + 1  # 类别数目

    ## Get train features
    temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False)

    trainFeatures = torch.load(result_path + f'/features/sample_features_epoch{ep}.pth').to(
        device)

    trainNoisyLabels = torch.LongTensor(temploader.dataset.noise_label.long()).to(device)
    train_new_labels = torch.LongTensor(temploader.dataset.noise_label.long()).to(device)

    agreement_measure = torch.zeros((len(temploader.dataset.noise_label),))  #.to(device)

    claas_0_indices = torch.where(trainNoisyLabels == 0)[0]
    class_1_indices = torch.where(trainNoisyLabels == 1)[0]
    class_2_indices = torch.where(trainNoisyLabels == 2)[0]
    class_3_indices = torch.where(trainNoisyLabels == 3)[0]


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

            if dataset=='afdb':
                sim_batch_0 = sim_batch[:, claas_0_indices].mean(dim=1)
                sim_batch_1 = sim_batch[:, class_1_indices].mean(dim=1)

                detected_label = torch.where(sim_batch_0 > sim_batch_1, 0, 1).squeeze()

            elif dataset=='adb':
                sim_batch_0 = sim_batch[:, claas_0_indices].mean(dim=1)
                sim_batch_1 = sim_batch[:, class_1_indices].mean(dim=1)
                sim_batch_2 = sim_batch[:, class_2_indices].mean(dim=1)
                sim_batch_3 = sim_batch[:, class_3_indices].mean(dim=1)

                sim_batches = torch.stack([sim_batch_0, sim_batch_1, sim_batch_2, sim_batch_3], dim=0).T
                detected_label = torch.argmax(sim_batches, dim=1)

            agreement_measure[start:end] = torch.where(detected_label == targets_batch, 1, 0)
    cls_time.update(time.time() - end_time)
    #### check acc
    clean_idx = (temploader.dataset.noise_label == temploader.dataset.clean_label).float()
    agreement_measure = agreement_measure.cpu()

    right_num = torch.sum((clean_idx == 1.0) & (agreement_measure == 1.0))
    wrong_num = torch.sum((clean_idx != 1.0) & (agreement_measure == 1.0))

    precision = right_num / torch.sum(agreement_measure == 1.0) * 100.0
    error_rate = wrong_num / torch.sum(agreement_measure == 1.0) * 100.0
    acc = right_num / torch.sum(clean_idx == 1.0) * 100.0

    progress_bar('precision: %.3f%% (%d/%d)' % (precision, right_num, torch.sum(agreement_measure == 1.0)))
    progress_bar('error_rate: %.3f%% (%d/%d)' % (error_rate, wrong_num, torch.sum(agreement_measure == 1.0)))
    progress_bar('acc: %.3f%% (%d/%d)' % (acc, right_num, torch.sum(clean_idx == 1.0)))

    return agreement_measure, precision, error_rate, acc

