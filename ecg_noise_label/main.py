import os
import random
import time
import argparse
import logging
import numpy as np
from torch import optim

from models.resnet import BasicBlock, ResNet18
from data_load.data_process import load_data
from xbm_memory import *
from tools import train, test, LabelNoiseDetection_Mean


logger = logging.getLogger()
logger.setLevel(logging.INFO)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    parser.add_argument('--data_path', type=str, default='./data/', help='data path')
    parser.add_argument('--dataset', type=str, default='afdb', help='adb / afdb / NSA')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='#ecgs in each mini-batch')
    parser.add_argument('--cuda_dev', type=int, default=0, help='GPU to select')
    parser.add_argument('--epoch', type=int, default=100, help='training epoches')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

    parser.add_argument('--noise_type', default='symmetric', help='symmetric or asymmetric')
    parser.add_argument('--noise_ratio', type=float, default=0.4, help='percent of noise')
    parser.add_argument('--sigma', type=float, default=1.0, help='temperature')
    parser.add_argument('--alpha', type=float, default=1.0, help='Beta distribution parameter for mixup')
    parser.add_argument('--seed', type=int, default=2025, help='random seed (default: 1)')
    parser.add_argument('--M', action='append', type=int, default=[125, 200], help="Milestones for the LR sheduler")
    parser.add_argument('--experiment_name', type=str, default = 'ecg_5',help='name of the experiment (for the output files)')
    parser.add_argument('--method', type=str, default='Fea_sim', help='Fea_sim')
    parser.add_argument('--initial_epoch', type=int, default=1, help="Star training at initial_epoch")
    parser.add_argument('--DA', type=str, default="complex", help='Choose simple or complex data augmentation')
    parser.add_argument('--low_dim', type=int, default=128, help='Size of contrastive learning embedding')
    parser.add_argument('--mix_labels', type=int, default=0, help='1: Interpolate two input data and "interpolate" labels')
    parser.add_argument('--batch_t', default=0.1, type=float, help='Contrastive learning temperature')
    parser.add_argument('--aprox', type=int, default=1, help='log_prob求法')
    parser.add_argument('--headType', type=str, default="Linear", help='Linear, NonLinear')
    parser.add_argument('--xbm_use', type=int, default=1, help='1: Use xbm')
    parser.add_argument('--xbm_begin', type=int, default=1, help='Epoch to begin using memory')
    parser.add_argument('--xbm_per_class', type=int, default=2000, help='Num of samples per class to store in the memory. Memory size = xbm_per_class*num_classes')
    parser.add_argument('--startLabelCorrection', type=int, default=20, help='Epoch to start label correction')
    parser.add_argument('--k_val', type=int, default=1000, help='k for k-nn correction')
    parser.add_argument('--use_cleanLabels', type=int, default=0, help='Train the classifier with clean labels')
    parser.add_argument('--PredictiveCorrection', type=int, default=1, help='Enable predictive label correction')
    parser.add_argument('--discrepancy_corrected', type=int, default=0, help='Use corrected label for discrepancy measure')


    args = parser.parse_args()
    return args



def main(args):
    # other parameters
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_dev)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noise_rate = f"{int(args.noise_ratio * 10):02d}"
    result_path = os.getcwd() + '/result/' + args.dataset + '/' + args.noise_type + '/' + noise_rate + '/'
    os.makedirs(result_path, exist_ok=True)
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed)  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed_initialization)  # GPU seed

    random.seed(args.seed)  # python seed for image transformation


    # data loader
    num_classes = args.num_classes
    train_loader, test_loader = load_data(args.data_path, args.dataset, args.batch_size, args.noise_type, args.noise_ratio, args.seed)
    # 保留原始数据的标签
    n_label = train_loader.dataset.noise_label
    c_label = train_loader.dataset.clean_label
    np.save(result_path + 'n_label.npy', n_label)
    np.save(result_path + 'c_label.npy', c_label)

    # model
    model = ResNet18(BasicBlock, num_classes=num_classes).to(device)
    print('Total params: {:.2f} M'.format((sum(p.numel() for p in model.parameters()) / 1000000.0)))


    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    # milestones = args.M
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    ############### Memory creation
    if args.xbm_use == 1:
        xbm = XBM(args, device)
    else:
        xbm = []


    agreement = torch.ones(len(train_loader.dataset))
    agreement_acc_ls = []
    agreement_pre_ls = []
    agreement_err_ls = []
    acc_train_ls = []
    loss_train_ls = []
    acc_test_ls = []
    loss_test_ls = []

    best_acc = 0.0
    best_ep = 0


    for epoch in range(args.initial_epoch, args.epoch + 1):
        print('\nEpoch: %d/%d' % (epoch, args.epoch))
        # scheduler.step()
        train_acc, train_loss = train(args, xbm, epoch, train_loader, model, optimizer, agreement, result_path, device)
        acc_train_ls.append(train_acc)
        loss_train_ls.append(train_loss)

        agreement, precision, error_rate, acc = LabelNoiseDetection_Mean(epoch, model, args.dataset, args.k_val, train_loader, args.discrepancy_corrected, args.sigma, result_path, device)
        agreement_acc_ls.append(acc)
        agreement_pre_ls.append(precision)
        agreement_err_ls.append(error_rate)

        print('######## Test ########')
        test_acc, test_loss = test(test_loader, model, criterion, device)
        acc_test_ls.append(test_acc)
        loss_test_ls.append(test_loss)
        if test_acc > best_acc:
            best_acc = test_acc
            best_ep = epoch
            torch.save(model.state_dict(), result_path + "/model_weights.pth")
            print('Best model saved at epoch %d' % epoch)


    print('best_acc: {:.4f}'.format(best_acc), 'at ep: {:d}'.format(best_ep))



if __name__ == "__main__":
    args = parse_args()
    logging.info(args)
    main(args)
