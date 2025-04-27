import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import time

from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors

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


def get_transforms(dataset):
    if dataset == 'MNIST':
        MEAN = [0.1307]
        STD = [0.3081]
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    elif dataset == 'CIFAR10':
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])
    elif dataset == 'CIFAR100':
        CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
        CIFAR_STD = [0.2673, 0.2564, 0.2762]

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    else:
        raise ValueError('Invalid value {}'.format(dataset))

    return train_transform, test_transform


def save_accs(path, label, accs):
    with open(os.path.join(path, label + '.csv'), 'w') as f:
        m = accs.shape[0]
        f.write(','.join(['test ' + str(i + 1) for i in range(m)]) + '\n')
        for i in range(accs.shape[1]):
            f.write(','.join([str(f) for f in accs[:, i]]) + '\n')


def save_acc(path, label, accs):
    with open(os.path.join(path, label + '.csv'), 'w') as f:
        for a in accs:
            f.write(str(a) + '\n')


def draw_curves(acc_train_list, loss_train_list, acc_val_list, loss_val_list, acc_test_list, loss_test_list, args, lr):
    path = 'result/' + args.sess + '/seed' + str(args.seed) + '/' + str(args.loss) + '-Res34' + '/' + str(
        args.dataset) + '/' + str(args.noise_type) + '/' + str(args.noise_rate) + '/'

    if not os.path.exists(path):
        os.makedirs(path)
        print(path + " has been created.")
    else:
        print(path + " already exists.")

    acc_train_list = torch.Tensor(acc_train_list).cpu()
    loss_train_list = torch.Tensor(loss_train_list).cpu()
    acc_val_list = torch.Tensor(acc_val_list).cpu()
    loss_val_list = torch.Tensor(loss_val_list).cpu()
    acc_test_list = torch.Tensor(acc_test_list).cpu()
    loss_test_list = torch.Tensor(loss_test_list).cpu()

    x1 = torch.arange(0, len(acc_train_list))
    x2 = torch.arange(0, len(loss_train_list))
    x3 = torch.arange(0, len(acc_val_list))
    x4 = torch.arange(0, len(loss_val_list))
    x5 = torch.arange(0, len(acc_test_list))
    x6 = torch.arange(0, len(loss_test_list))

    plt.figure(1)
    line1, = plt.plot(x1, acc_train_list, color='red', marker='d', linestyle='--', markersize=6, alpha=0.5, linewidth=3)
    line2, = plt.plot(x2, loss_train_list, color='red', marker='o', linestyle='-', markersize=6, alpha=0.5, linewidth=3)
    line3, = plt.plot(x3, acc_val_list, color='yellow', marker='d', linestyle='--', markersize=6, alpha=0.5,
                      linewidth=3)
    line4, = plt.plot(x4, loss_val_list, color='yellow', marker='o', linestyle='-', markersize=6, alpha=0.5,
                      linewidth=3)
    line5, = plt.plot(x5, acc_test_list, color='blue', marker='d', linestyle='--', markersize=6, alpha=0.5, linewidth=3)
    line6, = plt.plot(x6, loss_test_list, color='blue', marker='o', linestyle='-', markersize=6, alpha=0.5, linewidth=3)
    plt.legend([line1, line2, line3, line4, line5, line6],
               ["ACC_train", "Loss_train", "ACC_val", "Loss_val", "ACC_test", "Loss_test"], loc='upper right')
    plt.title(u"all curves", fontsize=14, color='k')
    plt.savefig(path + str(lr) + 'all_curves.svg', dpi=600)  # 保存图片
    plt.show()


def draw_pies_max_and_12bios(output_info, result_folder):
    """
    获取标签正确和错误样本的信息
    """
    right_label_samples = output_info[output_info[:, 11] == output_info[:, 12]]
    error_label_samples = output_info[output_info[:, 11] != output_info[:, 12]]
    print('right label samples: ', right_label_samples.shape, 'error label samples: ', error_label_samples.shape)

    '''
    取出信息中的概率部分并由大到小排序
    '''
    origen_rlsamp_preds = right_label_samples[:, 1:11]
    origen_elsamp_preds = error_label_samples[:, 1:11]
    sorted_rlsamp_preds = torch.sort(right_label_samples[:, 1:11], descending=True)
    sorted_elsamp_preds = torch.sort(error_label_samples[:, 1:11], descending=True)

    '''
    记录概率中最大的值
    '''
    max_pred_value_rlsamp = sorted_rlsamp_preds.values[:, 0]
    max_pred_value_elsamp = sorted_elsamp_preds.values[:, 0]

    '''
    记录概率中第一第二差值
    '''
    bios12_value_rlsamp = sorted_rlsamp_preds.values[:, 0] - sorted_rlsamp_preds.values[:, 1]
    bios12_value_elsamp = sorted_elsamp_preds.values[:, 0] - sorted_elsamp_preds.values[:, 1]

    count_right_sample_max_pred = Counter(np.array(torch.round(max_pred_value_rlsamp * 10) / 10))
    count_error_sample_max_pred = Counter(np.array(torch.round(max_pred_value_elsamp * 10) / 10))
    count_bios12_right_label_sample = Counter(np.array(torch.round(bios12_value_rlsamp * 10) / 10))
    count_bios12_error_label_sample = Counter(np.array(torch.round(bios12_value_elsamp * 10) / 10))

    count_right_sample_max_pred = dict(sorted(count_right_sample_max_pred.items()))
    count_error_sample_max_pred = dict(sorted(count_error_sample_max_pred.items()))
    count_bios12_right_label_sample = dict(sorted(count_bios12_right_label_sample.items()))
    count_bios12_error_label_sample = dict(sorted(count_bios12_error_label_sample.items()))

    fig1, axs1 = plt.subplots(2, 2, figsize=(20, 20))

    unique_labels = list(set(count_right_sample_max_pred.keys()).union(
        count_error_sample_max_pred.keys(),
        count_bios12_right_label_sample.keys(),
        count_bios12_error_label_sample.keys()
    ))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    label_color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    # 生成颜色列表函数
    def get_color_list(counter):
        return [label_color_map[label] for label in counter.keys()]

    # 绘制饼状图
    axs1[0, 0].pie(count_right_sample_max_pred.values(), labels=count_right_sample_max_pred.keys(), autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 16}, colors=get_color_list(count_right_sample_max_pred))
    axs1[0, 0].set_title('count_right_sample_max_pred', fontsize=20)

    axs1[0, 1].pie(count_error_sample_max_pred.values(), labels=count_error_sample_max_pred.keys(), autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 16}, colors=get_color_list(count_error_sample_max_pred))
    axs1[0, 1].set_title('count_error_sample_max_pred', fontsize=20)

    axs1[1, 0].pie(count_bios12_right_label_sample.values(), labels=count_bios12_right_label_sample.keys(),
                   autopct='%1.1f%%', startangle=90, textprops={'fontsize': 16},
                   colors=get_color_list(count_bios12_right_label_sample))
    axs1[1, 0].set_title('count_bios12_right_label_sample', fontsize=20)

    axs1[1, 1].pie(count_bios12_error_label_sample.values(), labels=count_bios12_error_label_sample.keys(),
                   autopct='%1.1f%%', startangle=90, textprops={'fontsize': 16},
                   colors=get_color_list(count_bios12_error_label_sample))
    axs1[1, 1].set_title('count_bios12_error_label_sample', fontsize=20)

    # 调整布局并保存图形
    plt.tight_layout()
    plt.savefig(result_folder + 'pie_pic_1.svg', dpi=600)

    # 显示图形
    plt.show()


def draw_pies_real_label_value_and_sort(output_info, result_folder):
    right_label_samples = output_info[output_info[:, 11] == output_info[:, 12]]
    error_label_samples = output_info[output_info[:, 11] != output_info[:, 12]]
    print('right label samples: ', right_label_samples.shape, 'error label samples: ', error_label_samples.shape)

    origen_rlsamp_preds = right_label_samples[:, 1:11]
    origen_elsamp_preds = error_label_samples[:, 1:11]
    sorted_rlsamp_preds = torch.sort(right_label_samples[:, 1:11], descending=True)
    sorted_elsamp_preds = torch.sort(error_label_samples[:, 1:11], descending=True)

    '''
    记录概率真实标签的概率大小位置和概率值
    '''
    real_location_rlsam = []
    real_location_elsam = []

    for i in range(len(right_label_samples[:, 11])):
        fin_label = right_label_samples[i, 11]
        local = (sorted_rlsamp_preds.indices[i, :] == (fin_label)).nonzero(as_tuple=True)
        real_location_rlsam.append(local)

    real_location_rlsam = torch.Tensor(real_location_rlsam)
    r_p_right_label_sample = torch.tensor(
        [sorted_rlsamp_preds.values[i, real_location_rlsam[i][0].int()] for i in range(len(real_location_rlsam))])

    for i in range(len(error_label_samples[:, 11])):
        fin_label = error_label_samples[i, 11]
        local = (sorted_elsamp_preds.indices[i, :] == (fin_label)).nonzero(as_tuple=True)
        real_location_elsam.append(local)

    real_location_elsam = torch.Tensor(real_location_elsam)
    r_p_error_label_sample = torch.tensor(
        [sorted_elsamp_preds.values[i, real_location_elsam[i][0].int()] for i in range(len(real_location_elsam))])

    count_actual_label_pv_rsamp = Counter(np.array(torch.round(r_p_right_label_sample * 10) / 10))
    count_actual_label_pv_esamp = Counter(np.array(torch.round(r_p_error_label_sample * 10) / 10))
    count_actual_label_location_rsamp = Counter(real_location_rlsam.numpy().flatten())
    count_actual_label_location_esamp = Counter(real_location_elsam.numpy().flatten())

    count_actual_label_pv_rsamp = dict(sorted(count_actual_label_pv_rsamp.items()))
    count_actual_label_pv_esamp = dict(sorted(count_actual_label_pv_esamp.items()))
    count_actual_label_location_rsamp = dict(sorted(count_actual_label_location_rsamp.items()))
    count_actual_label_location_esamp = dict(sorted(count_actual_label_location_esamp.items()))

    fig2, axs2 = plt.subplots(2, 2, figsize=(20, 20))

    unique_labels = list(set(count_actual_label_pv_rsamp.keys()).union(
        count_actual_label_pv_esamp.keys(),
        count_actual_label_location_rsamp.keys(),
        count_actual_label_location_esamp.keys()
    ))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    label_color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    # 生成颜色列表函数
    def get_color_list(counter):
        return [label_color_map[label] for label in counter.keys()]

    # 绘制饼状图
    axs2[0, 0].pie(count_actual_label_pv_rsamp.values(), labels=count_actual_label_pv_rsamp.keys(), autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 16}, colors=get_color_list(count_actual_label_pv_rsamp))
    axs2[0, 0].set_title('count_actual_label_pv_rsamp', fontsize=20)

    axs2[0, 1].pie(count_actual_label_pv_esamp.values(), labels=count_actual_label_pv_esamp.keys(), autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 16}, colors=get_color_list(count_actual_label_pv_esamp))
    axs2[0, 1].set_title('count_actual_label_pv_esamp', fontsize=20)

    axs2[1, 0].pie(count_actual_label_location_rsamp.values(), labels=count_actual_label_location_rsamp.keys(),
                   autopct='%1.1f%%', startangle=90, textprops={'fontsize': 16},
                   colors=get_color_list(count_actual_label_location_rsamp))
    axs2[1, 0].set_title('count_actual_label_location_rsamp', fontsize=20)

    axs2[1, 1].pie(count_actual_label_location_esamp.values(), labels=count_actual_label_location_esamp.keys(),
                   autopct='%1.1f%%', startangle=90, textprops={'fontsize': 16},
                   colors=get_color_list(count_actual_label_location_esamp))
    axs2[1, 1].set_title('count_actual_label_location_esamp', fontsize=20)

    # 调整布局并保存图形
    plt.tight_layout()
    plt.savefig(result_folder + 'pie_pic_2.svg', dpi=600)

    # 显示图形
    plt.show()


# 计算 k-距离
def compute_k_distance(X, k):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = distances[:, -1]  # 取第 k 个距离
    return np.sort(k_distances)


def get_model(model, k):
    if model == 'k-means':
        return KMeans(n_clusters=k, random_state=2024)
    elif model == 'AGNES':
        return AgglomerativeClustering(n_clusters=k)
    elif model == 'DBSCAN':
        return DBSCAN(eps=0.601, min_samples=512)
    else:
        print('Unknown model')


def DBSCAN_cluster(data, result_path):
    min_samples = data.shape[1]
    k_distances = compute_k_distance(data, min_samples)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(k_distances)), k_distances, marker='o')
    plt.xlabel('Points')
    plt.ylabel(f'{min_samples}-Distance')
    plt.title('k-Distance Graph')

    # 计算梯度的二阶导数
    gradients = np.gradient(k_distances)
    second_derivative = np.gradient(gradients)
    # 找到 elbow points 候选
    elbow_points = np.argwhere(second_derivative < -np.percentile(second_derivative, 99)).flatten()
    # 绘制肘部点
    plt.plot(elbow_points, k_distances[elbow_points], 'rx', label='Elbow Point')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(len(elbow_points))

    scores = []
    label_list = []
    bioss = []

    for i in range(len(elbow_points)-1):
        bios = k_distances[elbow_points[i+1]] - k_distances[elbow_points[i]]
        bioss.append(bios)

    max_bios = np.argmax(bioss)
    final_indice = elbow_points[max_bios+1]

    eps = k_distances[final_indice]
    print(eps)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    db = model.fit(data)
    labels = db.labels_

    #     if elbow_points[i] > 0:
    #         eps = elbow_points[i]
    #         model = DBSCAN(eps=eps, min_samples=min_samples)
    #         db = model.fit(data)
    #         labels = db.labels_
    #         if len(np.unique(labels)) > 1:
    #             # 假设 X 是数据，labels 是 DBSCAN 聚类结果的标签
    #             silhouette_avg = silhouette_score(data, labels)
    #             scores.append(silhouette_avg)
    #             label_list.append(labels)
    #
    # final_indice = np.argmax(scores)
    # Labels = label_list[final_indice]

    return labels


def my_cluster_function(data, model, k, result_path):
    if model == 'k-means':
        model = get_model(model, k)
        labels = model.fit_predict(data)
        cluster_centers = model.cluster_centers_
        # 使用 transform 方法计算每个样本点到每个聚类中心的距离
        distances = model.transform(data)
        nearest_core_distances = np.min(distances, axis=1)  # 获取每个点到最近核心点的距离

    else:
        model = get_model(model, k)
        labels = model.fit_predict(data)
        # 计算每个簇的中心
        cluster_centers = []
        for label in np.unique(labels):
            cluster_points = data[labels == label]
            center = np.mean(cluster_points, axis=0)
            cluster_centers.append(center)
        distances = pairwise_distances(data, cluster_centers)
        nearest_core_distances = np.min(distances, axis=1)  # 获取每个点到最近核心点的距离

    # print("Cluster centers:", cluster_centers)

    # 使用 PCA 降维到 2 维以便可视化
    pca = PCA(n_components=2)
    data_reduced = pca.fit_transform(data)

    # 绘制散点图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=labels, cmap='viridis', edgecolor='k')

    # 绘制聚类中心
    centers_reduced = pca.transform(cluster_centers)
    plt.scatter(centers_reduced[:, 0], centers_reduced[:, 1], c='red', marker='x', s=100, label='Cluster Centers')

    # 添加颜色条
    plt.colorbar(scatter, label='Cluster Labels')

    # 添加标题和标签
    plt.title('Hierarchical Clustering with PCA Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.savefig(result_path + 'cluster_fig.svg')
    # 显示图形
    plt.show()

    return labels, cluster_centers, distances


'''
统计某个距离阈值下各类样本的噪声占比
'''


def above_blow_noise_ratio(labels, centers, distances, output_fea, result_path):
    above_avg_indices = []
    below_avg_indices = []
    above_avg_noisy_indices = []
    below_avg_noisy_indices = []
    noisy_indices = np.where(output_fea[:, -3] != output_fea[:, -2])[0]
    lamda = 1.0

    for clas in range(len(centers)):
        clas_indices = np.where(labels == clas)[0]  # 找到属于当前簇的样本
        clas_distances_to_c = distances[clas_indices, clas]
        av_dist = np.mean(clas_distances_to_c)  # 当前簇距离中心点的平均距离
        clas_noise_indices = np.intersect1d(clas_indices, noisy_indices)  # 找到属于当前簇的噪声样本

        # 分离样本:以平均距离为阈值
        above_avg = clas_indices[clas_distances_to_c > lamda * av_dist]
        below_avg = clas_indices[clas_distances_to_c <= lamda * av_dist]
        # 大于和小于阈值的样本中的噪声样本统计
        above_avg_noisy = np.intersect1d(above_avg, clas_noise_indices)
        below_avg_noisy = np.intersect1d(below_avg, clas_noise_indices)

        above_avg_indices.append(above_avg)
        below_avg_indices.append(below_avg)
        above_avg_noisy_indices.append(above_avg_noisy)
        below_avg_noisy_indices.append(below_avg_noisy)

        # 打印结果
        print(f"类 {clas} 中，距中心平均距离: {av_dist:.2f}\n"
              f"距离大于平均距离的样本num: {len(above_avg)}",
              f"噪声占比：{(len(above_avg_noisy) / len(above_avg)):.3f}")
        print(f"距离小于或等于平均距离的样本num: {len(below_avg)}",
              f"噪声占比：{(len(below_avg_noisy) / len(below_avg)):.3f}")

        with open(result_path + "above_blow_noise_ratio.txt", "a", encoding="utf-8") as file:
            file.write(f"类 {clas} 中，距中心平均距离: {av_dist:.2f}" + "\n")
            file.write(f"距离大于平均距离的样本num: {len(above_avg)}" + " ")
            file.write(f"噪声占比：{(len(above_avg_noisy) / len(above_avg)):.3f}" + "\n")
            file.write(f"距离小于或等于平均距离的样本num: {len(below_avg)}" + " ")
            file.write(f"噪声占比：{(len(below_avg_noisy) / len(below_avg)):.3f}" + "\n")

    return above_avg_indices, below_avg_indices, above_avg_noisy_indices, below_avg_noisy_indices


'''
统计每一类别下noisy-sample据中心点距离分布
'''


def distribution_noisy_sample_dist_to_center(labels, class_centers, distances, output_fea, layer_num,
                                             result_path):
    counter_fresh_list = []
    counter_noisy_list = []
    noisy_indices = np.where(output_fea[:, -3] != output_fea[:, -2])[0]

    for clas in range(len(class_centers)):
        # 找到属于当前簇的样本
        clas_indices = np.where(labels == clas)[0]
        common_indices = np.intersect1d(clas_indices, noisy_indices)  # common即noisy
        fresh_indices = clas_indices[~np.isin(clas_indices, common_indices)]

        combined_bc = np.concatenate((common_indices, fresh_indices))
        print(np.array_equal(np.sort(clas_indices), np.sort(combined_bc)))

        # 依据indices取出距离
        noisy_sample_dis = distances[common_indices, clas]
        fresh_sample_dis = distances[fresh_indices, clas]

        # 保留一位小数
        noisy_dis = np.round(noisy_sample_dis, 2)
        fresh_dis = np.round(fresh_sample_dis, 2)

        # 使用 Counter 统计每个值的出现次数
        counter_noisy = Counter(noisy_dis)
        counter_fresh = Counter(fresh_dis)

        counter_noisy_list.append(counter_noisy)
        counter_fresh_list.append(counter_fresh)

    # 遍历每个子图
    fig1, axs1 = plt.subplots(5, 2, figsize=(12, 20), constrained_layout=True)
    fig1.suptitle(f'layer{layer_num}_{output_fea.shape[1] - 4} noisy-sample距聚类中心中心距离分布', fontsize=16)
    for i in range(5):
        for j in range(2):
            idx = i * 2 + j  # 计算当前图像的索引
            axs1[i, j].pie(counter_noisy_list[idx].values(), labels=counter_noisy_list[idx].keys(), autopct='%1.1f%%',
                           startangle=90)
            axs1[i, j].set_title(f'簇{idx} noisy-sample')

    plt.savefig(result_path + f'layer{layer_num}_{output_fea.shape[1] - 4} noisy-sample距聚类中心中心距离分布.svg')
    plt.show()

    fig2, axs2 = plt.subplots(5, 2, figsize=(12, 20), constrained_layout=True)
    fig2.suptitle(f'layer{layer_num} fresh-sample距聚类中心中心距离分布', fontsize=16)

    for i in range(5):
        for j in range(2):
            idx = i * 2 + j  # 计算当前图像的索引
            axs2[i, j].pie(counter_fresh_list[idx].values(), labels=counter_fresh_list[idx].keys(), autopct='%1.1f%%',
                           startangle=90)
            axs2[i, j].set_title(f'簇{idx} fresh-sample')

    plt.savefig(result_path + f'layer{layer_num}_{output_fea.shape[1] - 4} fresh-sample距聚类中心中心距离分布.svg')
    plt.show()

    return counter_noisy_list, counter_fresh_list





def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (1, k, d, d)
    """
    res = torch.zeros(mat_a.shape).to(mat_a.device)

    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        res[:, i, :, :] = mat_a_i.mm(mat_b_i.to(mat_a.device)).unsqueeze(1)

    return res


def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)