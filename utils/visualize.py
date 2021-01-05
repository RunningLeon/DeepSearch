import os
import sys
import glob
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

from matplotlib import pyplot as plt

from .parse_result import decode_img, load_data_from_pkl
from .calculate import get_common_attacks

def plot_row_images(images, labels, title='', figsize=(12, 6), save_path='',  block=False):
    """
    可视化一行图像数据

    Args:
        images ([type]): [description]
        labels ([type]): [description]
        title (str, optional): [description]. Defaults to ''.
        figsize (tuple, optional): [description]. Defaults to (12, 6).
        save_path (str, optional): [description]. Defaults to ''.
        block (bool, optional): [description]. Defaults to False.
    """
    fig = plt.figure(figsize=figsize)
    n = len(images)
    for i in range(n):
        ax = plt.subplot(1, n, i+1)
        ax.imshow(images[i])
        ax.set_title(labels[i])
        ax.axis('off')
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show(block=block)


def boxplot(data, labels, title='', save_path='', figsize=(12,8),  block=False):
    """
    画箱线图

    Args:
        data ([type]): [description]
        labels ([type]): [description]
        title (str, optional): [description]. Defaults to ''.
        save_path (str, optional): [description]. Defaults to ''.
        figsize (tuple, optional): [description]. Defaults to (12,8).
        block (bool, optional): [description]. Defaults to False.
    """
    fig = plt.figure(figsize=figsize)
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xlabel('Attack Algorithm')
    plt.ylabel('Number of attack query')
    plt.grid()
    if title == '':
        title = 'Number of attack successfull query on 1000 image in ImageNet dataset' 
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show(block=block)


def plot_deepsearch_attack(model, ds_dir, dsref_dir, x_test, y_test, class_names, save_dir, topk=10):
    """
    可视化原图, DeepSeach 成功攻击图, 以及 Refine 后的攻击图

    Args:
        model ([type]): [description]
        ds_dir ([type]): [description]
        dsref_dir ([type]): [description]
        x_test ([type]): [description]
        y_test ([type]): [description]
        save_dir ([type]): [description]
        topk (int, optional): [description]. Defaults to 10.
    """
    labels = ['Original', 'Attacked', 'Refined']
    for i in tqdm(range(topk), desc='Ploting ds attack'):
        img_ori = decode_img(x_test[i])
        label_true = y_test[i]
        ds_img = load_data_from_pkl(os.path.join(ds_dir, f'image_{i}.pkl'))
        dsref_img = load_data_from_pkl(os.path.join(dsref_dir, f'image_{i}.pkl'))
        ds_pred = np.argmax(model.predict(ds_img)[0])
        dsref_pred = np.argmax(model.predict(dsref_img)[0])
        label_true = class_names[label_true]
        ds_pred = class_names[ds_pred]
        dsref_pred = class_names[dsref_pred]
        names = [label_true, ds_pred, dsref_pred]
        new_labels = [f'{labels[_]}: {names[_]}' for _ in range(len(names))]
        images = [img_ori, decode_img(ds_img), decode_img(dsref_img)]
        title = f'True class: {label_true}\n'
        title += f'Attacked class: {ds_pred}\n'
        title += f'Refined class: {dsref_pred}'
        save_path = os.path.join(save_dir, f'image_{i}.png')
        plot_row_images(images, labels, title, save_path=save_path, block=False)


def plot_all_attack(model, attack_images_li, attack_success_li, algo_names, x_test, y_test, class_names, save_dir, topk=10):
    """
    可视化所有对抗算法共同对抗成功的图片

    Args:
        model ([type]): [description]
        attack_images_li ([type]): [description]
        attack_success_li ([type]): [description]
        algo_names ([type]): [description]
        x_test ([type]): [description]
        y_test ([type]): [description]
        class_names ([type]): [description]
        save_dir ([type]): [description]
        topk (int, optional): [description]. Defaults to 10.
    """
    common_li = get_common_attacks(attack_success_li)
    nrof_common = len(common_li)

    if nrof_common > 0:
        topk = min(topk, nrof_common)
        image_mat = []
        plt_labels = ['Original'] + algo_names
        for i in tqdm(range(topk), desc='ploting all attack '):
            img_id = common_li[i]
            label = class_names[y_test[img_id]]
            ori_img = decode_img(x_test[img_id])
            imgs_row = [ori_img]
            title = f'Original: {label}\n'
            for j in range(len(algo_names)):
                atk_images = attack_images_li[j]
                atk_img = atk_images[img_id:img_id+1]
                atk_pred = np.argmax(model.predict(atk_img)[0])
                atk_label = class_names[atk_pred]
                title += f'{algo_names[j]}: {atk_label}\n'
                atk_img = decode_img(atk_img)
                imgs_row.append(atk_img)
            save_path = save_dir + f'image_{img_id}.png'
            plot_row_images(imgs_row, plt_labels, title, save_path=save_path, block=False)


def plot_time(time_li, algo_names=None, save_path='', block=False):
    """
    算法耗时直方图, 时间单位是小时

    Args:
        time_li ([type]): [description]
        algo_names ([type]): [description]
        save_path (str, optional): [description]. Defaults to ''.
        block (bool, optional): [description]. Defaults to False.
    """
    if algo_names is None:
        algo_names = ['QL-NES', 'Bandits', 'SimBA', 'Parsimonious', 'DeepSearch', 'DSRefine']
    fig = plt.figure(figsize=(10, 8))
    plt.bar(range(len(time_li)), time_li, tick_label=algo_names)
    plt.xticks = algo_names
    plt.grid()
    plt.xlabel('Attack Algo')
    plt.ylabel('Time/h')
    plt.title('Total time to perform attack on 1000 ImageNet test images')
    plt.show(block=block)
    if save_path:
        plt.savefig(save_path)
