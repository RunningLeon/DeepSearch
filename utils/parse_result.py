import os
import sys

import pickle
import numpy as np
from tqdm import tqdm

def decode_img(img):
    """
    将 float 类型[1 x H x W x C]图像数据转为 uint8 类型图像数据
    Args:
        img ([type]): [description]

    Returns:
        [type]: [description]
    """
    img = np.squeeze(img * 255).astype(np.uint8)
    return img


def load_data_from_pkl(pkl_path):
    """
    输入文件进行反序列化    
    Args:
        pkl_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        return data


def get_images_from_pkl(input_dir, batch_size=1, num_image=1000):
    """
    获取输入文件夹里所有 图像的 pkl 文件, 此版本只支持 ImageNet 里输出的 pkl文件

    Args:
        input_dir ([type]): [description]
        batch_size (int, optional): [description]. Defaults to 1.
        num_image (int, optional): [description]. Defaults to 1000.

    Returns:
        [type]: [description]
    """
    data_li = []
    pbar = tqdm(range(0, num_image, batch_size), desc='Reading image ')
    for i in pbar:
        if batch_size == 1:
            filename = f'image_{i}.pkl'
        else:
            filename = f'image_{i}_to_{i+batch_size-1}.pkl'
        pkl_path = os.path.join(input_dir, filename)
        assert os.path.exists(pkl_path), f'File not exists:{pkl_path}'
        data = load_data_from_pkl(pkl_path)
        data_li.append(data)
    ret = np.vstack(data_li)
    return ret


def parse_result_from_pkl(pkl_path):
    """
    从 data.pkl 文件中解析出攻击成功, 成功率, 查询次数等信息

    Args:
        pkl_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    success_li, query_li = [], []
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        keys = sorted(data.keys())
        for i in keys:
            result = data[i]
            if len(result) == 2:
                success, num_query = result
            else:
                # 次数对 DeepSearch ReFine 的输出里的查询次数也进行统计, 取了最后一个查询次数
                # process DSRef data.pkl with success, [(402, 0.027757436),...], xx
                success = result[0]
                num_query = result[1][-1][0]
            success = 1 if success else 0
            query_li.append(num_query)
            success_li.append(success)
    success_rate = 0 if not success_li else sum(success_li) / len(success_li)
    success_query = [i for i, j in zip(query_li, success_li) if j]
    med = np.median(success_query)
    avg = np.mean(success_query)
    print(f'success rate: {success_rate} average query: {avg} median query: {med}')
    return success_li, success_query, success_rate, avg, med

