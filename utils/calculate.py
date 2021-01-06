import numpy as np
from tqdm import tqdm

def cal_avg_distance(adv_images, gt_images, attack_success_li):
    """
    计算原始图像和攻击成功的图像之间的 L2和无穷范数的平均值

    Args:
        adv_images ([type]): 所有对抗图像数据
        gt_images ([type]): 所有原始图像数据
        attack_success_li ([type]): 是否对抗成功列表

    Returns:
        [type]: [description]
    """
    k = 0
    s_inf = 0
    s_l2 = 0
    min_num = min(len(adv_images), len(attack_success_li))
    for i in range(min_num):
        success = attack_success_li[i]
        if not success:
            continue
        k += 1
        ## 将类型强转为 np.float32
        gt_img = gt_images[i].reshape(-1).astype(np.float32)
        adv_img = adv_images[i].reshape(-1).astype(np.float32)
        # s_inf += np.linalg.norm(gt_img-adv_img, np.inf) / np.linalg.norm(gt_img, np.inf)
        diff_img = np.abs(gt_img - adv_img)
        s_inf += np.max(diff_img) / np.max(np.abs(gt_img))
        s_l2 += np.linalg.norm(diff_img) / np.linalg.norm(gt_img)
    avg_inf = -1 if k == 0 else s_inf / float(k)
    avg_l2 = -1 if k == 0 else s_l2 / float(k)
    
    print(f'Average distance, l2: {avg_l2} Linf: {avg_inf}')
    return avg_inf, avg_l2

def get_prediction(model, x_test, batch_size=1):
    """
    批量化模型预测

    Args:
        model ([type]): [description]
        x_test ([type]): [description]
        batch_size (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    nrof_data = x_test.shape[0]
    assert nrof_data % batch_size == 0, 'Must divide by batch_size'
    y_pred = []
    nrof_batch = nrof_data // batch_size
    pbar = tqdm(range(batch_size), desc='Predicting ')
    for i in pbar:
        x_batch = x_test[i*batch_size:(i+1)*batch_size, :]
        preds = model.predict(x_batch)
        preds = [np.argmax(_) for _ in preds]
        y_pred += preds
    return y_pred

def get_common_attacks(all_attack_success_li):
    """
    找到所有算法都攻击成功的图片 index

    Args:
        all_attack_success_li ([type]): [description]
    """
    ## find common attacked successfully image id
    common_li = []
    nrof_max = min([len(_) for _ in all_attack_success_li])
    nrof_algo = len(all_attack_success_li)
    success_chosen = [li[:nrof_max] for li in all_attack_success_li]
    attack_success = np.asarray(success_chosen).reshape(nrof_algo, nrof_max)
    for i in range(nrof_max):
        success = attack_success[:, i] == 1
        if all(success):
            common_li.append(i)
    return common_li