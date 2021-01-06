import os
import os.path as osp
import sys
import argparse

from imgntWrapper import x_test, y_test ,mymodel

CURRENT_DIR = osp.dirname(__file__)
sys.path.insert(0, osp.join('..'))
from utils import *


def main(args):
    class_names = IMAGENET_CLASSES

    assert osp.exists(args.input), f'Directory not exists: {args.input}'
    if args.output is None:
        args.output = args.input
    os.makedirs(args.output, exist_ok=True)
    data_dir = args.input
    output_dir = args.output
    ### 下面的文件路径变量需要改成合适的值
    qlnes_dir = data_dir + 'QLNES/2020-12-24/'
    bandits_dir = data_dir + 'Bandits/2020-12-24/'
    simba_dir = data_dir + 'SimBA/2020-12-24/'
    parsi_dir = data_dir + 'Parsimonious/2020-12-22/'
    ds_dir = data_dir + 'DSBatched/2020-12-22/'
    dsref_dir = data_dir + 'DSRefine/2020-12-22/'
    input_dirs = [qlnes_dir, bandits_dir, simba_dir, parsi_dir, ds_dir, dsref_dir]
    algo_names = ['QL-NES', 'Bandits', 'SimBA', 'Parsimonious', 'DeepSearch', 'DSRefine']
    batch_sizes = [10, 40, 1, 1, 1, 1]
    # algo_names = algo_names[:2]
    # input_dirs = input_dirs[:2]
    # batch_sizes = batch_sizes[:2]

    # 1.plot time bar plot
    time_li = [39, 30, 68, 3, 2.5, 34] # hour
    time_bar_path = osp.join(output_dir, 'time-bar.png')
    # plot_time(time_li, algo_names, time_bar_path)

    # 2. plot ds attack images
    ds_example_dir = osp.join(output_dir, 'ds_examples/')
    os.makedirs(ds_example_dir, exist_ok=True)
    # plot_deepsearch_attack(mymodel, ds_dir, dsref_dir, x_test, y_test, 
    #         class_names, ds_example_dir, topk=20)

    ## process_excel
    # 3. 获取对抗数据
    excel_data = []
    attack_success_li = []
    attack_query_li = []
    attack_images_li = []
    nrof_algo = len(algo_names)
    for i in range(nrof_algo):
        print(f'---- process {algo_names[i]}  {i+1}/{nrof_algo} -----')
        dir = input_dirs[i]
        batch_size = batch_sizes[i]
        atk_images = get_images_from_pkl(dir, batch_size)
        attack_images_li.append(atk_images)
        pkl_path = os.path.join(dir, 'data.pkl')
        result = parse_result_from_pkl(pkl_path)
        success_li, query_li, success_rate, query_avg, query_med = result
        attack_success_li.append(success_li)
        attack_query_li.append(query_li)
        avg_inf, avg_l2 = cal_avg_distance(atk_images, x_test, success_li)
        li = [success_rate, avg_inf, avg_l2, query_avg, query_med]
        excel_data.append(li)

    # 3. save to excel
    output_excel = osp.join(output_dir, 'table-result.xlsx')
    column_names = ['Success Rate', 'Avg. LInf', 'Avg. L2', 'Avg. queries', 'Med. queries']
    df_exc = pd.DataFrame(excel_data, columns=column_names, index=algo_names)
    df_exc.to_excel(output_excel)
    print(df_exc.head())

    ## 4. boxplot 
    boxplot_path = osp.join(output_dir, 'query-boxplot.png')
    boxplot(attack_query_li, labels=algo_names, save_path=boxplot_path)


    ## 5. plot compare multi algo
    multi_algo_save_dir = osp.join(output_dir, 'algos_examples/')
    os.makedirs(multi_algo_save_dir, exist_ok=True)
    plot_all_attack(mymodel, attack_images_li, attack_success_li, algo_names, x_test, y_test, 
            class_names, save_dir=multi_algo_save_dir, topk=10)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, 
        help='Input directory that has pkl files.')
    parser.add_argument('-o', '--output', type=str, default=None, 
        help='Output directory to save results')
    parser.add_argument('-d', '--debug', action='store_true',
        help='Whether to debug.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
