import matplotlib.pyplot as plt
import json
import seaborn as sns
# plt.rcParams.update({'font.size': 22})
from matplotlib.ticker import FormatStrFormatter

# sns.set_theme()
import seaborn as sns
import numpy as np
from matplotlib import rcParams
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.color_palette("tab10")
plt.rcParams['text.usetex'] = True

sns.set_theme(style="ticks", palette="tab10", font_scale=1.7, rc=custom_params)

def find_frontier(data, higher_better=True):
    sorted_data = {k: v for k, v in sorted(data.items(), key=lambda item: item[1])}
    candidate_idx = []
    last_flops = 0
    for cfg_id, values in sorted_data.items():
        flops, score = values
        if abs(last_flops - flops) > 10:
            candidate_idx.append(cfg_id)
            last_flops = flops
        else:
            if higher_better:
                if score > data[candidate_idx[-1]][1]:
                    candidate_idx[-1] = cfg_id
            else:
                if score < data[candidate_idx[-1]][1]:
                    candidate_idx[-1] = cfg_id
    return candidate_idx


def plot_base_large():
    with open('dpt_deit3_b_stitch_l_nyu.json', 'r') as f:
        data = json.load(f)

    with open('model_flops/flops_dpt_deit3_b_stitch_l_nyu.json', 'r') as f:
        flops_params = json.load(f)

    name_map = {
        'a1': 'δ>1.25',
        'a2': 'δ>1.25^2',
        'a3': 'δ>1.25^3',
    }

    small_res = [0.7799, 0.9541, 0.9874, 0.1678, 0.5131, 0.0649]
    base_res = [0.8321, 0.9729, 0.9931, 0.1488, 0.4455, 0.0586]
    large_res = [0.8821, 0.9874, 0.9979, 0.1292, 0.3873, 0.0518]

    anchor_flops = [197230172160, 292347752448, 602608435200]

    metrics = list(data['0']['metric'].keys())[:6]
    total_res_dict = {}
    for met in metrics:
        met_res = {}
        for cfg_id, res in data.items():
            met_res[int(cfg_id)] = res['metric'][met]
        total_res_dict[met] = met_res

    flops_res = {int(k): item/1e9 for k, item in flops_params.items()}
    fig, axs = plt.subplots(1, 6, figsize=(20, 4), dpi=300)

    # base large
    baseline_flops = [292347752448/1e9, 602608435200/1e9]

    for i, met in enumerate(metrics):
        # flops_res = {}
        # eval_res = {}
        baseline_mIoU = [base_res[i], large_res[i]]
        sns.scatterplot(x=baseline_flops, y=baseline_mIoU, marker='*', s=500, color='#fca311', ax=axs[i])

        sns.scatterplot(x=flops_res, y=total_res_dict[met], alpha=0.3, ax=axs[i])

        total_data = {}
        for cfg_id, cfg_f in flops_params.items():
            total_data[int(cfg_id)] = [cfg_f/1e9, total_res_dict[met][int(cfg_id)]]


        high_better = i <= 2
        frontier_indices = find_frontier(total_data, higher_better=high_better)
        frontier_flops = [flops_res[idx] for idx in frontier_indices]
        frontier_mious = [total_res_dict[met][idx] for idx in frontier_indices]

        sns.lineplot(x=frontier_flops, y=frontier_mious, marker='o', markersize=9, ax=axs[i])
        axs[i].set_xlabel('FLOPs (G)')

        if met == 'a1':
            axs[i].set_title(r'$\delta>1.25$')
        elif met == 'a2':
            axs[i].set_title(r'$\delta>1.25^2$')
        elif met == 'a3':
            axs[i].set_title(r'$\delta>1.25^3$')
        elif met == 'abs_rel':
            axs[i].set_title('AbsRel')
        elif met == 'rmse':
            axs[i].set_title('RMSE')
        elif met == 'log_10':
            axs[i].set_title('log10')


        axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        axs[i].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.35)
    # plt.title(f'{metric}')
    # plt.show()
    plt.savefig('figures/b_l_res.pdf')


def plot_small_large():
    with open('dpt_deit3_s_stitch_l_nyu_4.json', 'r') as f:
        data = json.load(f)

    with open('model_flops/flops_dpt_deit3_s_stitch_l_nyu.json', 'r') as f:
        flops_params = json.load(f)

    name_map = {
        'a1': 'δ>1.25',
        'a2': 'δ>1.25^2',
        'a3': 'δ>1.25^3',
    }

    small_res = [0.7799, 0.9541, 0.9874, 0.1678, 0.5131, 0.0649]
    base_res = [0.8321, 0.9729, 0.9931, 0.1488, 0.4455, 0.0586]
    large_res = [0.8821, 0.9874, 0.9979, 0.1292, 0.3873, 0.0518]

    anchor_flops = [197230172160, 292347752448, 602608435200]

    metrics = list(data['0']['metric'].keys())[:6]
    total_res_dict = {}
    for met in metrics:
        met_res = {}
        for cfg_id, res in data.items():
            met_res[int(cfg_id)] = res['metric'][met]
        total_res_dict[met] = met_res

    flops_res = {int(k): item/1e9 for k, item in flops_params.items()}
    fig, axs = plt.subplots(1, 6, figsize=(20, 4), dpi=300)

    # base large
    baseline_flops = [197230172160/1e9, 602608435200/1e9]

    for i, met in enumerate(metrics):
        # flops_res = {}
        # eval_res = {}
        baseline_mIoU = [small_res[i], large_res[i]]
        sns.scatterplot(x=baseline_flops, y=baseline_mIoU, marker='*', s=500, color='#fca311', ax=axs[i])

        sns.scatterplot(x=flops_res, y=total_res_dict[met], alpha=0.3, ax=axs[i])

        total_data = {}
        for cfg_id, cfg_f in flops_params.items():
            total_data[int(cfg_id)] = [cfg_f/1e9, total_res_dict[met][int(cfg_id)]]


        high_better = i <= 2
        frontier_indices = find_frontier(total_data, higher_better=high_better)
        frontier_flops = [flops_res[idx] for idx in frontier_indices]
        frontier_mious = [total_res_dict[met][idx] for idx in frontier_indices]

        sns.lineplot(x=frontier_flops, y=frontier_mious, marker='o', markersize=9, ax=axs[i])
        axs[i].set_xlabel('FLOPs (G)')

        if met == 'a1':
            axs[i].set_title(r'$\delta>1.25$')
        elif met == 'a2':
            axs[i].set_title(r'$\delta>1.25^2$')
        elif met == 'a3':
            axs[i].set_title(r'$\delta>1.25^3$')
        elif met == 'abs_rel':
            axs[i].set_title('AbsRel')
        elif met == 'rmse':
            axs[i].set_title('RMSE')
        elif met == 'log_10':
            axs[i].set_title('log10')


        axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        axs[i].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.35)
    # plt.title(f'{metric}')
    # plt.show()
    plt.savefig('figures/s_l_res.pdf')

if __name__ == '__main__':
    plot_base_large()
    # plot_small_large()