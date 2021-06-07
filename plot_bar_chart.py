import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
# sns.set_theme(style="whitegrid")
rc = {'axes.facecolor': 'white',}
plt.rcParams.update(rc)

def draw_bar_chart():
    # color_table = ['#8b2be2',
    #                '#ff7c00',
    #                '#1ac938',
    #                '#023eff',
    #                '#e8000b',
    #                '#8b2be2',
    #                '#9f4800',
    #                '#f14cc1',
    #                '#a3a3a3',
    #                '#ffc400',
    #                '#00d7ff']
    color_table = ['#ff7c00', '#e8000b', '#1ac938', '#023eff', '#023eff',
                   '#9f4800', '#f14cc1',
                   '#a3a3a3', '#ffc400', '#00d7ff']
    # step_labels = ['1e5', '2e5', '3e5', '4e5', '5e5']
    step_labels = ['100k', '200k', '300k', '400k', '500k']

    ret_mean_cpm = [826.50, 917.50, 919, 946.00, 958.00]
    ret_mean_pm = [461.00, 835.50, 877.00, 918.00, 953.00]
    ret_mean_ac = [572.50, 875.50, 908.50, 901.50, 951.00]

    ret_mean_rad = [473.50, 857.00, 933.50, 942.50,955.00]

    ret_std_cpm = [209, 156, 151, 137, 137]
    ret_std_pm = [166, 152, 151, 148, 143]
    ret_std_ac = [246, 199, 166, 150, 149]
    ret_std_rad = [191, 189, 184, 161, 157]

    SHOW_STD = False

    ticks =  np.arange(len(step_labels))

    width = 0.2  # the width of the bars
    space = 0.01
    r1 = [x - 1.5 * width for x in ticks]
    r2 = [x - 0.5 * width for x in ticks]
    r3 = [x + 0.5 * width for x in ticks]
    r4 = [x + 1.5 * width for x in ticks]


    fig, ax = plt.subplots()
    # TODO: Bar configuration
    linewidth = 0.2
    alpha = 0.6


    if SHOW_STD:
        ax.bar(r1, ret_mean_cpm, yerr=ret_std_cpm, align='center', alpha=alpha, width=width, ecolor='black', capsize=5, label='CPM')
        ax.bar(r2, ret_mean_pm, yerr=ret_std_pm, align='center', alpha=alpha, width=width, ecolor='black', capsize=5, label='PM')
        ax.bar(r3, ret_mean_ac, yerr=ret_std_ac, align='center', alpha=alpha, width=width, ecolor='black', capsize=5, label='AC')
        ax.bar(r4, ret_mean_rad, yerr=ret_std_rad, align='center', alpha=alpha, width=width, ecolor='black', capsize=5, label='Scratch')
    else:
        rect1 = ax.bar(r1, ret_mean_ac, align='center', alpha=alpha, width=width - space, ecolor='black',
               capsize=5, label='RL+AC', color=color_table[0])
        rect2 = ax.bar(r2, ret_mean_pm, align='center', alpha=alpha, width=width- space, ecolor='black',
               capsize=5, label='RL+PM', color=color_table[1])
        rect3 = ax.bar(r3, ret_mean_rad, align='center', alpha=alpha, width=width- space,
               ecolor='black', capsize=5, label='RL Scratch', color=color_table[2])
        rect4 = ax.bar(r4, ret_mean_cpm, align='center', alpha=alpha, width=width - space,
               ecolor='black', capsize=5, label='RL+CPM', color=color_table[3])
    ax.set_ylabel('Evaluation Score')

    def autolabel(rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        xpos = xpos.lower()  # normalize the case of the parameter
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

        for rect in rects:
            height = math.ceil(rect.get_height())
            ax.text(rect.get_x() + rect.get_width() * offset[xpos], 1.01 * height,
                    '{}'.format(height), ha=ha[xpos], va='bottom', fontsize=8, rotation=45)

    # plt.xticks([r + 2 * width for r in range(len(ret_mean_cpm))], step_labels)
    plt.xticks(ticks, step_labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.tick_params(axis='x', top='off', bottom='off', left='off', right='off')
    # ax.set_title('Meadian Scores on DMControl over training ')
    plt.xlabel('Environment steps', color='k', fontsize=14)
    plt.ylabel('Episode Return', color='k', fontsize=14)
    plt.ylim(200, 1200)
    plt.xlim(- 2 * width - space, len(ticks) -1 + 2* width + space)

    autolabel(rect1)
    autolabel(rect2)
    autolabel(rect3)
    autolabel(rect4)

    # Save the figure and show
    plt.tight_layout()
    plt.legend(loc='upper right', fontsize=14, frameon=True,
                   facecolor='white', edgecolor='grey', ncol=2)
    plt.show()

def draw_bar_chart_v1():
    # color_table = ['#ff7c00', '#e8000b', '#1ac938', '#023eff', '#023eff',
    #                '#9f4800', '#f14cc1',
    #                '#a3a3a3', '#ffc400', '#00d7ff']
    color_table = ['#023eff', '#e8000b']
    # step_labels = ['1e5', '2e5', '3e5', '4e5', '5e5']
    step_labels = ['RL+CPM', 'RL+CURL', 'RL+AE', 'RL+PM', 'RL scratch']

    ret_mean_cpm=[826.50, 917.5, 958.00]
    ret_mean_ac=[572.50, 875.5, 949.50]
    ret_mean_ae=[546.5, 851.5, 934]
    ret_mean_pm=[461.00, 835.5, 951.50]
    ret_mean_rad=[473.50, 856.5, 955.00]


    ret_means_100 = [ret_mean_cpm[0], ret_mean_ac[0], ret_mean_ae[0], ret_mean_pm[0], ret_mean_rad[0]]
    ret_means_200 = [ret_mean_cpm[1], ret_mean_ac[1], ret_mean_ae[1], ret_mean_pm[1], ret_mean_rad[1]]
    ret_means_500 = [ret_mean_cpm[2], ret_mean_ac[2], ret_mean_ae[2], ret_mean_pm[2], ret_mean_rad[2]]

    ret_std_cpm = [209, 156, 151, 137, 137]
    ret_std_pm = [166, 152, 151, 148, 143]
    ret_std_ac = [246, 199, 166, 150, 149]
    ret_std_rad = [191, 189, 184, 161, 157]

    SHOW_STD = False

    ticks =  np.arange(len(step_labels))

    width = 0.3  # the width of the bars
    space = 0.01
    # r1 = [x - 1.5 * width for x in ticks]
    r2 = [x - 0.5 * width for x in ticks]
    r3 = [x + 0.5 * width for x in ticks]
    # r4 = [x + 1.5 * width for x in ticks]


    fig, ax = plt.subplots()
    # TODO: Bar configuration
    alpha = 0.6


    if SHOW_STD:
        # ax.bar(r1, ret_mean_cpm, yerr=ret_std_cpm, align='center', alpha=alpha, width=width, ecolor='black', capsize=5, label='CPM')
        ax.bar(r2, ret_mean_pm, yerr=ret_std_pm, align='center', alpha=alpha, width=width, ecolor='black', capsize=5, label='PM')
        ax.bar(r3, ret_mean_ac, yerr=ret_std_ac, align='center', alpha=alpha, width=width, ecolor='black', capsize=5, label='AC')
        # ax.bar(r4, ret_mean_rad, yerr=ret_std_rad, align='center', alpha=alpha, width=width, ecolor='black', capsize=5, label='Scratch')
    else:
        rect2 = ax.bar(r2, ret_means_100, align='center', alpha=alpha, width=width- space, ecolor='black',
               capsize=5, label='RL+PM', color=color_table[0])
        rect3 = ax.bar(r3, ret_means_200, align='center', alpha=alpha, width=width- space,
               ecolor='black', capsize=5, label='RL Scratch', color=color_table[1])
    ax.set_ylabel('Evaluation Score')

    def autolabel(rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        xpos = xpos.lower()  # normalize the case of the parameter
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

        for rect in rects:
            height = math.ceil(rect.get_height())
            ax.text(rect.get_x() + rect.get_width() * offset[xpos], 1.01 * height,
                    '{}'.format(height), ha=ha[xpos], va='bottom', fontsize=10, rotation=0)

    # plt.xticks([r + 2 * width for r in range(len(ret_mean_cpm))], step_labels)
    plt.xticks(ticks, step_labels, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tick_params(axis='x', top='off', bottom='off', left='off', right='off')
    # ax.set_title('Meadian Scores on DMControl over training ')
    plt.xlabel('Environment steps', color='k', fontsize=12)
    plt.ylabel('Episode Return', color='k', fontsize=12)
    plt.ylim(200, 1090)
    plt.xlim(- 2 * width - space, len(ticks) -1 + 2* width + space)

    # autolabel(rect1)
    autolabel(rect2)
    autolabel(rect3)
    # autolabel(rect4)

    # Save the figure and show
    plt.tight_layout()
    plt.legend(['100k env steps', '200k env steps'], loc='upper right', fontsize=14, frameon=True,
                   facecolor='white', edgecolor='grey', ncol=1)
    plt.show()

def draw_legends():
    import pylab
    # color_table = ['k', '#ff7c00', '#e8000b', '#1ac938', '#8b2be2', '#023eff', '#9f4800', '#f14cc1',
    #                '#a3a3a3', '#ffc400', '#00d7ff']
    color_table = ['#1ac938', '#ff7c00', '#e8000b', '#9f4800', '#8b2be2', '#023eff', '#f14cc1', '#a3a3a3', '#ffc400', '#00d7ff']
    x = np.arange(10)
    figData = pylab.figure()
    ax = pylab.gca()
    legends = ['RL scratch', 'RL+CURL', 'RL+AE', 'RL+PM', 'RL+CPM frozen', 'RL+CPM finetune']

    for i in range(6):
        if i == 0:
            pylab.plot(x, x * (i + 1), label=legends[i], color=color_table[i], linestyle='-')
        elif i == 4 or i == 5:
            pylab.plot(x, x * (i + 1), label=legends[i], color=color_table[i], linewidth=2)
        else:
            pylab.plot(x, x * (i + 1), label=legends[i], color=color_table[i])

    # create a second figure for the legend
    figlegend = pylab.figure(figsize=(5, 2))

    # produce a legend for the objects in the other figure
    pylab.figlegend(*ax.get_legend_handles_labels(), loc='center', ncol=3)
    # lines = ax.plot(range(10), pylab.randn(10),
    #                 range(10), pylab.randn(10),
    #                 range(10), pylab.randn(10),
    #                 range(10), pylab.randn(10),
    #                 range(10), pylab.randn(10),
    #                 range(10), pylab.randn(10))
    # figlegend.legend(lines, ('state', 'RL+AC', 'RL+PM', 'RL scratch', 'RL+CPM frozen', 'RL+CPM finetune'), 'center',
    #                  ncol=3)
    # fig.show()
    # figlegend.show()
    figlegend.savefig('legend.pdf')

def draw_legends_ablation():
    import pylab
    color_table = ['#1ac938', '#9f4800', '#ff7c00', '#023eff']
    x = np.arange(10)
    figData = pylab.figure()
    ax = pylab.gca()
    legends = ['RL scratch', 'RL+CFDM', 'RL+IDM', 'RL+CPM']

    for i in range(4):
        if i == 3:
            pylab.plot(x, x * (i + 1), label=legends[i], color=color_table[i], linewidth=2)
        else:
            pylab.plot(x, x * (i + 1), label=legends[i], color=color_table[i])

    # create a second figure for the legend
    figlegend = pylab.figure(figsize=(10, 2))

    # produce a legend for the objects in the other figure
    pylab.figlegend(*ax.get_legend_handles_labels(), loc='center', ncol=4)
    figlegend.savefig('legend.pdf')

def draw_legends_multitasks():
    import pylab
    color_table = ['#1ac938', '#8b2be2', '#023eff']
    x = np.arange(10)
    figData = pylab.figure()
    ax = pylab.gca()
    legends = ['RL scratch', 'RL+CPM frozen', 'RL+CPM finetune']

    for i in range(3):
        if i == 3:
            pylab.plot(x, x * (i + 1), label=legends[i], color=color_table[i], linewidth=2)
        else:
            pylab.plot(x, x * (i + 1), label=legends[i], color=color_table[i])

    # create a second figure for the legend
    figlegend = pylab.figure(figsize=(5, 2))

    # produce a legend for the objects in the other figure
    pylab.figlegend(*ax.get_legend_handles_labels(), loc='center', ncol=3)
    figlegend.savefig('legend.pdf')

if __name__ == '__main__':
    # draw_bar_chart()
    draw_bar_chart_v1()
    # draw_legends()
    # draw_legends_ablation()
    # draw_legends_multitasks()
