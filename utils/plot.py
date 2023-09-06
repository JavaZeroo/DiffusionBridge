import numpy as np
from matplotlib import pyplot as plt

def plot_bridge(ts, bridge, source_sample, target_sample, show_rate=1.0, show_gt=False, title=r'Gaussian to 2$\times$Gaussian'):
    if show_rate < 1.0:
        indices = np.arange(len(source_sample))

        # 打乱索引数组
        np.random.shuffle(indices)

        # 重新排列数组
        source_sample = source_sample[indices]
        target_sample = target_sample[indices]
        bridge = bridge[:, indices]
        
        source_sample = source_sample[:int(len(source_sample) * show_rate)]
        target_sample = target_sample[:int(len(target_sample) * show_rate)]
        bridge = bridge[:, :int(bridge.shape[-1] * show_rate)]
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax.scatter(np.zeros_like(source_sample), source_sample, c='r', alpha=0.8, label='Source', s=10)
    ax.scatter(np.ones_like(target_sample), target_sample, c='g', alpha=0.8, label='Target', s=10)
    # ax.plot(ts, bridge, c='gray', alpha=0.3)
    
    for i in range(bridge.shape[1]):
        line_color = 'orange' if bridge[-1, i] > 0 else 'blue'
        plt.plot(ts, bridge[:, i], color=line_color, alpha=0.3)
    
    if show_gt:
        # 计算 X1, Y1, X2, Y2
        X1 = np.zeros_like(source_sample)
        Y1 = source_sample
        X2 = np.ones_like(target_sample)
        Y2 = target_sample

        for i in range(X1.shape[0]):
            plt.plot([X1[i, 0], X2[i, 0]], [Y1[i, 0], Y2[i, 0]], color='gray',alpha=0.3)

         
    ax.set_title(title)
    ax.legend()
    fig.show()
    return fig, ax

def plot_t(traj):
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    ax.scatter(*traj[:, 0, :].T, c='r', alpha=0.8, label='Source', s=10)
    ax.scatter(*traj[:, -1, :].T, c='b', alpha=0.8, label='Target', s=10)
    for path in traj:
        ax.plot(*path[:, :].T, c='grey', alpha=0.2)
    
    fig.legend()
    return fig, ax