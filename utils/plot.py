import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import shutil
from rich.progress import track
import imageio


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


def save_gif_frame(bridge, save_path=None, save_name=None, bound=15):
    assert save_path is not None, "save_path cannot be None"
    save_path = Path(save_path)
    if not isinstance(bridge, np.ndarray):
        try:
            bridge = bridge.numpy()
        except:
            raise TypeError("bridge must be numpy.ndarray")

    bridge = bridge.transpose((1, 0, 2))
    bridge = bridge[::10, :, :]  if bridge.shape[0] >= 200 else bridge # 降低采样率
    
    temp_dir = save_path / 'temp'
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True)
    frame = 0
    
    color_map = -np.sqrt(bridge[0, :, 0]**2 + bridge[0, :, 1]**2)
    for i in track(range(bridge.shape[0]), description="Processing image"):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        x = bridge[i, :, 0]  # 注意：
        y = bridge[i, :, 1]  # 注意：
        
        ax.scatter(x, y, c=color_map, alpha=1, s=10)
        fig.savefig(save_path / 'temp' / f'{frame:03d}.png', dpi=100)
        frame += 1
        fig.show()
        plt.close('all')
    frames = []
    for i in range(bridge.shape[0]):
        frame_image = imageio.imread(save_path / 'temp' / f'{i:03d}.png')
        frames.append(frame_image)
    imageio.mimsave(save_path/save_name, frames, duration=0.2)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        
def save_gif_traj(bridge, save_path=None, save_name=None, bound=15):
    assert save_path is not None, "save_path cannot be None"
    save_path = Path(save_path)
    if not isinstance(bridge, np.ndarray):
        try:
            bridge = bridge.numpy()
        except:
            raise TypeError("bridge must be numpy.ndarray")

    bridge = bridge.transpose((1, 0, 2))
    bridge = bridge[::10, :, :]  if bridge.shape[0] >= 200 else bridge # 降低采样率
    
    temp_dir = save_path / 'temp'
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True)
    frame = 0
    
    color_map = -np.sqrt(bridge[0, :, 0]**2 + bridge[0, :, 1]**2)
    for i in track(range(bridge.shape[0]), description="Processing image"):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        x = bridge[i, :, 0]  # 注意：
        y = bridge[i, :, 1]  # 注意：
        
        
        for s in range(bridge.shape[1]):
            ax.plot(bridge[:i,s, 0], bridge[:i,s, 1], c='gray', alpha=0.7, lw=0.2)
        
        ax.scatter(x, y, c=color_map, alpha=1, s=10)
        
        fig.savefig(save_path / 'temp' / f'{frame:03d}.png', dpi=100)
        frame += 1
        fig.show()
        plt.close('all')
    frames = []
    for i in range(bridge.shape[0]):
        frame_image = imageio.imread(save_path / 'temp' / f'{i:03d}.png')
        frames.append(frame_image)
    imageio.mimsave(save_path/save_name, frames, duration=0.2)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)