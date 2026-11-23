"""
PrintShop_EEG_Synthesizer: 信噪比加权与批量导出模块
==================================================

核心功能：
1. 动态SNR计算与噪声加权
2. 多噪声源混合
3. 标准化数据集生成
4. 批量导出（按SNR分文件夹保存）

Author: PrintShop_EEG_Synthesizer Team
"""

import os
import numpy as np
import scipy.io as scio
from typing import Optional, Dict, List, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from tqdm import tqdm
import warnings

from ..noise_generators import (
    PLINoiseGenerator,
    BaselineWanderGenerator,
    MechanicalShockGenerator,
    CompositeNoiseGenerator,
    create_noise_generator
)
from ..artifact_loader import ArtifactLoader


class SNRMode(Enum):
    """SNR计算模式"""
    POWER_RATIO = "power_ratio"      # 功率比 SNR = 10 * log10(P_signal / P_noise)
    AMPLITUDE_RATIO = "amplitude"   # 幅度比 SNR = 20 * log10(A_signal / A_noise)
    MANUAL = "manual"               # 手动指定噪声幅度


@dataclass
class SynthesizerConfig:
    """合成器配置"""
    # 目标采样率
    sampling_rate: int = 128

    # SNR设置
    snr_mode: SNRMode = SNRMode.POWER_RATIO
    snr_values_db: List[float] = field(default_factory=lambda: [0, -5, -10])  # dB

    # 噪声源开关与权重
    enable_pli: bool = True
    enable_bw: bool = True
    enable_shock: bool = True
    enable_emg: bool = True
    enable_eog: bool = True

    # 噪声幅度权重
    pli_weight: float = 1.0
    bw_weight: float = 1.0
    shock_weight: float = 1.0
    emg_weight: float = 1.0
    eog_weight: float = 1.0

    # 物理噪声参数
    pli_frequency: float = 50.0
    pli_amplitude: float = 1.0
    bw_amplitude: float = 0.5
    shock_amplitude: float = 2.0

    # 随机种子
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        d = self.__dict__.copy()
        d['snr_mode'] = self.snr_mode.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SynthesizerConfig':
        """从字典创建"""
        if isinstance(d.get('snr_mode'), str):
            d['snr_mode'] = SNRMode(d['snr_mode'])
        return cls(**d)


class EEGSynthesizer:
    """
    EEG噪声合成器

    将清洁EEG信号与合成的工业噪声混合，生成标准化的含噪数据集。

    核心流程:
    1. 加载/生成噪声信号
    2. 计算目标SNR下的噪声幅度
    3. 混合信号与噪声
    4. 保存标准化数据集
    """

    def __init__(self, config: Optional[SynthesizerConfig] = None,
                 artifact_loader: Optional[ArtifactLoader] = None):
        """
        Args:
            config: 合成器配置
            artifact_loader: 伪影加载器
        """
        self.config = config or SynthesizerConfig()
        self.rng = np.random.default_rng(self.config.seed)

        # 初始化噪声生成器
        self._init_noise_generators()

        # 初始化伪影加载器
        self.artifact_loader = artifact_loader or ArtifactLoader(
            target_sampling_rate=self.config.sampling_rate
        )

    def _init_noise_generators(self):
        """初始化噪声生成器"""
        self.pli_gen = PLINoiseGenerator(
            sampling_rate=self.config.sampling_rate,
            frequency=self.config.pli_frequency,
            amplitude=self.config.pli_amplitude,
            seed=self.config.seed
        )

        self.bw_gen = BaselineWanderGenerator(
            sampling_rate=self.config.sampling_rate,
            amplitude=self.config.bw_amplitude,
            seed=self.config.seed
        )

        self.shock_gen = MechanicalShockGenerator(
            sampling_rate=self.config.sampling_rate,
            amplitude=self.config.shock_amplitude,
            seed=self.config.seed
        )

        self.composite_gen = CompositeNoiseGenerator(
            sampling_rate=self.config.sampling_rate,
            pli_generator=self.pli_gen,
            bw_generator=self.bw_gen,
            shock_generator=self.shock_gen,
            seed=self.config.seed
        )

        # 设置权重
        self.composite_gen.set_weights(
            pli=self.config.pli_weight,
            bw=self.config.bw_weight,
            shock=self.config.shock_weight
        )

    def compute_noise_amplitude_for_snr(self, signal: np.ndarray,
                                        target_snr_db: float) -> float:
        """
        计算达到目标SNR所需的噪声幅度

        Args:
            signal: 清洁信号 (n_channels, n_samples) 或 (n_samples,)
            target_snr_db: 目标SNR (dB)

        Returns:
            需要的噪声幅度
        """
        # 计算信号功率
        if signal.ndim == 2:
            signal_power = np.mean(signal ** 2)
        else:
            signal_power = np.mean(signal ** 2)

        if signal_power < 1e-10:
            warnings.warn("信号功率过小，可能导致SNR计算不准确")
            signal_power = 1e-10

        # 计算噪声幅度
        # SNR_db = 10 * log10(P_signal / P_noise)
        # => P_noise = P_signal / 10^(SNR_db/10)
        # => A_noise = sqrt(P_noise / n_samples) * sqrt(n_samples) = sqrt(P_noise)
        noise_power = signal_power / (10 ** (target_snr_db / 10))
        noise_amplitude = np.sqrt(noise_power)

        return noise_amplitude

    def add_physiological_noise(self, signal: np.ndarray,
                               emg_weight: float = 1.0,
                               eog_weight: float = 1.0) -> np.ndarray:
        """
        添加生理伪影（EMG/EOG）

        Args:
            signal: 原始EEG信号
            emg_weight: EMG权重
            eog_weight: EOG权重

        Returns:
            叠加生理伪影后的信号
        """
        n_channels, n_samples = signal.shape

        # 添加EMG
        if self.config.enable_emg:
            try:
                emg = self.artifact_loader.get_random_emg_segment(
                    n_samples, n_channels, seed=self.rng.integers(0, 2**31)
                )
                if emg.shape != signal.shape:
                    emg = self._adjust_array_shape(emg, signal.shape)
                # 归一化EMG到与EEG相近的量级
                emg = emg * np.std(signal) / (np.std(emg) + 1e-10) * emg_weight
                signal = signal + emg
            except Exception as e:
                warnings.warn(f"添加EMG失败: {e}")

        # 添加EOG
        if self.config.enable_eog:
            try:
                eog = self.artifact_loader.get_random_eog_segment(
                    n_samples, n_channels, seed=self.rng.integers(0, 2**31)
                )
                if eog.shape != signal.shape:
                    eog = self._adjust_array_shape(eog, signal.shape)
                eog = eog * np.std(signal) / (np.std(eog) + 1e-10) * eog_weight
                signal = signal + eog
            except Exception as e:
                warnings.warn(f"添加EOG失败: {e}")

        return signal

    def _adjust_array_shape(self, arr: np.ndarray,
                           target_shape: Tuple[int, ...]) -> np.ndarray:
        """调整数组形状以匹配目标"""
        if arr.shape == target_shape:
            return arr

        result = np.zeros(target_shape)
        min_shape = tuple(min(a, b) for a, b in zip(arr.shape, target_shape))

        if arr.ndim == 1:
            result[:min_shape[0]] = arr[:min_shape[0]]
        elif arr.ndim == 2:
            result[:min_shape[0], :min_shape[1]] = arr[:min_shape[0], :min_shape[1]]

        return result

    def synthesize_single(self, clean_eeg: np.ndarray,
                        target_snr_db: float,
                        return_components: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        合成单个EEG样本

        Args:
            clean_eeg: 清洁EEG信号 (n_channels, n_samples)
            target_snr_db: 目标SNR (dB)
            return_components: 是否返回噪声分量

        Returns:
            含噪EEG信号，或 (含噪EEG, 噪声分量字典)
        """
        if clean_eeg.ndim != 2:
            raise ValueError("clean_eeg必须是2D数组 (n_channels, n_samples)")

        n_channels, n_samples = clean_eeg.shape

        # 生成物理噪声
        physical_noise = np.zeros((n_channels, n_samples))

        if self.config.enable_pli:
            pli = self.pli_gen.generate_multi_channel(
                n_samples, n_channels, duration=None
            )
            physical_noise += self.config.pli_weight * pli

        if self.config.enable_bw:
            for ch in range(n_channels):
                bw = self.bw_gen.generate(n_samples)
                physical_noise[ch] += self.config.bw_weight * bw

        if self.config.enable_shock:
            shock = self.shock_gen.generate_multi_channel(
                n_samples, n_channels, duration=None
            )
            physical_noise += self.config.shock_weight * shock

        # 计算物理噪声幅度并缩放
        physical_noise_power = np.mean(physical_noise ** 2)
        if physical_noise_power > 1e-10:
            physical_noise *= self.compute_noise_amplitude_for_snr(
                clean_eeg, target_snr_db
            ) / np.sqrt(physical_noise_power)

        # 添加生理伪影
        noisy_eeg = clean_eeg.copy()
        if self.config.enable_emg or self.config.enable_eog:
            noisy_eeg = self.add_physiological_noise(
                noisy_eeg,
                emg_weight=self.config.emg_weight,
                eog_weight=self.config.eog_weight
            )

        # 叠加物理噪声
        noisy_eeg = noisy_eeg + physical_noise

        if return_components:
            components = {
                'physical_noise': physical_noise,
                'pli': pli if self.config.enable_pli else None,
                'bw': physical_noise if self.config.enable_bw else None,
                'shock': physical_noise if self.config.enable_shock else None,
            }
            return noisy_eeg, components

        return noisy_eeg

    def synthesize_batch(self, clean_eegs: np.ndarray,
                        labels: np.ndarray,
                        snr_values: Optional[List[float]] = None,
                        subject_ids: Optional[np.ndarray] = None,
                        show_progress: bool = True) -> Dict[str, np.ndarray]:
        """
        批量合成含噪EEG数据

        Args:
            clean_eegs: 清洁EEG数据 (N, n_channels, n_samples)
            labels: 标签 (N,)
            snr_values: SNR值列表，如果为None则使用配置中的值
            subject_ids: 被试ID (N,)，用于LOSO验证
            show_progress: 是否显示进度条

        Returns:
            字典，包含各SNR级别下的数据
        """
        if snr_values is None:
            snr_values = self.config.snr_values_db

        n_samples = clean_eegs.shape[0]
        results = {}

        iterator = range(n_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="合成含噪数据")

        for snr_db in snr_values:
            noisy_data = []
            noisy_labels = []
            noisy_subjects = []

            for i in iterator:
                clean = clean_eegs[i]  # (n_channels, n_samples)
                noisy = self.synthesize_single(clean, snr_db)
                noisy_data.append(noisy)
                noisy_labels.append(labels[i])

                if subject_ids is not None:
                    noisy_subjects.append(subject_ids[i])

            results[f'snr_{snr_db}dB'] = {
                'X': np.array(noisy_data),
                'y': np.array(noisy_labels),
                'subject_ids': np.array(noisy_subjects) if subject_ids is not None else None,
                'snr_db': snr_db
            }

        return results

    def calculate_actual_snr(self, clean: np.ndarray, noisy: np.ndarray) -> float:
        """
        计算实际SNR

        Args:
            clean: 清洁信号
            noisy: 含噪信号

        Returns:
            实际SNR (dB)
        """
        noise = noisy - clean
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean(noise ** 2)

        if noise_power < 1e-10:
            return float('inf')

        snr_db = 10 * np.log10(signal_power / noise_power)
        return snr_db

    def save_dataset(self, results: Dict[str, np.ndarray],
                    output_dir: str,
                    dataset_name: str = "noisy_eeg_dataset",
                    format: str = "numpy") -> Dict[str, str]:
        """
        保存合成数据集

        Args:
            results: synthesize_batch返回的结果
            output_dir: 输出目录
            dataset_name: 数据集名称
            format: 保存格式 ('numpy', 'matlab', 'both')

        Returns:
            保存的文件路径字典
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}

        for snr_key, data in results.items():
            snr_dir = os.path.join(output_dir, snr_key)
            os.makedirs(snr_dir, exist_ok=True)

            # 保存numpy格式
            if format in ['numpy', 'both']:
                X_path = os.path.join(snr_dir, f'{dataset_name}_X.npy')
                y_path = os.path.join(snr_dir, f'{dataset_name}_y.npy')
                np.save(X_path, data['X'].astype(np.float32))
                np.save(y_path, data['y'])
                saved_files[f'{snr_key}_X'] = X_path
                saved_files[f'{snr_key}_y'] = y_path

                if data['subject_ids'] is not None:
                    subj_path = os.path.join(snr_dir, f'{dataset_name}_subjects.npy')
                    np.save(subj_path, data['subject_ids'])
                    saved_files[f'{snr_key}_subjects'] = subj_path

            # 保存matlab格式
            if format in ['matlab', 'both']:
                mat_path = os.path.join(snr_dir, f'{dataset_name}_{snr_key}.mat')
                scio.savemat(mat_path, {
                    'X': data['X'].astype(np.float32),
                    'y': data['y'],
                    'subject_ids': data['subject_ids'] if data['subject_ids'] is not None else np.array([]),
                    'snr_db': data['snr_db']
                })
                saved_files[f'{snr_key}_mat'] = mat_path

        # 保存配置和元数据
        config_path = os.path.join(output_dir, 'synthesizer_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        saved_files['config'] = config_path

        # 保存数据集统计信息
        stats = self._compute_dataset_stats(results)
        stats_path = os.path.join(output_dir, 'dataset_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        saved_files['statistics'] = stats_path

        return saved_files

    def _compute_dataset_stats(self, results: Dict[str, np.ndarray]) -> Dict:
        """计算数据集统计信息"""
        stats = {}

        for snr_key, data in results.items():
            snr_stats = {
                'n_samples': len(data['y']),
                'snr_db': data['snr_db'],
                'class_distribution': {},
                'x_mean': float(np.mean(data['X'])),
                'x_std': float(np.std(data['X'])),
                'x_min': float(np.min(data['X'])),
                'x_max': float(np.max(data['X'])),
            }

            # 类别分布
            unique, counts = np.unique(data['y'], return_counts=True)
            for label, count in zip(unique, counts):
                snr_stats['class_distribution'][int(label)] = int(count)

            # 被试分布
            if data['subject_ids'] is not None:
                unique_subj, subj_counts = np.unique(data['subject_ids'], return_counts=True)
                snr_stats['n_subjects'] = len(unique_subj)
                snr_stats['samples_per_subject'] = {
                    int(s): int(c) for s, c in zip(unique_subj, subj_counts)
                }

            stats[snr_key] = snr_stats

        return stats


def create_synthesizer_from_config(config_dict: Dict[str, Any]) -> EEGSynthesizer:
    """
    从配置字典创建合成器

    Args:
        config_dict: 配置字典

    Returns:
        EEGSynthesizer实例
    """
    config = SynthesizerConfig.from_dict(config_dict)
    return EEGSynthesizer(config)


def load_existing_dataset(data_dir: str,
                         snr: str = "snr_0dB") -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    加载已存在的合成数据集

    Args:
        data_dir: 数据集根目录
        snr: SNR目录名

    Returns:
        (X, y, subjects) 元组
    """
    import glob

    X_files = glob.glob(os.path.join(data_dir, snr, "*_X.npy"))
    if not X_files:
        raise FileNotFoundError(f"在 {data_dir}/{snr} 中未找到数据文件")

    X_path = X_files[0]
    base_path = os.path.splitext(X_path)[0].replace('_X', '')

    X = np.load(X_path)
    y = np.load(f"{base_path}_y.npy")

    subj_path = f"{base_path}_subjects.npy"
    subjects = np.load(subj_path) if os.path.exists(subj_path) else None

    return X, y, subjects


def merge_with_clean_data(clean_data_dir: str,
                          noisy_data_dir: str,
                          output_dir: str,
                          snr_values: List[float] = [0, -5, -10]) -> Dict[str, str]:
    """
    将清洁数据与含噪数据合并，创建完整的数据集矩阵

    Args:
        clean_data_dir: 清洁数据目录
        noisy_data_dir: 含噪数据目录
        output_dir: 输出目录
        snr_values: SNR值列表

    Returns:
        保存的文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = {}

    # 加载清洁数据
    clean_X = np.load(os.path.join(clean_data_dir, "seed_bde_X.npy"))
    clean_y = np.load(os.path.join(clean_data_dir, "seed_bde_y.npy"))
    clean_subj = np.load(os.path.join(clean_data_dir, "seed_bde_subjects.npy"))

    # 保存清洁数据
    clean_dir = os.path.join(output_dir, "clean")
    os.makedirs(clean_dir, exist_ok=True)
    np.save(os.path.join(clean_dir, "X.npy"), clean_X)
    np.save(os.path.join(clean_dir, "y.npy"), clean_y)
    np.save(os.path.join(clean_dir, "subjects.npy"), clean_subj)
    saved['clean'] = clean_dir

    # 加载并复制含噪数据
    for snr in snr_values:
        snr_dir = os.path.join(noisy_data_dir, f"snr_{snr}dB")
        if os.path.exists(snr_dir):
            target_dir = os.path.join(output_dir, f"snr_{snr}dB")
            os.makedirs(target_dir, exist_ok=True)

            for fname in os.listdir(snr_dir):
                src = os.path.join(snr_dir, fname)
                dst = os.path.join(target_dir, fname)
                np.save(dst, np.load(src))
            saved[f'snr_{snr}dB'] = target_dir

    return saved


# 便捷函数
def quick_synthesize(clean_eeg: np.ndarray,
                    snr_db: float = -5,
                    seed: Optional[int] = None,
                    **kwargs) -> np.ndarray:
    """
    快速合成单个样本

    Args:
        clean_eeg: 清洁EEG (n_channels, n_samples)
        snr_db: 目标SNR (dB)
        seed: 随机种子
        **kwargs: 其他合成器配置

    Returns:
        含噪EEG
    """
    config = SynthesizerConfig(snr_values_db=[snr_db], seed=seed, **kwargs)
    synthesizer = EEGSynthesizer(config)
    return synthesizer.synthesize_single(clean_eeg, snr_db)


def batch_synthesize_dataset(input_npy: str,
                            output_dir: str,
                            snr_values: List[float] = [0, -5, -10],
                            seed: Optional[int] = None,
                            **kwargs) -> Dict[str, str]:
    """
    批量合成整个数据集

    Args:
        input_npy: 输入numpy文件路径（包含X, y）
        output_dir: 输出目录
        snr_values: SNR值列表
        seed: 随机种子
        **kwargs: 其他配置

    Returns:
        保存的文件路径
    """
    # 加载数据
    data = np.load(input_npy, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        # 可能是字典格式
        if len(data.shape) == 0:
            data = data.item()
            X = data['X']
            y = data['y']
            subjects = data.get('subjects')
        else:
            X, y = data[()]['X'], data[()]['y']
            subjects = data[()].get('subjects')
    else:
        X = data['X']
        y = data['y']
        subjects = data.get('subjects') if isinstance(data, dict) else None

    # 创建合成器
    config = SynthesizerConfig(snr_values_db=snr_values, seed=seed, **kwargs)
    synthesizer = EEGSynthesizer(config)

    # 合成
    results = synthesizer.synthesize_batch(X, y, snr_values, subjects)

    # 保存
    return synthesizer.save_dataset(results, output_dir, format='numpy')
