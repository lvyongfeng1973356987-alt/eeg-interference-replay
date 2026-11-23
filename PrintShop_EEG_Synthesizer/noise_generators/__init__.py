"""
PrintShop_EEG_Synthesizer: 物理噪声生成器模块
============================================

实现印刷车间环境中的三类物理噪声建模：
1. PLI (Power Line Interference) - 工频电磁干扰 50Hz
2. BW (Baseline Wander) - 基线漂移 0.1-0.5Hz
3. Mechanical Shock - 机械冲击突变伪影

理论基础：
- Zhang et al. (2021) - EEGdenoiseNet, J. Neural Engineering
- Al-Saegh et al. (2021) - Deep learning for EEG cognitive load recognition

Author: PrintShop_EEG_Synthesizer Team
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import warnings
from scipy.signal import butter, lfilter


class BaseNoiseGenerator(ABC):
    """物理噪声生成器抽象基类"""

    def __init__(self, sampling_rate: int = 128, seed: Optional[int] = None):
        """
        Args:
            sampling_rate: 采样率 (Hz)，默认128Hz（与SEED/DEAP预处理后一致）
            seed: 随机种子，用于结果可复现
        """
        self.sampling_rate = sampling_rate
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def generate(self, n_samples: int, duration: Optional[float] = None,
                 amplitude: float = 1.0, **kwargs) -> np.ndarray:
        """
        生成噪声信号

        Args:
            n_samples: 采样点数
            duration: 信号时长（秒），如果提供则忽略n_samples
            amplitude: 幅度缩放因子
            **kwargs: 子类特定参数

        Returns:
            噪声信号，形状 (n_samples,) 或 (n_channels, n_samples)
        """
        pass

    def _validate_duration(self, n_samples: int, duration: Optional[float]) -> int:
        """验证并返回有效的采样点数"""
        if duration is not None:
            return int(duration * self.sampling_rate)
        return n_samples


class PLINoiseGenerator(BaseNoiseGenerator):
    """
    工频电磁干扰 (Power Line Interference) 生成器

    模拟印刷车间50Hz电网辐射造成的电磁干扰。

    数学模型:
        n_PLI(t) = A_PLI * sin(2π * f * t + φ), φ ~ U(0, 2π)

    特性:
        - 主频: 50Hz (中国/亚洲标准) 或 60Hz (北美标准)
        - 可叠加2/3次谐波
        - 随机相位避免固定相位效应
    """

    def __init__(self, sampling_rate: int = 128, frequency: float = 50.0,
                 amplitude: float = 1.0, harmonics: bool = True,
                 seed: Optional[int] = None):
        """
        Args:
            sampling_rate: 采样率 (Hz)
            frequency: 工频频率，默认50Hz
            amplitude: 基线幅度 (μV)
            harmonics: 是否叠加谐波 (100Hz, 150Hz)
            seed: 随机种子
        """
        super().__init__(sampling_rate, seed)
        self.frequency = frequency
        self.base_amplitude = amplitude
        self.harmonics = harmonics

    def generate(self, n_samples: int, duration: Optional[float] = None,
                 amplitude: float = 1.0, **kwargs) -> np.ndarray:
        """
        生成PLI噪声

        Args:
            n_samples: 采样点数
            duration: 时长（秒）
            amplitude: 幅度缩放因子（相对于base_amplitude）

        Returns:
            噪声信号 (n_samples,)
        """
        n = self._validate_duration(n_samples, duration)
        t = np.arange(n) / self.sampling_rate

        # 主频成分
        phase = self.rng.uniform(0, 2 * np.pi)
        noise = self.base_amplitude * amplitude * np.sin(
            2 * np.pi * self.frequency * t + phase
        )

        # 叠加谐波 (2次、3次谐波，幅度递减)
        if self.harmonics:
            harmonic_2 = 0.3 * self.base_amplitude * amplitude * np.sin(
                2 * np.pi * 2 * self.frequency * t +
                self.rng.uniform(0, 2 * np.pi)
            )
            harmonic_3 = 0.15 * self.base_amplitude * amplitude * np.sin(
                2 * np.pi * 3 * self.frequency * t +
                self.rng.uniform(0, 2 * np.pi)
            )
            noise = noise + harmonic_2 + harmonic_3

        return noise

    def generate_multi_channel(self, n_samples: int, n_channels: int,
                               duration: Optional[float] = None,
                               amplitude: float = 1.0,
                               inter_channel_phase_var: float = 0.5) -> np.ndarray:
        """
        生成多通道PLI噪声（模拟不同电极接收到的不同相位干扰）

        Args:
            n_samples: 采样点数
            n_channels: 通道数
            duration: 时长（秒）
            amplitude: 幅度缩放因子
            inter_channel_phase_var: 通道间相位差异程度 (0-1)

        Returns:
            噪声信号 (n_channels, n_samples)
        """
        n = self._validate_duration(n_samples, duration)
        t = np.arange(n) / self.sampling_rate

        noise_matrix = np.zeros((n_channels, n))

        for ch in range(n_channels):
            # 每个通道有不同的相位偏移
            phase_base = self.rng.uniform(0, 2 * np.pi)
            phase_offset = self.rng.uniform(-np.pi, np.pi) * inter_channel_phase_var
            phase = phase_base + phase_offset

            amplitude_ch = self.base_amplitude * amplitude * self.rng.uniform(0.8, 1.2)
            noise = amplitude_ch * np.sin(2 * np.pi * self.frequency * t + phase)

            if self.harmonics:
                noise += 0.3 * amplitude_ch * np.sin(
                    2 * np.pi * 2 * self.frequency * t +
                    self.rng.uniform(0, 2 * np.pi)
                )
                noise += 0.15 * amplitude_ch * np.sin(
                    2 * np.pi * 3 * self.frequency * t +
                    self.rng.uniform(0, 2 * np.pi)
                )

            noise_matrix[ch] = noise

        return noise_matrix


class BaselineWanderGenerator(BaseNoiseGenerator):
    """
    基线漂移 (Baseline Wander) 生成器

    模拟电极接触阻抗变化、低频线缆摆动等造成的低频干扰。

    数学模型:
        n_BW(t) = Σ A_k * sin(2π * f_k * t + φ_k), f_k ~ U(0.1, 0.5)Hz

    特性:
        - 频率范围: 0.1-0.5Hz
        - 多正弦叠加模拟非平稳漂移
        - 可设置漂移速率
    """

    def __init__(self, sampling_rate: int = 128, n_components: int = 5,
                 freq_range: Tuple[float, float] = (0.1, 0.5),
                 amplitude: float = 1.0, seed: Optional[int] = None):
        """
        Args:
            sampling_rate: 采样率 (Hz)
            n_components: 正弦分量数量
            freq_range: 频率范围 (Hz)
            amplitude: 基线幅度 (μV)
            seed: 随机种子
        """
        super().__init__(sampling_rate, seed)
        self.n_components = n_components
        self.freq_range = freq_range
        self.base_amplitude = amplitude

    def generate(self, n_samples: int, duration: Optional[float] = None,
                 amplitude: float = 1.0, **kwargs) -> np.ndarray:
        """
        生成基线漂移噪声

        Args:
            n_samples: 采样点数
            duration: 时长（秒）
            amplitude: 幅度缩放因子

        Returns:
            噪声信号 (n_samples,)
        """
        n = self._validate_duration(n_samples, duration)
        t = np.arange(n) / self.sampling_rate

        noise = np.zeros(n)

        # 随机生成多个正弦分量
        frequencies = self.rng.uniform(
            self.freq_range[0], self.freq_range[1], self.n_components
        )
        amplitudes = self.rng.uniform(0.2, 1.0, self.n_components)
        amplitudes = amplitudes / amplitudes.sum() * self.n_components  # 归一化

        for i in range(self.n_components):
            phase = self.rng.uniform(0, 2 * np.pi)
            noise += (self.base_amplitude * amplitude * amplitudes[i] *
                      np.sin(2 * np.pi * frequencies[i] * t + phase))

        return noise

    def generate_walking_baseline(self, n_samples: int,
                                  duration: Optional[float] = None,
                                  amplitude: float = 1.0,
                                  walk_speed: float = 0.1) -> np.ndarray:
        """
        生成缓慢游走的基线漂移（更真实的模拟）

        Args:
            n_samples: 采样点数
            duration: 时长（秒）
            amplitude: 幅度缩放因子
            walk_speed: 基线游走速度 (0-1)

        Returns:
            噪声信号 (n_samples,)
        """
        n = self._validate_duration(n_samples, duration)

        # 使用低通滤波的高斯白噪声模拟缓慢漂移
        dt = 1.0 / self.sampling_rate
        noise = self.rng.standard_normal(n) * self.base_amplitude * amplitude * 0.1

        # 简单的一阶低通滤波模拟缓慢变化
        alpha = walk_speed * 0.01
        output = np.zeros(n)
        output[0] = noise[0]
        for i in range(1, n):
            output[i] = alpha * noise[i] + (1 - alpha) * output[i-1]

        return output


class MechanicalShockGenerator(BaseNoiseGenerator):
    """
    机械冲击突变伪影 (Mechanical Shock) 生成器

    模拟印刷机冲压、瞬间撞击等机械冲击造成的宽带噪声。

    数学模型:
        n_Shock(t) = A_impulse * exp(-(t-t0)²/(2σ²)) * ξ(t), ξ(t) ~ N(0,1)

    特性:
        - 高斯衰减包络
        - 宽带噪声叠加
        - 可设置冲击时刻和强度
    """

    def __init__(self, sampling_rate: int = 128, amplitude: float = 1.0,
                 pulse_duration: float = 0.1, seed: Optional[int] = None):
        """
        Args:
            sampling_rate: 采样率 (Hz)
            amplitude: 冲击基线幅度 (μV)
            pulse_duration: 冲击脉冲持续时间 (秒)
            seed: 随机种子
        """
        super().__init__(sampling_rate, seed)
        self.base_amplitude = amplitude
        self.pulse_duration = pulse_duration

    def generate(self, n_samples: int, duration: Optional[float] = None,
                 amplitude: float = 1.0, **kwargs) -> np.ndarray:
        """
        生成单个机械冲击噪声

        Args:
            n_samples: 采样点数
            duration: 时长（秒）
            amplitude: 幅度缩放因子

        Returns:
            噪声信号 (n_samples,)
        """
        n = self._validate_duration(n_samples, duration)
        t = np.arange(n) / self.sampling_rate

        # 冲击时刻（随机位置）
        shock_time = self.rng.uniform(0.2, 0.8) * n / self.sampling_rate
        sigma = self.pulse_duration / 3  # 高斯包络宽度

        # 高斯包络
        envelope = np.exp(-((t - shock_time) ** 2) / (2 * sigma ** 2))

        # 宽带噪声叠加
        white_noise = self.rng.standard_normal(n)

        # 低通滤波使噪声更真实
        cutoff = 0.3 / sigma  # 与脉冲宽度相关
        b, a = self._butter_lowpass(cutoff, self.sampling_rate)
        filtered_noise = self._apply_filter(white_noise, b, a)

        noise = self.base_amplitude * amplitude * envelope * filtered_noise

        return noise

    def generate_multiple_shocks(self, n_samples: int,
                                  duration: Optional[float] = None,
                                  amplitude: float = 1.0,
                                  n_shocks: int = 3,
                                  shock_interval_min: float = 0.5) -> np.ndarray:
        """
        生成多个机械冲击（模拟连续冲压）

        Args:
            n_samples: 采样点数
            duration: 时长（秒）
            amplitude: 幅度缩放因子
            n_shocks: 冲击次数
            shock_interval_min: 最小冲击间隔 (秒)

        Returns:
            噪声信号 (n_samples,)
        """
        n = self._validate_duration(n_samples, duration)
        t = np.arange(n) / self.sampling_rate
        noise = np.zeros(n)

        total_duration = n / self.sampling_rate
        max_interval = (total_duration - n_shocks * self.pulse_duration) / n_shocks

        current_time = self.rng.uniform(0.1, 0.3)

        for _ in range(n_shocks):
            if current_time + self.pulse_duration > total_duration * 0.9:
                break

            sigma = self.pulse_duration / 3 / self.rng.uniform(0.8, 1.2)
            envelope = np.exp(-((t - current_time) ** 2) / (2 * sigma ** 2))

            # 随机幅度变化
            amp = self.base_amplitude * amplitude * self.rng.uniform(0.5, 1.5)
            white_noise = self.rng.standard_normal(n)
            b, a = self._butter_lowpass(0.3 / sigma, self.sampling_rate)
            filtered_noise = self._apply_filter(white_noise, b, a)

            noise += amp * envelope * filtered_noise

            # 下一个冲击时刻
            interval = self.rng.uniform(shock_interval_min, max_interval)
            current_time += self.pulse_duration + interval

        return noise

    def generate_multi_channel(self, n_samples: int, n_channels: int,
                                duration: Optional[float] = None,
                                amplitude: float = 1.0,
                                spatial_decay: float = 0.3) -> np.ndarray:
        """
        生成多通道机械冲击（冲击在空间上有传播衰减）

        Args:
            n_samples: 采样点数
            n_channels: 通道数
            duration: 时长（秒）
            amplitude: 幅度缩放因子
            spatial_decay: 空间衰减系数

        Returns:
            噪声信号 (n_channels, n_samples)
        """
        n = self._validate_duration(n_samples, duration)
        noise_matrix = np.zeros((n_channels, n))

        # 主冲击信号
        main_shock = self.generate(n, duration=duration, amplitude=amplitude)

        for ch in range(n_channels):
            # 不同通道有不同衰减
            decay = np.exp(-spatial_decay * abs(ch - n_channels // 2))
            channel_amp = amplitude * decay * self.rng.uniform(0.7, 1.0)
            noise_matrix[ch] = self.generate(n, duration=duration,
                                            amplitude=channel_amp)

        return noise_matrix

    @staticmethod
    def _butter_lowpass(cutoff: float, fs: float, order: int = 4) -> Tuple:
        """设计低通滤波器"""
        nyq = 0.5 * fs
        normal_cutoff = min(cutoff / nyq, 0.99)
        b, a = butter(order, normal_cutoff, btype='low')
        return b, a

    @staticmethod
    def _apply_filter(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
        """应用滤波器"""
        return lfilter(b, a, x)


class CompositeNoiseGenerator(BaseNoiseGenerator):
    """
    复合噪声生成器

    将多个物理噪声源组合生成复合干扰信号。

    数学模型:
        n_composite(t) = w1*n_EMG + w2*n_EOG + w3*n_PLI + w4*n_BW + w5*n_Shock

    其中生理伪影(EMG/EOG)由artifact_loader提供，物理噪声由本模块生成。
    """

    def __init__(self, sampling_rate: int = 128,
                 pli_generator: Optional[PLINoiseGenerator] = None,
                 bw_generator: Optional[BaselineWanderGenerator] = None,
                 shock_generator: Optional[MechanicalShockGenerator] = None,
                 seed: Optional[int] = None):
        """
        Args:
            sampling_rate: 采样率 (Hz)
            pli_generator: PLI噪声生成器
            bw_generator: 基线漂移生成器
            shock_generator: 机械冲击生成器
            seed: 随机种子
        """
        super().__init__(sampling_rate, seed)

        self.pli = pli_generator or PLINoiseGenerator(sampling_rate, seed=seed)
        self.bw = bw_generator or BaselineWanderGenerator(sampling_rate, seed=seed)
        self.shock = shock_generator or MechanicalShockGenerator(sampling_rate, seed=seed)

        # 默认权重
        self.weights = {
            'pli': 1.0,
            'bw': 1.0,
            'shock': 1.0
        }

    def set_weights(self, pli: float = 1.0, bw: float = 1.0, shock: float = 1.0):
        """设置各噪声源的权重"""
        self.weights = {'pli': pli, 'bw': bw, 'shock': shock}

    def generate(self, n_samples: int, duration: Optional[float] = None,
                 amplitude: float = 1.0, **kwargs) -> np.ndarray:
        """
        生成复合噪声

        Args:
            n_samples: 采样点数
            duration: 时长（秒）
            amplitude: 总幅度缩放因子

        Returns:
            复合噪声信号 (n_samples,)
        """
        n = self._validate_duration(n_samples, duration)

        noise = np.zeros(n)

        noise += self.weights['pli'] * self.pli.generate(n, duration=duration)
        noise += self.weights['bw'] * self.bw.generate(n, duration=duration)
        noise += self.weights['shock'] * self.shock.generate(n, duration=duration)

        return noise * amplitude

    def generate_from_config(self, n_samples: int,
                            config: Dict[str, Any],
                            duration: Optional[float] = None) -> np.ndarray:
        """
        根据配置字典生成复合噪声

        Args:
            n_samples: 采样点数
            config: 噪声配置，如 {'pli_amp': 1.0, 'bw_amp': 0.5, 'shock_amp': 2.0}
            duration: 时长（秒）

        Returns:
            复合噪声信号
        """
        n = self._validate_duration(n_samples, duration)
        noise = np.zeros(n)

        if config.get('pli_enabled', True):
            pli_amp = config.get('pli_amp', 1.0)
            noise += self.weights['pli'] * pli_amp * self.pli.generate(n, duration=duration)

        if config.get('bw_enabled', True):
            bw_amp = config.get('bw_amp', 1.0)
            noise += self.weights['bw'] * bw_amp * self.bw.generate(n, duration=duration)

        if config.get('shock_enabled', True):
            shock_amp = config.get('shock_amp', 1.0)
            noise += self.weights['shock'] * shock_amp * self.shock.generate(n, duration=duration)

        return noise

    def generate_multi_channel(self, n_samples: int, n_channels: int,
                                duration: Optional[float] = None,
                                amplitude: float = 1.0,
                                emg_noise: Optional[np.ndarray] = None,
                                eog_noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        生成多通道复合噪声（可叠加EMG/EOG生理伪影）

        Args:
            n_samples: 采样点数
            n_channels: 通道数
            duration: 时长（秒）
            amplitude: 幅度缩放因子
            emg_noise: EMG噪声 (n_channels, n_samples) 或 (n_samples,)
            eog_noise: EOG噪声 (n_channels, n_samples) 或 (n_samples,)

        Returns:
            复合噪声信号 (n_channels, n_samples)
        """
        n = self._validate_duration(n_samples, duration)

        # PLI噪声（多通道）
        pli = self.pli.generate_multi_channel(n, n_channels, duration=duration)

        # 基线漂移（多通道）
        bw = np.zeros((n_channels, n))
        for ch in range(n_channels):
            bw[ch] = self.bw.generate(n, duration=duration) * self.rng.uniform(0.8, 1.2)

        # 机械冲击（多通道）
        shock = self.shock.generate_multi_channel(n, n_channels, duration=duration)

        # 叠加各噪声源
        noise = (self.weights['pli'] * pli +
                self.weights['bw'] * bw +
                self.weights['shock'] * shock)

        # 叠加生理伪影（如果提供）
        if emg_noise is not None:
            emg = self._broadcast_to_2d(emg_noise, n_channels, n)
            noise += emg * config.get('emg_amp', 1.0) if isinstance(config.get('emg_amp'), (int, float)) else emg

        if eog_noise is not None:
            eog = self._broadcast_to_2d(eog_noise, n_channels, n)
            noise += eog * config.get('eog_amp', 1.0) if isinstance(config.get('eog_amp'), (int, float)) else eog

        return noise * amplitude

    @staticmethod
    def _broadcast_to_2d(arr: np.ndarray, n_channels: int, n_samples: int) -> np.ndarray:
        """将1D或3D噪声广播到2D (n_channels, n_samples)"""
        if arr.ndim == 1:
            return np.tile(arr, (n_channels, 1))
        elif arr.ndim == 3:
            return arr.reshape(-1, n_samples)[:n_channels]
        return arr


def create_noise_generator(noise_type: str, **kwargs) -> BaseNoiseGenerator:
    """
    工厂函数：创建噪声生成器

    Args:
        noise_type: 噪声类型 ('pli', 'bw', 'shock', 'composite')
        **kwargs: 传递给生成器的参数

    Returns:
        对应的噪声生成器实例
    """
    generators = {
        'pli': PLINoiseGenerator,
        'baseline_wander': BaselineWanderGenerator,
        'bw': BaselineWanderGenerator,
        'shock': MechanicalShockGenerator,
        'mechanical_shock': MechanicalShockGenerator,
        'composite': CompositeNoiseGenerator,
    }

    if noise_type.lower() not in generators:
        raise ValueError(f"未知的噪声类型: {noise_type}。"
                        f"可选: {list(generators.keys())}")

    return generators[noise_type.lower()](**kwargs)
