"""
阶段二：引擎耦合与真实节律注入
=============================

核心功能：
1. 随机时序切片采样 - 从2小时真实节律中随机截取片段
2. 动态调制机械冲击 - 基于包络权重调制冲击强度
3. 动态调制基线漂移 - 声音强度与BW幅度挂钩

数学模型：
- 机械冲击: A_shock(t) = A_base * w_real(t) * exp decay
- 基线漂移: A_BW(t) = A_base * (1 + k * w_real(t))

Author: EEG_NOISE_MODEL_LYF Team
"""

import numpy as np
from typing import Optional, Dict, Tuple, Union, Callable
from scipy.signal import butter, lfilter
import warnings

from .noise_generators import (
    BaseNoiseGenerator,
    PLINoiseGenerator,
    BaselineWanderGenerator,
    MechanicalShockGenerator,
    CompositeNoiseGenerator
)
from .acoustic_processor import RealRhythmExtractor, audio_to_eeg_control_signal


class DynamicShockGenerator(MechanicalShockGenerator):
    """
    动态调制机械冲击生成器

    根据真实节律权重 w_real(t) 动态调制冲击强度和时刻
    """

    def __init__(
        self,
        sampling_rate: int = 128,
        amplitude: float = 1.0,
        pulse_duration: float = 0.1,
        shock_threshold: float = 0.6,  # 触发冲击的最低权重
        seed: Optional[int] = None
    ):
        """
        Args:
            sampling_rate: 采样率 (Hz)
            amplitude: 基础冲击幅度
            pulse_duration: 冲击脉冲持续时间 (秒)
            shock_threshold: 触发冲击的权重阈值 [0,1]
            seed: 随机种子
        """
        super().__init__(sampling_rate, amplitude, pulse_duration, seed)
        self.shock_threshold = shock_threshold

    def generate_from_rhythm(
        self,
        n_samples: int,
        w_real: np.ndarray,
        duration: Optional[float] = None,
        intensity_modulation: float = 1.0
    ) -> np.ndarray:
        """
        基于真实节律生成动态调制冲击

        Args:
            n_samples: 输出采样点数
            w_real: 真实节律权重数组 (n_samples,)
            duration: 信号时长（秒）
            intensity_modulation: 强度调制系数

        Returns:
            动态冲击信号 (n_samples,)
        """
        n = self._validate_duration(n_samples, duration)

        # 确保w_real长度匹配
        if len(w_real) < n:
            w_real = np.pad(w_real, (0, n - len(w_real)), mode='edge')
        elif len(w_real) > n:
            w_real = w_real[:n]

        t = np.arange(n) / self.sampling_rate
        noise = np.zeros(n)

        # 找到所有超过阈值的时刻
        high_intensity_indices = np.where(w_real > self.shock_threshold)[0]

        if len(high_intensity_indices) == 0:
            return noise

        # 将连续高强度区域分组为独立的冲击事件
        gaps = np.diff(high_intensity_indices)
        group_starts = [0]
        group_starts.extend(np.where(gaps > self.sampling_rate * 0.1)[0] + 1)

        for i in range(len(group_starts)):
            if i < len(group_starts) - 1:
                group_end = group_starts[i + 1]
            else:
                group_end = len(high_intensity_indices)

            group_indices = high_intensity_indices[group_starts[i]:group_end]
            if len(group_indices) == 0:
                continue

            # 找峰值位置
            peak_idx = group_indices[np.argmax(w_real[group_indices])]
            peak_weight = w_real[peak_idx]

            # 生成单个冲击（幅度由峰值权重调制）
            shock_amp = self.base_amplitude * peak_weight * intensity_modulation
            sigma = self.pulse_duration / 3 / (1 + peak_weight)  # 高权重时冲击更尖锐

            envelope = np.exp(-((t - t[peak_idx]) ** 2) / (2 * sigma ** 2))
            envelope = envelope * peak_weight  # 能量与权重成正比

            # 宽带噪声
            white_noise = self.rng.standard_normal(n)
            cutoff = 0.3 / sigma
            b, a = self._butter_lowpass(cutoff, self.sampling_rate)
            filtered_noise = self._apply_filter(white_noise, b, a)

            noise += shock_amp * envelope * filtered_noise

        return noise


class DynamicBaselineWanderGenerator(BaselineWanderGenerator):
    """
    动态调制基线漂移生成器

    基线漂移幅度随声音能量包络动态变化
    """

    def __init__(
        self,
        sampling_rate: int = 128,
        n_components: int = 5,
        freq_range: Tuple[float, float] = (0.1, 0.5),
        amplitude: float = 1.0,
        base_bw_amplitude: float = 0.5,  # 安静时的BW幅度
        max_bw_amplitude: float = 2.0,   # 嘈杂时的最大BW幅度
        seed: Optional[int] = None
    ):
        """
        Args:
            sampling_rate: 采样率 (Hz)
            n_components: 正弦分量数量
            freq_range: 频率范围 (Hz)
            amplitude: 最大幅度缩放
            base_bw_amplitude: 安静时的基础BW幅度
            max_bw_amplitude: 嘈杂时的最大BW幅度
            seed: 随机种子
        """
        super().__init__(sampling_rate, n_components, freq_range, amplitude, seed)
        self.base_bw_amplitude = base_bw_amplitude
        self.max_bw_amplitude = max_bw_amplitude

    def generate_from_rhythm(
        self,
        n_samples: int,
        w_real: np.ndarray,
        duration: Optional[float] = None
    ) -> np.ndarray:
        """
        基于真实节律生成动态基线漂移

        Args:
            n_samples: 采样点数
            w_real: 真实节律权重 (n_samples,)
            duration: 时长（秒）

        Returns:
            动态基线漂移信号
        """
        n = self._validate_duration(n_samples, duration)

        # 确保w_real长度匹配
        if len(w_real) < n:
            w_real = np.pad(w_real, (0, n - len(w_real)), mode='edge')
        elif len(w_real) > n:
            w_real = w_real[:n]

        t = np.arange(n) / self.sampling_rate

        # 动态幅度：安静时小，嘈杂时大
        # 使用缓动函数避免突变
        amplitude_map = self.base_bw_amplitude + (self.max_bw_amplitude - self.base_bw_amplitude) * w_real

        # 生成分段正弦基线漂移
        noise = np.zeros(n)
        n_segments = max(1, n // (self.sampling_rate * 10))  # 10秒一段
        segment_length = n // n_segments

        for seg in range(n_segments):
            start = seg * segment_length
            end = min((seg + 1) * segment_length, n)
            seg_w_real = np.mean(w_real[start:min(start + segment_length, n)])

            # 该段的幅度
            seg_amp = self.base_bw_amplitude + (self.max_bw_amplitude - self.base_bw_amplitude) * seg_w_real

            # 生成该段的正弦噪声
            seg_t = t[start:end]
            seg_noise = np.zeros(end - start)

            frequencies = self.rng.uniform(
                self.freq_range[0], self.freq_range[1], self.n_components
            )
            amplitudes = self.rng.uniform(0.2, 1.0, self.n_components)
            amplitudes = amplitudes / amplitudes.sum() * self.n_components

            for i in range(self.n_components):
                phase = self.rng.uniform(0, 2 * np.pi)
                seg_noise += (self.base_amplitude * seg_amp * amplitudes[i] *
                            np.sin(2 * np.pi * frequencies[i] * seg_t + phase))

            noise[start:end] = seg_noise

        return noise


class RealRhythmNoiseGenerator(BaseNoiseGenerator):
    """
    真实节律驱动噪声生成器

    将声学提取的真实物理节律深度融合到噪声合成中
    """

    def __init__(
        self,
        sampling_rate: int = 128,
        w_real: Optional[np.ndarray] = None,
        shock_threshold: float = 0.6,
        bw_base: float = 0.5,
        bw_max: float = 2.0,
        seed: Optional[int] = None
    ):
        """
        Args:
            sampling_rate: 采样率 (Hz)
            w_real: 预计算的真实节律权重数组
            shock_threshold: 冲击阈值
            bw_base: 基线漂移基础幅度
            bw_max: 基线漂移最大幅度
            seed: 随机种子
        """
        super().__init__(sampling_rate, seed)

        self.w_real = w_real
        self.total_length = len(w_real) if w_real is not None else 0

        # 初始化子生成器
        self.shock_gen = DynamicShockGenerator(
            sampling_rate=sampling_rate,
            shock_threshold=shock_threshold,
            seed=seed
        )

        self.bw_gen = DynamicBaselineWanderGenerator(
            sampling_rate=sampling_rate,
            base_bw_amplitude=bw_base,
            max_bw_amplitude=bw_max,
            seed=seed
        )

        self.pli_gen = PLINoiseGenerator(sampling_rate=sampling_rate, seed=seed)

        # 权重配置
        self.weights = {'pli': 1.0, 'bw': 1.0, 'shock': 1.0}

    def set_w_real(self, w_real: np.ndarray):
        """设置真实节律权重数组"""
        self.w_real = w_real
        self.total_length = len(w_real)

    def random_slice(self, n_samples: int) -> np.ndarray:
        """
        随机时序切片采样

        从2小时的w_real数组中随机选取一个起点，截取n_samples长度

        Args:
            n_samples: 需要的采样点数

        Returns:
            截取的权重片段
        """
        if self.w_real is None:
            raise ValueError("必须先设置w_real数组")

        if self.total_length <= n_samples:
            # 如果数组太短，返回全部
            return self.w_real.copy()

        # 随机起点
        max_start = self.total_length - n_samples
        start_idx = self.rng.integers(0, max_start + 1)

        # 截取
        return self.w_real[start_idx:start_idx + n_samples].copy()

    def generate(
        self,
        n_samples: int,
        duration: Optional[float] = None,
        amplitude: float = 1.0,
        use_real_rhythm: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        生成含真实节律的复合噪声

        Args:
            n_samples: 采样点数
            duration: 时长（秒）
            amplitude: 幅度缩放
            use_real_rhythm: 是否使用真实节律（False则使用随机生成）

        Returns:
            噪声信号 (n_samples,)
        """
        n = self._validate_duration(n_samples, duration)

        # 获取权重切片
        if use_real_rhythm and self.w_real is not None:
            w_slice = self.random_slice(n)
        else:
            # 随机权重（用于对比）
            w_slice = self.rng.uniform(0.1, 0.9, n)

        # 生成各分量
        noise = np.zeros(n)

        # PLI（工频干扰）- 强度与权重相关
        pli = self.pli_gen.generate(n, duration=duration)
        pli = pli * (0.8 + 0.4 * w_slice)  # 权重越高，PLI越强
        noise += self.weights['pli'] * pli

        # 基线漂移 - 动态调制
        bw = self.bw_gen.generate_from_rhythm(n, w_slice)
        noise += self.weights['bw'] * bw

        # 机械冲击 - 由权重峰值触发
        shock = self.shock_gen.generate_from_rhythm(n, w_slice)
        noise += self.weights['shock'] * shock

        return noise * amplitude

    def generate_multi_channel(
        self,
        n_samples: int,
        n_channels: int,
        duration: Optional[float] = None,
        amplitude: float = 1.0,
        spatial_decay: float = 0.2,
        rng: Optional[np.random.Generator] = None,
        **kwargs
    ) -> np.ndarray:
        """
        生成多通道含真实节律的噪声（优化版：批量生成）

        Args:
            n_samples: 采样点数
            n_channels: 通道数
            duration: 时长（秒）
            amplitude: 幅度缩放
            spatial_decay: 通道间空间衰减系数
            rng: 可选的外部随机数发生器。传入时用临时替换 self.rng 的方式
                 使各子生成器 (pli/bw/shock) 也用该 RNG, 从而保证
                 "相同 rng seed -> 相同噪声波形" (便于并行 + 可复现)。

        Returns:
            噪声信号 (n_channels, n_samples)
        """
        n = self._validate_duration(n_samples, duration)

        # 可选: 临时替换 RNG, 使本次调用完全确定
        _restore = None
        if rng is not None:
            _restore = []
            for obj in [self, self.pli_gen, self.bw_gen, self.shock_gen]:
                if hasattr(obj, "rng"):
                    _restore.append((obj, obj.rng))
                    obj.rng = rng

        try:
            # 获取公共权重切片（所有通道共享）
            if self.w_real is not None:
                w_slice = self.random_slice(n)
            else:
                w_slice = self.rng.uniform(0.1, 0.9, n)

            # 批量生成基础噪声分量（所有通道共享）
            # PLI
            pli_base = self.pli_gen.generate(n, duration=duration)
            pli_base = pli_base * (0.8 + 0.4 * w_slice)

            # 基线漂移
            bw_base = self.bw_gen.generate_from_rhythm(n, w_slice)

            # 机械冲击
            shock_base = self.shock_gen.generate_from_rhythm(n, w_slice)

            # 组合基础噪声
            noise_base = (self.weights['pli'] * pli_base +
                         self.weights['bw'] * bw_base +
                         self.weights['shock'] * shock_base)

            # 为每个通道添加空间变化
            noise_matrix = np.zeros((n_channels, n))

            for ch in range(n_channels):
                # 空间衰减
                ch_decay = np.exp(-spatial_decay * abs(ch - n_channels // 2))
                ch_amp = amplitude * ch_decay * self.rng.uniform(0.7, 1.0)

                # 基础噪声（共享）+ 通道特定相位噪声
                ch_noise = ch_amp * noise_base
                phase_noise = 0.1 * ch_amp * self.rng.standard_normal(n)

                noise_matrix[ch] = ch_noise + phase_noise

            return noise_matrix
        finally:
            if _restore is not None:
                for obj, old_rng in _restore:
                    obj.rng = old_rng

    def set_weights(self, pli: float = 1.0, bw: float = 1.0, shock: float = 1.0):
        """设置各分量权重"""
        self.weights = {'pli': pli, 'bw': bw, 'shock': shock}


class RealRhythmSynthesizer:
    """
    真实节律驱动的EEG噪声合成器

    将声学处理得到的真实节律注入到EEG噪声合成全流程
    """

    def __init__(
        self,
        sampling_rate: int = 128,
        w_real: Optional[np.ndarray] = None,
        audio_path: Optional[str] = None,
        audio_signal: Optional[np.ndarray] = None,
        audio_sampling_rate: float = 44100.0,
        enable_pli: bool = True,
        enable_bw: bool = True,
        enable_shock: bool = True,
        seed: Optional[int] = None
    ):
        """
        Args:
            sampling_rate: EEG采样率
            w_real: 预计算的真实节律权重
            audio_path: 音频文件路径（二选一）
            audio_signal: 直接传入的音频信号
            audio_sampling_rate: 音频采样率
            enable_pli: 是否启用PLI
            enable_bw: 是否启用基线漂移
            enable_shock: 是否启用机械冲击
            seed: 随机种子
        """
        self.sampling_rate = sampling_rate
        self.rng = np.random.default_rng(seed)

        # 处理音频获取节律
        if w_real is not None:
            self.w_real = w_real
        elif audio_path is not None or audio_signal is not None:
            self.w_real = audio_to_eeg_control_signal(
                audio_path=audio_path,
                audio_signal=audio_signal,
                audio_sampling_rate=audio_sampling_rate,
                eeg_sampling_rate=sampling_rate
            )
        else:
            self.w_real = None

        # 初始化噪声生成器
        self.noise_gen = RealRhythmNoiseGenerator(
            sampling_rate=sampling_rate,
            w_real=self.w_real,
            seed=seed
        )

        self.enable_pli = enable_pli
        self.enable_bw = enable_bw
        self.enable_shock = enable_shock

        # 配置权重
        if not enable_pli:
            self.noise_gen.weights['pli'] = 0.0
        if not enable_bw:
            self.noise_gen.weights['bw'] = 0.0
        if not enable_shock:
            self.noise_gen.weights['shock'] = 0.0

    def synthesize_single(
        self,
        clean_eeg: np.ndarray,
        target_snr_db: float,
        return_components: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        合成单个含噪EEG样本

        Args:
            clean_eeg: 清洁EEG (n_channels, n_samples)
            target_snr_db: 目标SNR (dB)
            return_components: 是否返回噪声分量

        Returns:
            含噪EEG 或 (含噪EEG, 噪声分量)
        """
        if clean_eeg.ndim != 2:
            raise ValueError("clean_eeg必须是2D数组 (n_channels, n_samples)")

        n_channels, n_samples = clean_eeg.shape

        # 计算信号功率
        signal_power = np.mean(clean_eeg ** 2)

        # 生成噪声（使用真实节律）
        noise = self.noise_gen.generate_multi_channel(n_samples, n_channels)

        # 计算当前噪声功率
        noise_power = np.mean(noise ** 2)

        if noise_power < 1e-10:
            warnings.warn("噪声功率过小")
            noise_power = 1e-10

        # 缩放到目标SNR
        # SNR_db = 10 * log10(P_signal / P_noise)
        # => P_noise = P_signal / 10^(SNR_db/10)
        target_noise_power = signal_power / (10 ** (target_snr_db / 10))
        noise = noise * np.sqrt(target_noise_power / noise_power)

        # 叠加
        noisy_eeg = clean_eeg + noise

        if return_components:
            components = {
                'noise': noise,
                'snr_db': target_snr_db,
                'actual_snr_db': self._calculate_snr(clean_eeg, noisy_eeg)
            }
            return noisy_eeg, components

        return noisy_eeg

    def _calculate_snr(self, clean: np.ndarray, noisy: np.ndarray) -> float:
        """计算实际SNR"""
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean((noisy - clean) ** 2)

        if noise_power < 1e-10:
            return float('inf')

        return 10 * np.log10(signal_power / noise_power)

    def synthesize_batch(
        self,
        clean_eegs: np.ndarray,
        labels: np.ndarray,
        snr_values: list = [0, -5, -10],
        subject_ids: Optional[np.ndarray] = None,
        show_progress: bool = True
    ) -> Dict:
        """
        批量合成

        Args:
            clean_eegs: 清洁EEG (N, n_channels, n_samples)
            labels: 标签 (N,)
            snr_values: SNR值列表
            subject_ids: 被试ID (N,)
            show_progress: 是否显示进度

        Returns:
            结果字典
        """
        from tqdm import tqdm

        results = {}
        n_samples = len(clean_eegs)

        for snr_db in snr_values:
            noisy_data = []
            noisy_labels = []
            noisy_subjects = []

            iterator = range(n_samples)
            if show_progress:
                iterator = tqdm(iterator, desc=f"合成 SNR={snr_db}dB")

            for i in iterator:
                clean = clean_eegs[i]
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


def create_real_rhythm_synthesizer(
    audio_path: Optional[str] = None,
    audio_signal: Optional[np.ndarray] = None,
    w_real: Optional[np.ndarray] = None,
    sampling_rate: int = 128,
    audio_sampling_rate: float = 44100.0,
    seed: Optional[int] = None,
    **kwargs
) -> RealRhythmSynthesizer:
    """
    工厂函数：创建真实节律合成器

    Args:
        audio_path: 音频文件路径
        audio_signal: 音频信号数组
        w_real: 预计算的权重数组
        sampling_rate: EEG采样率
        audio_sampling_rate: 音频采样率
        seed: 随机种子
        **kwargs: 其他参数

    Returns:
        RealRhythmSynthesizer实例
    """
    return RealRhythmSynthesizer(
        sampling_rate=sampling_rate,
        w_real=w_real,
        audio_path=audio_path,
        audio_signal=audio_signal,
        audio_sampling_rate=audio_sampling_rate,
        seed=seed,
        **kwargs
    )


def quick_synthesize_with_rhythm(
    clean_eeg: np.ndarray,
    w_real: np.ndarray,
    target_snr_db: float = -5,
    sampling_rate: int = 128,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    快速合成单个样本（便捷函数）

    Args:
        clean_eeg: 清洁EEG (n_channels, n_samples)
        w_real: 真实节律权重
        target_snr_db: 目标SNR
        sampling_rate: 采样率
        seed: 随机种子

    Returns:
        含噪EEG
    """
    synthesizer = RealRhythmSynthesizer(
        sampling_rate=sampling_rate,
        w_real=w_real,
        seed=seed
    )
    return synthesizer.synthesize_single(clean_eeg, target_snr_db)
