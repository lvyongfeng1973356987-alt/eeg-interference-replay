"""
阶段一：声学信号向脑电干扰特征的跨模态映射
============================================

核心功能：
1. 时间尺度对齐 - 将高采样率音频降采样到EEG采样率
2. 能量包络提取 - 通过低通滤波提取缓慢起伏的能量轮廓
3. 归一化控制信号构建 - 缩放到[0,1]区间

理论依据：
- 工厂机器运转产生的声音具有明显的物理节律
- 声音能量包络与物理冲击时刻高度相关
- 通过音频包络可精准定位高噪声干扰时刻

Author: EEG_NOISE_MODEL_LYF Team
"""

import numpy as np
from scipy.signal import butter, lfilter, resample_poly, firwin
from scipy.ndimage import uniform_filter1d
from typing import Tuple, Optional, Union
import warnings


class AcousticSignalProcessor:
    """
    声学信号处理器

    将高采样率音频信号转换为脑电尺度的时变控制信号
    """

    def __init__(
        self,
        audio_sampling_rate: float = 44100.0,
        eeg_sampling_rate: float = 128.0,
        lowpass_cutoff: float = 1.5,  # Hz - 超低频截止
        envelope_smooth_window: float = 0.5,  # 秒 - 包络平滑窗口
        normalize_range: Tuple[float, float] = (0.0, 1.0),
        seed: Optional[int] = None
    ):
        """
        Args:
            audio_sampling_rate: 音频采样率 (Hz)
            eeg_sampling_rate: EEG采样率 (Hz)
            lowpass_cutoff: 低通滤波器截止频率 (Hz)
            envelope_smooth_window: 包络平滑窗口大小 (秒)
            normalize_range: 归一化范围
            seed: 随机种子
        """
        self.audio_sr = audio_sampling_rate
        self.eeg_sr = eeg_sampling_rate
        self.lowpass_cutoff = lowpass_cutoff
        self.envelope_smooth_window = envelope_smooth_window
        self.normalize_range = normalize_range
        self.rng = np.random.default_rng(seed)

        # 预计算降采样比例
        self.gcd = np.gcd(int(audio_sampling_rate), int(eeg_sampling_rate))
        self.downsample_factor = int(audio_sampling_rate) // self.gcd
        self.upsample_factor = int(eeg_sampling_rate) // self.gcd

        # 预设计低通滤波器
        self._design_lowpass_filter()

    def _design_lowpass_filter(self):
        """设计低通滤波器用于包络提取"""
        nyquist = 0.5 * self.eeg_sr
        normalized_cutoff = min(self.lowpass_cutoff / nyquist, 0.99)

        # 使用Butterworth滤波器（阶数4，相位延迟小）
        self.filter_order = 4
        self.filter_b, self.filter_a = butter(
            self.filter_order, normalized_cutoff, btype='low'
        )

        # 零相位滤波（避免相位偏移）
        self._zero_phase = True

    def time_align(self, audio_signal: np.ndarray) -> np.ndarray:
        """
        时间尺度对齐：将音频降采样到EEG采样率

        使用重采样技术保证时域精度，避免欠采样失真

        Args:
            audio_signal: 原始音频信号 (n_samples,)

        Returns:
            对齐后的信号 (n_aligned_samples,)
        """
        if audio_signal.ndim != 1:
            raise ValueError("audio_signal must be 1D array")

        n_original = len(audio_signal)

        # 使用scipy.resample_poly进行高效重采样
        # 目标采样点数 = 原采样点数 * (目标采样率 / 原始采样率)
        target_n_samples = int(n_original * self.eeg_sr / self.audio_sr)

        # 确保为目标采样率
        aligned = resample_poly(
            audio_signal,
            self.upsample_factor,
            self.downsample_factor
        )

        # 精确截取到目标长度
        if len(aligned) > target_n_samples:
            aligned = aligned[:target_n_samples]
        elif len(aligned) < target_n_samples:
            # 补零
            padded = np.zeros(target_n_samples)
            padded[:len(aligned)] = aligned
            aligned = padded

        return aligned

    def extract_energy_envelope(
        self,
        audio_signal: np.ndarray,
        method: str = 'hilbert'
    ) -> np.ndarray:
        """
        能量包络提取：提取声音的缓慢起伏轮廓

        两种方法：
        1. 'hilbert': Hilbert变换（解析信号模值），精度高但计算量大
        2. 'rectify_filter': 整流+低通滤波，快速但精度稍低

        Args:
            audio_signal: 对齐后的音频信号
            method: 包络提取方法

        Returns:
            能量包络信号
        """
        if method == 'hilbert':
            return self._hilbert_envelope(audio_signal)
        elif method == 'rectify_filter':
            return self._rectify_filter_envelope(audio_signal)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _hilbert_envelope(self, signal: np.ndarray) -> np.ndarray:
        """
        Hilbert变换提取包络

        优点：相位保持，无额外延迟
        缺点：计算量大，可能出现端点效应
        """
        from scipy.signal import hilbert

        # 计算解析信号
        analytic_signal = hilbert(signal)

        # 包络 = 解析信号的模值
        envelope = np.abs(analytic_signal)

        # 应用低通滤波平滑
        envelope = self._smooth_envelope(envelope)

        return envelope

    def _rectify_filter_envelope(self, signal: np.ndarray) -> np.ndarray:
        """
        整流+低通滤波提取包络

        步骤：
        1. 取绝对值（半波整流）
        2. 超低频低通滤波
        3. 平滑处理
        """
        # 半波整流
        rectified = np.abs(signal)

        # 零相位低通滤波
        envelope = self._zero_phase_filter(rectified)

        # 额外平滑
        envelope = self._smooth_envelope(envelope)

        return envelope

    def _zero_phase_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        零相位低通滤波（filtfilt）
        避免滤波带来的相位偏移
        """
        # 使用filtfilt实现零相位滤波
        from scipy.signal import filtfilt
        return filtfilt(
            self.filter_b, self.filter_a, signal,
            padlen=min(3 * max(len(self.filter_a), len(self.filter_b)), len(signal) - 1)
        )

    def _smooth_envelope(self, envelope: np.ndarray) -> np.ndarray:
        """
        包络平滑处理

        使用滑动平均平滑高频波动
        """
        # 窗口大小（采样点数）
        window_size = max(
            1,
            int(self.envelope_smooth_window * self.eeg_sr)
        )

        # 滑动平均
        smoothed = uniform_filter1d(
            envelope,
            size=window_size,
            mode='nearest'
        )

        return smoothed

    def normalize_to_unit_range(
        self,
        envelope: np.ndarray,
        clip_outliers: bool = True,
        percentile_clip: float = 99.0
    ) -> np.ndarray:
        """
        归一化到[0,1]区间

        Args:
            envelope: 能量包络
            clip_outliers: 是否裁剪异常值
            percentile_clip: 裁剪百分位

        Returns:
            归一化后的包络
        """
        envelope = envelope.copy()

        if clip_outliers:
            # 裁剪极端值
            clip_value = np.percentile(envelope, percentile_clip)
            envelope = np.clip(envelope, 0, clip_value)

        # Min-Max归一化到[0,1]
        min_val = envelope.min()
        max_val = envelope.max()

        if max_val - min_val < 1e-10:
            warnings.warn("包络方差过小，归一化可能无效")
            return np.zeros_like(envelope)

        normalized = (envelope - min_val) / (max_val - min_val)

        # 缩放到指定范围
        lo, hi = self.normalize_range
        normalized = normalized * (hi - lo) + lo

        return normalized

    def process_full_audio(
        self,
        audio_signal: np.ndarray,
        return_components: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        """
        完整处理流程：将音频转换为EEG尺度的时变权重信号

        流水线：
        1. 时间对齐 -> 2. 能量包络提取 -> 3. 归一化

        Args:
            audio_signal: 原始音频信号
            return_components: 是否返回中间结果

        Returns:
            时变权重信号 w_real(t)，形状 (n_samples,)
            或 (权重信号, 中间结果字典)
        """
        # Step 1: 时间对齐
        aligned = self.time_align(audio_signal)

        # Step 2: 能量包络提取
        envelope = self.extract_energy_envelope(aligned, method='rectify_filter')

        # Step 3: 归一化
        w_real = self.normalize_to_unit_range(envelope)

        if return_components:
            components = {
                'aligned_audio': aligned,
                'energy_envelope': envelope,
                'duration_aligned_seconds': len(aligned) / self.eeg_sr,
                'original_duration_seconds': len(audio_signal) / self.audio_sr,
            }
            return w_real, components

        return w_real


class RealRhythmExtractor:
    """
    真实物理节律提取器

    专门用于处理工厂录音，提取物理冲击的时变控制信号
    """

    def __init__(
        self,
        audio_sampling_rate: float = 44100.0,
        eeg_sampling_rate: float = 128.0,
        shock_detection_threshold: float = 0.7,  # 冲击检测阈值
        min_shock_interval: float = 0.2,  # 秒 - 最小冲击间隔
        seed: Optional[int] = None
    ):
        """
        Args:
            audio_sampling_rate: 音频采样率
            eeg_sampling_rate: EEG采样率
            shock_detection_threshold: 冲击检测阈值（相对于最大能量）
            min_shock_interval: 最小冲击间隔（秒）
            seed: 随机种子
        """
        self.audio_sr = audio_sampling_rate
        self.eeg_sr = eeg_sampling_rate
        self.shock_threshold = shock_detection_threshold
        self.min_shock_interval = min_shock_interval

        self.processor = AcousticSignalProcessor(
            audio_sampling_rate=audio_sampling_rate,
            eeg_sampling_rate=eeg_sampling_rate,
            seed=seed
        )

    def extract_rhythm(
        self,
        audio_signal: np.ndarray,
        detect_shocks: bool = True
    ) -> dict:
        """
        从音频中提取物理节律

        Args:
            audio_signal: 原始音频信号
            detect_shocks: 是否检测冲击时刻

        Returns:
            字典包含：
            - w_real: 时变权重信号
            - shock_times: 冲击时刻（秒）
            - rhythm_stats: 统计信息
        """
        # 处理音频得到权重信号
        w_real, components = self.processor.process_full_audio(
            audio_signal, return_components=True
        )

        result = {
            'w_real': w_real,
            'aligned_duration': len(w_real) / self.eeg_sr,
            'components': components
        }

        # 检测冲击时刻
        if detect_shocks:
            shock_indices = self._detect_shock_peaks(w_real)
            shock_times = shock_indices / self.eeg_sr  # 转换为秒

            result['shock_indices'] = shock_indices
            result['shock_times'] = shock_times
            result['n_shocks'] = len(shock_times)

        # 计算统计信息
        result['rhythm_stats'] = self._compute_rhythm_stats(w_real, result.get('shock_times'))

        return result

    def _detect_shock_peaks(self, w_real: np.ndarray) -> np.ndarray:
        """
        检测包络中的冲击峰值

        使用自适应阈值和最小间隔约束
        """
        from scipy.signal import find_peaks

        # 动态阈值（基于包络统计）
        threshold = self.shock_threshold * w_real.max()

        # 最小距离（采样点数）
        min_distance = int(self.min_shock_interval * self.eeg_sr)

        # 峰值检测
        peaks, properties = find_peaks(
            w_real,
            height=threshold,
            distance=min_distance,
            prominence=0.1  # 突出度阈值
        )

        return peaks

    def _compute_rhythm_stats(self, w_real: np.ndarray, shock_times: Optional[np.ndarray]) -> dict:
        """计算节律统计信息"""
        stats = {
            'mean': float(np.mean(w_real)),
            'std': float(np.std(w_real)),
            'min': float(np.min(w_real)),
            'max': float(np.max(w_real)),
            'median': float(np.median(w_real)),
            'high_intensity_ratio': float(np.mean(w_real > 0.7)),  # 高强度占比
            'low_intensity_ratio': float(np.mean(w_real < 0.2)),    # 低强度占比
        }

        if shock_times is not None and len(shock_times) > 1:
            intervals = np.diff(shock_times)
            stats['shock_rate'] = len(shock_times) / (len(w_real) / self.eeg_sr)  # 冲击/秒
            stats['mean_shock_interval'] = float(np.mean(intervals))
            stats['std_shock_interval'] = float(np.std(intervals))

        return stats


def load_audio_file(
    file_path: str,
    expected_sr: float = 44100.0,
    mono: bool = True
) -> np.ndarray:
    """
    加载音频文件

    支持格式：WAV, MP3, FLAC等（通过scipy.io.wavfile或soundfile）

    Args:
        file_path: 音频文件路径
        expected_sr: 期望采样率（用于验证）
        mono: 是否转换为单声道

    Returns:
        音频信号数组
    """
    import os

    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.wav':
        from scipy.io import wavfile
        sr, data = wavfile.read(file_path)
        # 归一化到[-1, 1]
        if data.dtype != np.float32:
            data = data.astype(np.float32)
            if data.max() > 1.0:
                data = data / 32768.0 if data.dtype == np.int16 else data / 2147483648.0
    else:
        try:
            import soundfile as sf
            data, sr = sf.read(file_path)
        except ImportError:
            raise ImportError("请安装soundfile库来读取非WAV格式音频: pip install soundfile")

    # 验证采样率
    if abs(sr - expected_sr) > 100:
        warnings.warn(
            f"音频采样率 {sr} 与期望 {expected_sr} 差异较大，"
            f"可能影响处理精度"
        )

    # 转换为单声道
    if mono and data.ndim > 1:
        data = np.mean(data, axis=1)

    return data.astype(np.float32)


def create_synthetic_factory_audio(
    duration: float,
    sampling_rate: float = 44100.0,
    machine_base_freq: float = 50.0,  # Hz - 机器基础频率
    n_machines: int = 3,
    shock_probability: float = 0.01,  # 每秒产生冲击的概率
    seed: Optional[int] = None
) -> np.ndarray:
    """
    创建合成工厂音频（用于测试）

    模拟多台机器运转 + 随机冲击的声音

    Args:
        duration: 时长（秒）
        sampling_rate: 采样率
        machine_base_freq: 机器基础频率
        n_machines: 机器数量
        shock_probability: 冲击概率
        seed: 随机种子

    Returns:
        合成音频信号
    """
    rng = np.random.default_rng(seed)

    n_samples = int(duration * sampling_rate)
    t = np.arange(n_samples) / sampling_rate

    # 多台机器的谐波噪声
    signal = np.zeros(n_samples)

    for i in range(n_machines):
        # 基础频率 + 谐波
        freq = machine_base_freq * (0.8 + 0.4 * rng.random())
        amplitude = 0.2 / n_machines * rng.random()

        # 基频
        signal += amplitude * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))

        # 谐波 (2-5次)
        for h in range(2, 6):
            amp = amplitude / (h ** 1.5)
            signal += amp * np.sin(2 * np.pi * h * freq * t + rng.uniform(0, 2 * np.pi))

    # 添加高斯白噪声（机械底噪）
    signal += 0.05 * rng.standard_normal(n_samples)

    # 添加随机冲击
    n_shocks = int(duration * shock_probability * 10)  # 放大概率便于测试
    shock_times = rng.uniform(0, duration, n_shocks)

    for shock_time in shock_times:
        # 冲击位置
        shock_sample = int(shock_time * sampling_rate)
        shock_width = int(0.05 * sampling_rate)  # 50ms冲击

        if shock_sample + shock_width < n_samples:
            # 高斯衰减冲击
            shock_env = np.exp(-((np.arange(shock_width) - shock_width // 2) ** 2) /
                              (2 * (shock_width // 4) ** 2))
            shock_magnitude = 0.5 * rng.uniform(0.5, 1.5)
            shock_signal = shock_magnitude * shock_env * rng.standard_normal(shock_width)
            signal[shock_sample:shock_sample + shock_width] += shock_signal

    # 归一化
    signal = signal / (np.max(np.abs(signal)) + 1e-10) * 0.9

    return signal.astype(np.float32)


# 便捷函数
def audio_to_eeg_control_signal(
    audio_path: Optional[str] = None,
    audio_signal: Optional[np.ndarray] = None,
    audio_sampling_rate: float = 44100.0,
    eeg_sampling_rate: float = 128.0,
    return_shock_times: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    将音频转换为EEG控制信号的便捷函数

    Args:
        audio_path: 音频文件路径（与audio_signal二选一）
        audio_signal: 直接传入音频信号
        audio_sampling_rate: 音频采样率
        eeg_sampling_rate: EEG采样率
        return_shock_times: 是否返回冲击时刻

    Returns:
        时变权重信号 w_real(t)
        如果return_shock_times=True，还返回冲击时刻数组
    """
    # 加载音频
    if audio_path is not None:
        audio = load_audio_file(audio_path, audio_sampling_rate)
    elif audio_signal is not None:
        audio = audio_signal
    else:
        raise ValueError("必须提供audio_path或audio_signal")

    # 提取节律
    extractor = RealRhythmExtractor(
        audio_sampling_rate=audio_sampling_rate,
        eeg_sampling_rate=eeg_sampling_rate
    )

    result = extractor.extract_rhythm(audio)

    if return_shock_times:
        return result['w_real'], result.get('shock_times', np.array([]))
    return result['w_real']
