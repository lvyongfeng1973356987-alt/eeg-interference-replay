"""
PrintShop_EEG_Synthesizer: 真实眼电肌电伪影加载器模块
=====================================================

从EDND和EARA数据集加载真实的EMG/EOG伪影信号。

数据集说明：
- EDND (EEGdenoiseNet Dataset): Zhang et al. 2021, J. Neural Engineering
  包含真实录制的眼电(EOG)、肌电(EMG)和清洁EEG数据
- EARA: 情绪识别相关伪影数据集

Author: PrintShop_EEG_Synthesizer Team
"""

import os
import glob
import numpy as np
import scipy.io as scio
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings


class ArtifactType(Enum):
    """伪影类型枚举"""
    EMG = "emg"
    EOG = "eog"
    CLEAN_EEG = "clean_eeg"
    CONTAMINATED = "contaminated"


@dataclass
class ArtifactMetadata:
    """伪影元数据"""
    artifact_type: ArtifactType
    dataset_name: str
    n_channels: int
    n_samples: int
    sampling_rate: int
    duration: float
    file_path: str


class EDNDLoader:
    """
    EDND (EEGdenoiseNet Dataset) 加载器

    数据集结构:
    - data/data_decreasing_SNR/: 清洁EEG数据
    - data/data_contaminated/: 被污染的EEG数据
    - data/EOG/: 眼电伪影数据
    - data/EMG/: 肌电伪影数据

    采样率: 256Hz
    通道数: 64 (与标准EEG系统兼容)
    """

    def __init__(self, data_dir: str = "DATA/EDND",
                 sampling_rate: int = 256):
        """
        Args:
            data_dir: EDND数据集根目录
            sampling_rate: 原始采样率 (256Hz)
        """
        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.emg_dir = os.path.join(data_dir, "data", "EMG")
        self.eog_dir = os.path.join(data_dir, "data", "EOG")
        self.clean_dir = os.path.join(data_dir, "data", "data_decreasing_SNR")
        self.contaminated_dir = os.path.join(data_dir, "data", "data_contaminated")

    def _list_mat_files(self, directory: str) -> List[str]:
        """列出目录下的所有.mat文件"""
        if not os.path.exists(directory):
            warnings.warn(f"目录不存在: {directory}")
            return []
        return sorted(glob.glob(os.path.join(directory, "*.mat")))

    def _load_mat_variable(self, file_path: str) -> np.ndarray:
        """加载.mat文件中的数据"""
        try:
            data = scio.loadmat(file_path)
            # 查找数据变量（通常是第一个非元数据的数组）
            for key in data.keys():
                if not key.startswith('__'):
                    arr = data[key]
                    if isinstance(arr, np.ndarray):
                        return arr
            raise ValueError(f"文件中未找到有效数据: {file_path}")
        except Exception as e:
            warnings.warn(f"加载文件失败 {file_path}: {e}")
            raise

    def load_emg_artifacts(self, resample_to: Optional[int] = 128) -> Dict[str, np.ndarray]:
        """
        加载所有EMG伪影数据

        Args:
            resample_to: 如果指定，则将数据重采样到该采样率

        Returns:
            字典，键为文件名，值为EMG数据 (n_channels, n_samples)
        """
        files = self._list_mat_files(self.emg_dir)
        if not files:
            warnings.warn(f"EMG目录为空或不存在: {self.emg_dir}")

        emg_data = {}
        for fpath in files:
            try:
                data = self._load_mat_variable(fpath)
                if resample_to and resample_to != self.sampling_rate:
                    data = self._resample(data, resample_to)
                name = os.path.splitext(os.path.basename(fpath))[0]
                emg_data[name] = data
            except Exception as e:
                print(f"跳过EMG文件 {fpath}: {e}")

        return emg_data

    def load_eog_artifacts(self, resample_to: Optional[int] = 128) -> Dict[str, np.ndarray]:
        """
        加载所有EOG伪影数据

        Args:
            resample_to: 如果指定，则将数据重采样到该采样率

        Returns:
            字典，键为文件名，值为EOG数据 (n_channels, n_samples)
        """
        files = self._list_mat_files(self.eog_dir)
        if not files:
            warnings.warn(f"EOG目录为空或不存在: {self.eog_dir}")

        eog_data = {}
        for fpath in files:
            try:
                data = self._load_mat_variable(fpath)
                if resample_to and resample_to != self.sampling_rate:
                    data = self._resample(data, resample_to)
                name = os.path.splitext(os.path.basename(fpath))[0]
                eog_data[name] = data
            except Exception as e:
                print(f"跳过EOG文件 {fpath}: {e}")

        return eog_data

    def load_clean_eeg(self, resample_to: Optional[int] = 128) -> Dict[str, np.ndarray]:
        """
        加载清洁EEG数据（用于验证）

        Args:
            resample_to: 如果指定，则将数据重采样到该采样率

        Returns:
            字典，键为文件名，值为清洁EEG数据
        """
        files = self._list_mat_files(self.clean_dir)
        clean_data = {}

        for fpath in files:
            try:
                data = self._load_mat_variable(fpath)
                if resample_to and resample_to != self.sampling_rate:
                    data = self._resample(data, resample_to)
                name = os.path.splitext(os.path.basename(fpath))[0]
                clean_data[name] = data
            except Exception as e:
                print(f"跳过清洁EEG文件 {fpath}: {e}")

        return clean_data

    def _resample(self, data: np.ndarray, target_fs: int) -> np.ndarray:
        """
        重采样数据到目标采样率

        使用线性插值实现下采样/上采样

        Args:
            data: 输入数据 (n_channels, n_samples)
            target_fs: 目标采样率

        Returns:
            重采样后的数据
        """
        from scipy.interpolate import interp1d

        n_channels, n_samples = data.shape
        duration = n_samples / self.sampling_rate
        n_target = int(duration * target_fs)

        new_indices = np.linspace(0, n_samples - 1, n_target)
        old_indices = np.arange(n_samples)

        resampled = np.zeros((n_channels, n_target))
        for ch in range(n_channels):
            f = interp1d(old_indices, data[ch], kind='linear')
            resampled[ch] = f(new_indices)

        return resampled

    def get_artifact_info(self) -> List[ArtifactMetadata]:
        """获取所有伪影的元数据信息"""
        artifacts = []

        # EMG
        for fpath in self._list_mat_files(self.emg_dir):
            try:
                data = self._load_mat_variable(fpath)
                artifacts.append(ArtifactMetadata(
                    artifact_type=ArtifactType.EMG,
                    dataset_name="EDND",
                    n_channels=data.shape[0] if data.ndim > 1 else 1,
                    n_samples=data.shape[1] if data.ndim > 1 else len(data),
                    sampling_rate=self.sampling_rate,
                    duration=data.shape[1] / self.sampling_rate if data.ndim > 1 else len(data) / self.sampling_rate,
                    file_path=fpath
                ))
            except:
                pass

        # EOG
        for fpath in self._list_mat_files(self.eog_dir):
            try:
                data = self._load_mat_variable(fpath)
                artifacts.append(ArtifactMetadata(
                    artifact_type=ArtifactType.EOG,
                    dataset_name="EDND",
                    n_channels=data.shape[0] if data.ndim > 1 else 1,
                    n_samples=data.shape[1] if data.ndim > 1 else len(data),
                    sampling_rate=self.sampling_rate,
                    duration=data.shape[1] / self.sampling_rate if data.ndim > 1 else len(data) / self.sampling_rate,
                    file_path=fpath
                ))
            except:
                pass

        return artifacts


class EARALoader:
    """
    EARA (Emotion Artifact) 数据集加载器

    数据集包含:
    - Pure_Data.mat: 清洁EEG数据
    - Contaminated_Data.mat: 被污染的EEG数据
    - HEOG.mat: 水平眼电数据

    采样率: 1000Hz (需要降采样到128Hz)
    """

    def __init__(self, data_dir: str = "DATA/EARA",
                 sampling_rate: int = 1000):
        """
        Args:
            data_dir: EARA数据集根目录
            sampling_rate: 原始采样率 (1000Hz)
        """
        self.data_dir = data_dir
        self.sampling_rate = sampling_rate

    def load_pure_data(self, resample_to: int = 128) -> Optional[np.ndarray]:
        """
        加载清洁EEG数据

        Args:
            resample_to: 目标采样率

        Returns:
            清洁EEG数据 (n_channels, n_samples)
        """
        pure_path = os.path.join(self.data_dir, "Pure_Data.mat")
        if not os.path.exists(pure_path):
            warnings.warn(f"文件不存在: {pure_path}")
            return None

        try:
            data = scio.loadmat(pure_path)
            # 查找数据变量
            for key in data.keys():
                if not key.startswith('__') and isinstance(data[key], np.ndarray):
                    arr = data[key]
                    if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:  # 应该是 (channels, samples)
                        if resample_to != self.sampling_rate:
                            return self._resample(arr, resample_to)
                        return arr
            return None
        except Exception as e:
            warnings.warn(f"加载Pure_Data.mat失败: {e}")
            return None

    def load_contaminated_data(self, resample_to: int = 128) -> Optional[np.ndarray]:
        """
        加载被污染的EEG数据

        Args:
            resample_to: 目标采样率

        Returns:
            被污染的EEG数据 (n_channels, n_samples)
        """
        cont_path = os.path.join(self.data_dir, "Contaminated_Data.mat")
        if not os.path.exists(cont_path):
            warnings.warn(f"文件不存在: {cont_path}")
            return None

        try:
            data = scio.loadmat(cont_path)
            for key in data.keys():
                if not key.startswith('__') and isinstance(data[key], np.ndarray):
                    arr = data[key]
                    if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                        if resample_to != self.sampling_rate:
                            return self._resample(arr, resample_to)
                        return arr
            return None
        except Exception as e:
            warnings.warn(f"加载Contaminated_Data.mat失败: {e}")
            return None

    def load_heog(self, resample_to: int = 128) -> Optional[np.ndarray]:
        """
        加载水平眼电(HEOG)数据

        Args:
            resample_to: 目标采样率

        Returns:
            HEOG数据 (n_channels, n_samples)
        """
        heog_path = os.path.join(self.data_dir, "HEOG.mat")
        if not os.path.exists(heog_path):
            warnings.warn(f"文件不存在: {heog_path}")
            return None

        try:
            data = scio.loadmat(heog_path)
            for key in data.keys():
                if not key.startswith('__') and isinstance(data[key], np.ndarray):
                    arr = data[key]
                    if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                        if resample_to != self.sampling_rate:
                            return self._resample(arr, resample_to)
                        return arr
            return None
        except Exception as e:
            warnings.warn(f"加载HEOG.mat失败: {e}")
            return None

    def _resample(self, data: np.ndarray, target_fs: int) -> np.ndarray:
        """重采样数据"""
        from scipy.interpolate import interp1d

        n_channels, n_samples = data.shape
        duration = n_samples / self.sampling_rate
        n_target = int(duration * target_fs)

        new_indices = np.linspace(0, n_samples - 1, n_target)
        old_indices = np.arange(n_samples)

        resampled = np.zeros((n_channels, n_target))
        for ch in range(n_channels):
            f = interp1d(old_indices, data[ch], kind='linear')
            resampled[ch] = f(new_indices)

        return resampled

    def get_all_artifacts(self, resample_to: int = 128) -> Dict[str, np.ndarray]:
        """
        加载所有可用的伪影数据

        Args:
            resample_to: 目标采样率

        Returns:
            包含所有伪影数据的字典
        """
        artifacts = {}

        pure = self.load_pure_data(resample_to)
        if pure is not None:
            artifacts['pure_eeg'] = pure

        contaminated = self.load_contaminated_data(resample_to)
        if contaminated is not None:
            artifacts['contaminated_eeg'] = contaminated

        heog = self.load_heog(resample_to)
        if heog is not None:
            artifacts['heog'] = heog

        return artifacts


class ArtifactLoader:
    """
    统一的伪影加载器接口

    同时支持EDND和EARA数据集，提供统一的API
    """

    def __init__(self,
                 ednd_dir: str = "DATA/EDND",
                 eara_dir: str = "DATA/EARA",
                 target_sampling_rate: int = 128):
        """
        Args:
            ednd_dir: EDND数据集路径
            eara_dir: EARA数据集路径
            target_sampling_rate: 目标采样率 (与SEED/DEAP预处理后一致)
        """
        self.target_sampling_rate = target_sampling_rate
        self.ednd = EDNDLoader(ednd_dir)
        self.eara = EARALoader(eara_dir)

        # 缓存已加载的伪影
        self._emg_cache: Optional[Dict[str, np.ndarray]] = None
        self._eog_cache: Optional[Dict[str, np.ndarray]] = None
        self._eara_cache: Optional[Dict[str, np.ndarray]] = None

    def load_all_emg(self, use_cache: bool = True) -> Dict[str, np.ndarray]:
        """
        加载所有EMG伪影（从EDND）

        Args:
            use_cache: 是否使用缓存

        Returns:
            EMG伪影字典
        """
        if use_cache and self._emg_cache is not None:
            return self._emg_cache

        self._emg_cache = self.ednd.load_emg_artifacts(self.target_sampling_rate)
        return self._emg_cache

    def load_all_eog(self, use_cache: bool = True) -> Dict[str, np.ndarray]:
        """
        加载所有EOG伪影（从EDND和EARA）

        Args:
            use_cache: 是否使用缓存

        Returns:
            EOG伪影字典
        """
        if use_cache and self._eog_cache is not None:
            return self._eog_cache

        # 从EDND加载EOG
        ednd_eog = self.ednd.load_eog_artifacts(self.target_sampling_rate)

        # 从EARA加载HEOG
        eara_artifacts = self.eara.get_all_artifacts(self.target_sampling_rate)
        if 'heog' in eara_artifacts:
            ednd_eog['eara_heog'] = eara_artifacts['heog']

        self._eog_cache = ednd_eog
        return self._eog_cache

    def get_random_emg_segment(self, n_samples: int,
                               n_channels: Optional[int] = None,
                               seed: Optional[int] = None) -> np.ndarray:
        """
        获取随机EMG片段

        Args:
            n_samples: 需要的采样点数
            n_channels: 需要的通道数（如果为None则使用原始通道数）
            seed: 随机种子

        Returns:
            EMG片段 (n_channels, n_samples) 或 (n_samples,)
        """
        rng = np.random.default_rng(seed)
        emg_data = self.load_all_emg()

        if not emg_data:
            warnings.warn("没有可用的EMG数据，返回零信号")
            if n_channels:
                return np.zeros((n_channels, n_samples))
            return np.zeros(n_samples)

        # 随机选择一个EMG样本
        name = rng.choice(list(emg_data.keys()))
        emg = emg_data[name]  # (orig_channels, orig_samples)

        # 提取随机片段
        if emg.shape[1] > n_samples:
            start = rng.integers(0, emg.shape[1] - n_samples)
            emg = emg[:, start:start + n_samples]
        else:
            # 重复以达到所需长度
            repeats = int(np.ceil(n_samples / emg.shape[1]))
            emg = np.tile(emg, (1, repeats))[:, :n_samples]

        # 调整通道数
        if n_channels and n_channels != emg.shape[0]:
            emg = self._adjust_channels(emg, n_channels, rng)

        return emg

    def get_random_eog_segment(self, n_samples: int,
                                n_channels: Optional[int] = None,
                                seed: Optional[int] = None) -> np.ndarray:
        """
        获取随机EOG片段

        Args:
            n_samples: 需要的采样点数
            n_channels: 需要的通道数
            seed: 随机种子

        Returns:
            EOG片段
        """
        rng = np.random.default_rng(seed)
        eog_data = self.load_all_eog()

        if not eog_data:
            warnings.warn("没有可用的EOG数据，返回零信号")
            if n_channels:
                return np.zeros((n_channels, n_samples))
            return np.zeros(n_samples)

        name = rng.choice(list(eog_data.keys()))
        eog = eog_data[name]

        if eog.shape[1] > n_samples:
            start = rng.integers(0, eog.shape[1] - n_samples)
            eog = eog[:, start:start + n_samples]
        else:
            repeats = int(np.ceil(n_samples / eog.shape[1]))
            eog = np.tile(eog, (1, repeats))[:, :n_samples]

        if n_channels and n_channels != eog.shape[0]:
            eog = self._adjust_channels(eog, n_channels, rng)

        return eog

    def _adjust_channels(self, data: np.ndarray, target_channels: int,
                         rng: np.random.Generator) -> np.ndarray:
        """调整数据通道数"""
        current_channels = data.shape[0]

        if target_channels <= current_channels:
            # 随机选择部分通道
            indices = rng.choice(current_channels, target_channels, replace=False)
            return data[indices]
        else:
            # 复制部分通道
            indices = rng.choice(current_channels, target_channels, replace=True)
            return data[indices]

    def clear_cache(self):
        """清除缓存"""
        self._emg_cache = None
        self._eog_cache = None
        self._eara_cache = None

    def get_dataset_info(self) -> Dict[str, any]:
        """
        获取已加载数据集的信息

        Returns:
            数据集信息字典
        """
        info = {
            'target_sampling_rate': self.target_sampling_rate,
            'ednd_available': os.path.exists(self.ednd.data_dir),
            'eara_available': os.path.exists(self.eara.data_dir),
            'emg_files': len(self.load_all_emg()) if self.ednd.emg_dir else 0,
            'eog_files': len(self.load_all_eog()) if self.ednd.eog_dir else 0,
        }
        return info
