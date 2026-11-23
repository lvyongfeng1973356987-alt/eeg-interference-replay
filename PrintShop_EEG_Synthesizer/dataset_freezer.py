"""
阶段三：数据集的规模化生成与"物理冻结"
=====================================

核心功能：
1. 全量矩阵化生成 - 对SEED/DEAP全部样本生成四个SNR级别数据
2. 数据固化 (Freezing) - 将生成的数据永久保存为.npy格式
3. 学术规范保证 - 一旦生成不可修改，确保所有实验对比在同一考卷上

SNR级别设计：
- Clean: 纯净EEG，无噪声
- SNR = 0dB: 信噪比1:1，严重噪声
- SNR = -5dB: 信噪比约0.56:1，极端噪声
- SNR = -10dB: 信噪比0.1:1，极限抗噪测试

Author: EEG_NOISE_MODEL_LYF Team
"""

import os
import numpy as np
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
from tqdm import tqdm

from .real_rhythm_generator import RealRhythmSynthesizer, create_real_rhythm_synthesizer
from .synthesizer import EEGSynthesizer, SynthesizerConfig


@dataclass
class DatasetManifest:
    """数据集清单 - 记录数据集的所有元信息"""
    dataset_name: str
    creation_date: str
    generator_type: str  # 'real_rhythm' or 'composite'

    # 音频信息（如果使用真实节律）
    audio_path: Optional[str] = None
    audio_hash: Optional[str] = None
    audio_duration_seconds: Optional[float] = None

    # 数据集规格
    n_samples_total: int = 0
    n_channels: int = 0
    n_timesteps: int = 0
    sampling_rate: int = 128

    # SNR级别
    snr_levels: List[str] = None

    # 各SNR级别样本数
    samples_per_level: Dict[str, int] = None

    # 类别分布
    class_distribution: Dict[str, Dict[int, int]] = None

    # 生成器配置哈希（用于复现验证）
    config_hash: Optional[str] = None

    # 状态标志
    is_frozen: bool = False
    frozen_timestamp: Optional[str] = None

    def __post_init__(self):
        if self.snr_levels is None:
            self.snr_levels = []
        if self.samples_per_level is None:
            self.samples_per_level = {}
        if self.class_distribution is None:
            self.class_distribution = {}

    def to_dict(self) -> dict:
        """转换为字典"""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'DatasetManifest':
        """从字典创建"""
        return cls(**d)


class FrozenNoiseDatasetGenerator:
    """
    冻结噪声数据集生成器

    功能：
    1. 加载清洁EEG数据集
    2. 生成四个SNR级别的含噪版本
    3. 永久保存为.npy格式
    4. 生成清单文件记录所有元信息
    """

    def __init__(
        self,
        dataset_name: str,
        output_dir: str,
        sampling_rate: int = 128,
        snr_levels: List[float] = None,
        generator_type: str = 'real_rhythm',
        seed: int = 42,
        **synthesizer_kwargs
    ):
        """
        Args:
            dataset_name: 数据集名称 ('SEED', 'DEAP')
            output_dir: 输出根目录
            sampling_rate: EEG采样率
            snr_levels: SNR级别列表
            generator_type: 噪声生成器类型
            seed: 随机种子
            **synthesizer_kwargs: 传递给合成器的参数
        """
        self.dataset_name = dataset_name.upper()
        self.output_dir = output_dir
        self.sampling_rate = sampling_rate
        self.snr_levels = snr_levels or [0, -5, -10]
        self.generator_type = generator_type
        self.seed = seed
        self.synthesizer_kwargs = synthesizer_kwargs

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 初始化清单
        self.manifest = DatasetManifest(
            dataset_name=self.dataset_name,
            creation_date=datetime.now().isoformat(),
            generator_type=generator_type,
            snr_levels=[f"snr_{s}dB" if s != 'clean' else 'clean' for s in self.snr_levels + ['clean'] if s != 0],
        )

        # 计算配置哈希（用于验证可复现性）
        self._compute_config_hash()

    def _compute_config_hash(self):
        """计算配置哈希"""
        config_str = json.dumps({
            'dataset_name': self.dataset_name,
            'sampling_rate': self.sampling_rate,
            'snr_levels': self.snr_levels,
            'generator_type': self.generator_type,
            'seed': self.seed,
            'kwargs': self.synthesizer_kwargs
        }, sort_keys=True)

        self.manifest.config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]

    def _load_clean_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加载清洁数据集

        子类应重写此方法以加载特定数据集

        Returns:
            (X, y, subject_ids)
        """
        # 默认实现：检查预定义路径
        data_paths = {
            'SEED': os.path.join(self.output_dir, '..', 'RBTransformer', 'Preprocessed_EEG'),
            'DEAP': os.path.join(self.output_dir, '..', 'RBTransformer', 'data_preprocessed_python'),
        }

        data_dir = data_paths.get(self.dataset_name)
        if data_dir is None or not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"数据集 {self.dataset_name} 路径不存在: {data_dir}\n"
                f"请确保数据集已预处理并放置在正确位置"
            )

        # 尝试加载标准格式
        npy_dir = os.path.join(self.output_dir, '..', 'clean_data')
        if os.path.exists(npy_dir):
            X_path = os.path.join(npy_dir, f'{self.dataset_name.lower()}_bde_X.npy')
            y_path = os.path.join(npy_dir, f'{self.dataset_name.lower()}_bde_y.npy')
            subj_path = os.path.join(npy_dir, f'{self.dataset_name.lower()}_bde_subjects.npy')

            if all(os.path.exists(p) for p in [X_path, y_path]):
                X = np.load(X_path)
                y = np.load(y_path)
                subjects = np.load(subj_path) if os.path.exists(subj_path) else None
                print(f"从 {npy_dir} 加载清洁数据: {X.shape}")
                return X, y, subjects

        raise FileNotFoundError(
            f"无法找到清洁数据集。"
            f"请先运行预处理脚本生成 {self.dataset_name} 的BDE特征。"
        )

    def _create_synthesizer(self, **extra_kwargs) -> Union[RealRhythmSynthesizer, EEGSynthesizer]:
        """创建合成器"""
        kwargs = {**self.synthesizer_kwargs, **extra_kwargs}

        if self.generator_type == 'real_rhythm':
            return create_real_rhythm_synthesizer(
                sampling_rate=self.sampling_rate,
                seed=self.seed,
                **kwargs
            )
        else:
            config = SynthesizerConfig(
                sampling_rate=self.sampling_rate,
                snr_values_db=self.snr_levels,
                seed=self.seed,
                **kwargs
            )
            return EEGSynthesizer(config)

    def generate_and_freeze(
        self,
        clean_X: Optional[np.ndarray] = None,
        clean_y: Optional[np.ndarray] = None,
        clean_subjects: Optional[np.ndarray] = None,
        audio_path: Optional[str] = None,
        w_real: Optional[np.ndarray] = None,
        force_regenerate: bool = False
    ) -> Dict[str, str]:
        """
        生成并冻结数据集

        Args:
            clean_X: 清洁EEG数据 (N, n_channels, n_samples)
            clean_y: 标签 (N,)
            clean_subjects: 被试ID (N,)
            audio_path: 音频文件路径（用于真实节律生成）
            w_real: 预计算的真实节律权重
            force_regenerate: 是否强制重新生成

        Returns:
            保存的文件路径字典
        """
        # 加载数据
        if clean_X is None:
            clean_X, clean_y, clean_subjects = self._load_clean_dataset()

        n_samples = len(clean_X)
        n_channels, n_timesteps = clean_X.shape[1], clean_X.shape[2]

        # 更新清单
        self.manifest.n_samples_total = n_samples
        self.manifest.n_channels = n_channels
        self.manifest.n_timesteps = n_timesteps
        self.manifest.sampling_rate = self.sampling_rate

        print(f"\n{'='*60}")
        print(f"开始生成冻结数据集: {self.dataset_name}")
        print(f"{'='*60}")
        print(f"样本数: {n_samples}, 通道: {n_channels}, 时间步: {n_timesteps}")
        print(f"SNR级别: {self.snr_levels}")
        print(f"生成器类型: {self.generator_type}")
        print(f"{'='*60}\n")

        saved_files = {}

        # Step 1: 保存Clean数据
        clean_dir = os.path.join(self.output_dir, 'clean')
        os.makedirs(clean_dir, exist_ok=True)

        clean_X_path = os.path.join(clean_dir, 'X.npy')
        clean_y_path = os.path.join(clean_dir, 'y.npy')
        clean_subj_path = os.path.join(clean_dir, 'subjects.npy')

        np.save(clean_X_path, clean_X.astype(np.float32))
        np.save(clean_y_path, clean_y)
        if clean_subjects is not None:
            np.save(clean_subj_path, clean_subjects)

        saved_files['clean_X'] = clean_X_path
        saved_files['clean_y'] = clean_y_path
        saved_files['clean_subjects'] = clean_subj_path

        # 统计Clean类别分布
        unique, counts = np.unique(clean_y, return_counts=True)
        self.manifest.class_distribution['clean'] = {
            int(u): int(c) for u, c in zip(unique, counts)
        }
        self.manifest.samples_per_level['clean'] = len(clean_y)

        print(f"[✓] Clean数据已保存: {clean_dir}")

        # Step 2: 生成各SNR级别数据
        for snr_db in self.snr_levels:
            snr_dir = os.path.join(self.output_dir, f'snr_{snr_db}dB')
            snr_X_path = os.path.join(snr_dir, 'X.npy')
            snr_y_path = os.path.join(snr_dir, 'y.npy')
            snr_subj_path = os.path.join(snr_dir, 'subjects.npy')

            # 检查是否已存在且不强制重新生成
            if not force_regenerate and os.path.exists(snr_X_path):
                print(f"[SKIP] {snr_db}dB数据已存在，跳过生成")
                saved_files[f'snr_{snr_db}dB_X'] = snr_X_path
                saved_files[f'snr_{snr_db}dB_y'] = snr_y_path

                # 更新清单
                existing_X = np.load(snr_X_path)
                self.manifest.samples_per_level[f'snr_{snr_db}dB'] = len(existing_X)
                continue

            os.makedirs(snr_dir, exist_ok=True)

            print(f"\n生成 SNR = {snr_db}dB ...")

            # 创建合成器
            if self.generator_type == 'real_rhythm' and (audio_path or w_real is not None):
                synthesizer = create_real_rhythm_synthesizer(
                    audio_path=audio_path,
                    w_real=w_real,
                    sampling_rate=self.sampling_rate,
                    seed=self.seed
                )
            else:
                synthesizer = self._create_synthesizer()

            # 批量合成
            noisy_data = []
            noisy_labels = []
            noisy_subjects = []

            for i in tqdm(range(n_samples), desc=f"  SNR={snr_db}dB"):
                clean = clean_X[i]
                noisy = synthesizer.synthesize_single(clean, snr_db)
                noisy_data.append(noisy)
                noisy_labels.append(clean_y[i])

                if clean_subjects is not None:
                    noisy_subjects.append(clean_subjects[i])

            noisy_X = np.array(noisy_data, dtype=np.float32)
            noisy_y = np.array(noisy_labels)
            noisy_subj = np.array(noisy_subjects) if clean_subjects is not None else None

            # 保存
            np.save(snr_X_path, noisy_X)
            np.save(snr_y_path, noisy_y)
            if noisy_subj is not None:
                np.save(snr_subj_path, noisy_subj)

            saved_files[f'snr_{snr_db}dB_X'] = snr_X_path
            saved_files[f'snr_{snr_db}dB_y'] = snr_y_path
            saved_files[f'snr_{snr_db}dB_subjects'] = snr_subj_path

            # 统计
            unique, counts = np.unique(noisy_y, return_counts=True)
            self.manifest.class_distribution[f'snr_{snr_db}dB'] = {
                int(u): int(c) for u, c in zip(unique, counts)
            }
            self.manifest.samples_per_level[f'snr_{snr_db}dB'] = len(noisy_y)

            print(f"  [✓] SNR={snr_db}dB 完成: {noisy_X.shape}")

        # Step 3: 生成并保存清单
        self.manifest.is_frozen = True
        self.manifest.frozen_timestamp = datetime.now().isoformat()

        manifest_path = os.path.join(self.output_dir, 'dataset_manifest.json')
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(self.manifest.to_dict(), f, indent=2, ensure_ascii=False)

        saved_files['manifest'] = manifest_path

        # Step 4: 保存生成器配置（用于复现）
        config_path = os.path.join(self.output_dir, 'generator_config.json')
        generator_config = {
            'dataset_name': self.dataset_name,
            'sampling_rate': self.sampling_rate,
            'snr_levels': self.snr_levels,
            'generator_type': self.generator_type,
            'seed': self.seed,
            'synthesizer_kwargs': self.synthesizer_kwargs,
            'config_hash': self.manifest.config_hash
        }
        with open(config_path, 'w') as f:
            json.dump(generator_config, f, indent=2)

        saved_files['config'] = config_path

        print(f"\n{'='*60}")
        print(f"✓ 数据集冻结完成!")
        print(f"  输出目录: {self.output_dir}")
        print(f"  清单文件: {manifest_path}")
        print(f"  配置哈希: {self.manifest.config_hash}")
        print(f"{'='*60}\n")

        return saved_files

    def verify_frozen_dataset(self) -> Dict[str, bool]:
        """
        验证冻结数据集的完整性

        Returns:
            验证结果字典
        """
        results = {}
        all_ok = True

        # 检查目录
        results['output_dir_exists'] = os.path.exists(self.output_dir)

        # 检查Clean数据
        clean_dir = os.path.join(self.output_dir, 'clean')
        results['clean_dir_exists'] = os.path.exists(clean_dir)
        if results['clean_dir_exists']:
            for fname in ['X.npy', 'y.npy']:
                results[f'clean_{fname}_exists'] = os.path.exists(
                    os.path.join(clean_dir, fname)
                )
                if not results[f'clean_{fname}_exists']:
                    all_ok = False

        # 检查各SNR级别数据
        for snr_db in self.snr_levels:
            snr_dir = os.path.join(self.output_dir, f'snr_{snr_db}dB')
            results[f'snr_{snr_db}dB_dir_exists'] = os.path.exists(snr_dir)

            if results[f'snr_{snr_db}dB_dir_exists']:
                for fname in ['X.npy', 'y.npy']:
                    path = os.path.join(snr_dir, fname)
                    results[f'snr_{snr_db}dB_{fname}_exists'] = os.path.exists(path)
                    if not results[f'snr_{snr_db}dB_{fname}_exists']:
                        all_ok = False

        # 检查清单
        manifest_path = os.path.join(self.output_dir, 'dataset_manifest.json')
        results['manifest_exists'] = os.path.exists(manifest_path)

        if results['manifest_exists']:
            with open(manifest_path) as f:
                manifest = DatasetManifest.from_dict(json.load(f))
            results['is_frozen'] = manifest.is_frozen
            if not manifest.is_frozen:
                warnings.warn("数据集清单标记为未冻结")
                all_ok = False

        results['all_ok'] = all_ok
        return results


class FrozenDatasetLoader:
    """
    冻结数据集加载器

    提供对已冻结数据集的只读访问
    """

    def __init__(self, frozen_dir: str, dataset_name: str = None):
        """
        Args:
            frozen_dir: 冻结数据集根目录
            dataset_name: 数据集名称（从清单读取）
        """
        self.frozen_dir = frozen_dir

        # 加载清单
        manifest_path = os.path.join(frozen_dir, 'dataset_manifest.json')
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                self.manifest = DatasetManifest.from_dict(json.load(f))
            self.dataset_name = self.manifest.dataset_name
        else:
            warnings.warn("未找到数据集清单")
            self.manifest = None
            self.dataset_name = dataset_name or 'Unknown'

        # 加载配置
        config_path = os.path.join(frozen_dir, 'generator_config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = {}

    def load(self, snr_level: str = 'clean') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加载指定SNR级别的数据

        Args:
            snr_level: SNR级别 ('clean', 'snr_0dB', 'snr_-5dB', 'snr_-10dB')

        Returns:
            (X, y, subjects) 元组
        """
        if snr_level == 'clean':
            data_dir = os.path.join(self.frozen_dir, 'clean')
        else:
            data_dir = os.path.join(self.frozen_dir, snr_level)

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        X = np.load(os.path.join(data_dir, 'X.npy'))
        y = np.load(os.path.join(data_dir, 'y.npy'))

        subj_path = os.path.join(data_dir, 'subjects.npy')
        subjects = np.load(subj_path) if os.path.exists(subj_path) else None

        return X, y, subjects

    def load_all_levels(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        加载所有SNR级别的数据

        Returns:
            {snr_level: (X, y, subjects)} 字典
        """
        levels = ['clean']
        if self.manifest:
            levels.extend(self.manifest.snr_levels)

        results = {}
        for level in levels:
            try:
                results[level] = self.load(level)
            except FileNotFoundError:
                warnings.warn(f"无法加载 {level} 数据")

        return results


def generate_seed_dataset(
    output_dir: str,
    audio_path: Optional[str] = None,
    w_real: Optional[np.ndarray] = None,
    snr_levels: List[float] = [0, -5, -10],
    seed: int = 42,
    force_regenerate: bool = False
) -> Dict[str, str]:
    """
    生成SEED数据集的冻结噪声版本

    Args:
        output_dir: 输出目录
        audio_path: 音频文件路径
        w_real: 预计算的真实节律
        snr_levels: SNR级别
        seed: 随机种子
        force_regenerate: 是否强制重新生成

    Returns:
        保存的文件路径
    """
    generator = FrozenNoiseDatasetGenerator(
        dataset_name='SEED',
        output_dir=output_dir,
        generator_type='real_rhythm' if (audio_path or w_real is not None) else 'composite',
        snr_levels=snr_levels,
        seed=seed
    )

    return generator.generate_and_freeze(
        audio_path=audio_path,
        w_real=w_real,
        force_regenerate=force_regenerate
    )


def generate_deap_dataset(
    output_dir: str,
    audio_path: Optional[str] = None,
    w_real: Optional[np.ndarray] = None,
    snr_levels: List[float] = [0, -5, -10],
    seed: int = 42,
    force_regenerate: bool = False
) -> Dict[str, str]:
    """
    生成DEAP数据集的冻结噪声版本

    Args:
        output_dir: 输出目录
        audio_path: 音频文件路径
        w_real: 预计算的真实节律
        snr_levels: SNR级别
        seed: 随机种子
        force_regenerate: 是否强制重新生成

    Returns:
        保存的文件路径
    """
    generator = FrozenNoiseDatasetGenerator(
        dataset_name='DEAP',
        output_dir=output_dir,
        generator_type='real_rhythm' if (audio_path or w_real is not None) else 'composite',
        snr_levels=snr_levels,
        seed=seed
    )

    return generator.generate_and_freeze(
        audio_path=audio_path,
        w_real=w_real,
        force_regenerate=force_regenerate
    )


def load_frozen_dataset(
    frozen_dir: str,
    snr_level: str = 'clean'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载冻结数据集的便捷函数

    Args:
        frozen_dir: 冻结数据集目录
        snr_level: SNR级别

    Returns:
        (X, y, subjects)
    """
    loader = FrozenDatasetLoader(frozen_dir)
    return loader.load(snr_level)
