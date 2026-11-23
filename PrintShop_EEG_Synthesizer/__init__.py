"""
PrintShop_EEG_Synthesizer
=========================

印刷车间环境EEG噪声合成工具包

该工具包用于生成工业环境下的合成EEG噪声数据集，支持：
- 物理噪声建模（PLI、基线漂移、机械冲击）
- 真实生理伪影注入（EMG、EOG）
- 动态SNR加权
- 标准化数据导出
- 声学信号处理与真实节律提取
- 数据集冻结与基线脆弱性诊断

主要模块：
- noise_generators: 物理噪声生成器
- artifact_loader: 真实伪影数据加载器
- synthesizer: 信噪比合成与批量导出
- acoustic_processor: 声学信号处理（阶段一）
- real_rhythm_generator: 真实节律驱动生成（阶段二）
- dataset_freezer: 数据集冻结（阶段三）
- baseline_diagnosis: 基线脆弱性诊断（阶段四）

Author: PrintShop_EEG_Synthesizer Team
"""

from .noise_generators import (
    BaseNoiseGenerator,
    PLINoiseGenerator,
    BaselineWanderGenerator,
    MechanicalShockGenerator,
    CompositeNoiseGenerator,
    create_noise_generator,
)

from .artifact_loader import (
    ArtifactLoader,
    EDNDLoader,
    EARALoader,
    ArtifactType,
    ArtifactMetadata,
)

from .synthesizer import (
    EEGSynthesizer,
    SynthesizerConfig,
    SNRMode,
    create_synthesizer_from_config,
    load_existing_dataset,
    merge_with_clean_data,
    quick_synthesize,
    batch_synthesize_dataset,
)

# 阶段一：声学信号处理
from .acoustic_processor import (
    AcousticSignalProcessor,
    RealRhythmExtractor,
    load_audio_file,
    create_synthetic_factory_audio,
    audio_to_eeg_control_signal,
)

# 阶段二：真实节律驱动生成
from .real_rhythm_generator import (
    DynamicShockGenerator,
    DynamicBaselineWanderGenerator,
    RealRhythmNoiseGenerator,
    RealRhythmSynthesizer,
    create_real_rhythm_synthesizer,
    quick_synthesize_with_rhythm,
)

# 阶段三：数据集冻结
from .dataset_freezer import (
    DatasetManifest,
    FrozenNoiseDatasetGenerator,
    FrozenDatasetLoader,
    generate_seed_dataset,
    generate_deap_dataset,
    load_frozen_dataset,
)


__version__ = "1.0.0"
__author__ = "PrintShop_EEG_Synthesizer Team"

__all__ = [
    # 噪声生成器
    "BaseNoiseGenerator",
    "PLINoiseGenerator",
    "BaselineWanderGenerator",
    "MechanicalShockGenerator",
    "CompositeNoiseGenerator",
    "create_noise_generator",
    # 伪影加载器
    "ArtifactLoader",
    "EDNDLoader",
    "EARALoader",
    "ArtifactType",
    "ArtifactMetadata",
    # 合成器
    "EEGSynthesizer",
    "SynthesizerConfig",
    "SNRMode",
    "create_synthesizer_from_config",
    "load_existing_dataset",
    "merge_with_clean_data",
    "quick_synthesize",
    "batch_synthesize_dataset",
    # 阶段一：声学信号处理
    "AcousticSignalProcessor",
    "RealRhythmExtractor",
    "load_audio_file",
    "create_synthetic_factory_audio",
    "audio_to_eeg_control_signal",
    # 阶段二：真实节律驱动生成
    "DynamicShockGenerator",
    "DynamicBaselineWanderGenerator",
    "RealRhythmNoiseGenerator",
    "RealRhythmSynthesizer",
    "create_real_rhythm_synthesizer",
    "quick_synthesize_with_rhythm",
    # 阶段三：数据集冻结
    "DatasetManifest",
    "FrozenNoiseDatasetGenerator",
    "FrozenDatasetLoader",
    "generate_seed_dataset",
    "generate_deap_dataset",
    "load_frozen_dataset",
    # 阶段四：基线脆弱性诊断
]
