import glob
import os
import yaml
from typing import Any, Dict, List, Union

from indicators import DEFAULT_ENABLED_INDICATORS, INDICATOR_ORDER, build_feature_list


def resolve_data_paths(data_path: Union[str, List[str]]) -> List[str]:
    """Normalize `data.data_path` (a single path, a glob pattern, or a list of
    either) into a flat, sorted list of concrete file paths.

    A single-file config still works unmodified -- this always returns a
    list, but training code decides whether "one file" or "many files"
    means anything different.
    """
    if data_path is None:
        return []

    raw_entries = data_path if isinstance(data_path, list) else [data_path]

    resolved = []
    for entry in raw_entries:
        if any(ch in entry for ch in '*?['):
            matches = sorted(glob.glob(entry))
            if not matches:
                raise FileNotFoundError(f"Glob pattern matched no files: {entry}")
            resolved.extend(matches)
        else:
            resolved.append(entry)

    missing = [p for p in resolved if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"data_path entries not found: {missing}")

    # de-duplicate while preserving order (a glob and an explicit path could overlap)
    seen = set()
    deduped = []
    for p in resolved:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


class ConfigLoader:

    def __init__(self, config_path: str):

        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        config = self._resolve_dynamic_paths(config)

        return config

    def _resolve_dynamic_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:

        exp_name = config.get('model_paths', {}).get('exp_name', '')
        if not exp_name:
            return config

        base_path = config.get('model_paths', {}).get('base_path', '')
        path_templates = {
            'base_save_path': f"{base_path}/{exp_name}",
            'finetuned_tokenizer': f"{base_path}/{exp_name}/tokenizer/best_model"
        }

        if 'model_paths' in config:
            for key, template in path_templates.items():
                if key in config['model_paths']:
                    # only use template when the original value is empty string
                    current_value = config['model_paths'][key]
                    if current_value == "" or current_value is None:
                        config['model_paths'][key] = template
                    else:
                        # if the original value is not empty, use template to replace the {exp_name} placeholder
                        if isinstance(current_value, str) and '{exp_name}' in current_value:
                            config['model_paths'][key] = current_value.format(exp_name=exp_name)

        return config

    def get(self, key: str, default=None):

        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_data_config(self) -> Dict[str, Any]:
        return self.config.get('data', {})

    def get_training_config(self) -> Dict[str, Any]:
        return self.config.get('training', {})

    def get_model_paths(self) -> Dict[str, str]:
        return self.config.get('model_paths', {})

    def get_experiment_config(self) -> Dict[str, Any]:
        return self.config.get('experiment', {})

    def get_device_config(self) -> Dict[str, Any]:
        return self.config.get('device', {})

    def get_distributed_config(self) -> Dict[str, Any]:
        return self.config.get('distributed', {})

    def get_features_config(self) -> Dict[str, Any]:
        return self.config.get('features', {})

    def update_config(self, updates: Dict[str, Any]):

        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        self.config = update_nested_dict(self.config, updates)

    def save_config(self, save_path: str = None):

        if save_path is None:
            save_path = self.config_path

        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, indent=2)

    def print_config(self):
        print("=" * 50)
        print("Current configuration:")
        print("=" * 50)
        yaml.dump(self.config, default_flow_style=False, allow_unicode=True, indent=2)
        print("=" * 50)


class CustomFinetuneConfig:

    def __init__(self, config_path: str = None, data_path_override=None):

        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

        self.loader = ConfigLoader(config_path)
        self._data_path_override = data_path_override
        self._load_all_configs()

    def _load_all_configs(self):

        data_config = self.loader.get_data_config()
        # Optional override lets inference scripts pass --input without requiring
        # the config's (possibly remote/Colab) data_path to exist on disk.
        self.data_path = (
            self._data_path_override
            if self._data_path_override is not None
            else data_config.get('data_path')
        )
        # Resolve globs/lists into concrete file paths eagerly, at config load
        # time, so a bad/unmounted data_path (e.g. Google Drive not mounted
        # in Colab, or a typo'd directory) fails fast with a clear message
        # instead of surfacing as a confusing error deep in training.
        self.data_paths = resolve_data_paths(self.data_path)
        self.lookback_window = data_config.get('lookback_window', 512)
        self.predict_window = data_config.get('predict_window', 48)
        self.max_context = data_config.get('max_context', 512)
        self.clip = data_config.get('clip', 5.0)
        self.train_ratio = data_config.get('train_ratio', 0.9)
        self.val_ratio = data_config.get('val_ratio', 0.1)
        self.test_ratio = data_config.get('test_ratio', 0.0)

        # which technical indicators are enabled, and the resulting feature list / d_in
        features_config = self.loader.get_features_config()
        indicators_config = features_config.get('indicators', {})
        self.enabled_indicators = {
            name: indicators_config.get(name, DEFAULT_ENABLED_INDICATORS[name])
            for name in INDICATOR_ORDER
        }
        self.feature_list = build_feature_list(self.enabled_indicators)
        self.d_in = len(self.feature_list)

        # training configuration
        training_config = self.loader.get_training_config()
        # support training epochs of tokenizer and basemodel separately
        self.tokenizer_epochs = training_config.get('tokenizer_epochs', 30)
        self.basemodel_epochs = training_config.get('basemodel_epochs', 30)

        if 'epochs' in training_config and 'tokenizer_epochs' not in training_config:
            self.tokenizer_epochs = training_config.get('epochs', 30)
        if 'epochs' in training_config and 'basemodel_epochs' not in training_config:
            self.basemodel_epochs = training_config.get('epochs', 30)

        self.batch_size = training_config.get('batch_size', 160)
        self.log_interval = training_config.get('log_interval', 50)
        self.num_workers = training_config.get('num_workers', 6)
        self.seed = training_config.get('seed', 100)
        self.tokenizer_learning_rate = training_config.get('tokenizer_learning_rate', 2e-4)
        self.predictor_learning_rate = training_config.get('predictor_learning_rate', 4e-5)
        self.adam_beta1 = training_config.get('adam_beta1', 0.9)
        self.adam_beta2 = training_config.get('adam_beta2', 0.95)
        self.adam_weight_decay = training_config.get('adam_weight_decay', 0.1)
        self.accumulation_steps = training_config.get('accumulation_steps', 1)

        # Mixed precision (AMP) training: 'fp16' needs gradient scaling
        # (GradScaler) to avoid underflow, 'bf16' does not.
        self.use_amp = training_config.get('use_amp', False)
        self.amp_dtype = training_config.get('amp_dtype', 'fp16')

        # Resumable Checkpoint: `resume=true` auto-continues from
        # `checkpoint_last.pt` in the save dir if one exists (see
        # checkpoint_utils.py / docs/adr/0003). `checkpoint_every_n_steps`
        # additionally saves mid-epoch as a safety net for sessions that can
        # be cut off (e.g. Kaggle/Colab time limits).
        self.resume = training_config.get('resume', True)
        self.checkpoint_every_n_steps = training_config.get('checkpoint_every_n_steps', 0)

        model_paths = self.loader.get_model_paths()
        self.exp_name = model_paths.get('exp_name', 'default_experiment')
        self.pretrained_tokenizer_path = model_paths.get('pretrained_tokenizer')
        self.pretrained_predictor_path = model_paths.get('pretrained_predictor')
        self.base_save_path = model_paths.get('base_save_path')
        self.tokenizer_save_name = model_paths.get('tokenizer_save_name', 'tokenizer')
        self.basemodel_save_name = model_paths.get('basemodel_save_name', 'basemodel')
        self.finetuned_tokenizer_path = model_paths.get('finetuned_tokenizer')

        experiment_config = self.loader.get_experiment_config()
        self.experiment_name = experiment_config.get('name', 'kronos_custom_finetune')
        self.experiment_description = experiment_config.get('description', '')
        self.use_comet = experiment_config.get('use_comet', False)
        self.train_tokenizer = experiment_config.get('train_tokenizer', True)
        self.train_basemodel = experiment_config.get('train_basemodel', True)
        self.skip_existing = experiment_config.get('skip_existing', False)

        unified_pretrained = experiment_config.get('pre_trained', None)
        # Note: unlike finetune_csv, the tokenizer defaults to *not* pretrained here,
        # since enabling any indicator changes d_in away from the pretrained checkpoint's
        # d_in=6 (see docs/adr/0001-technical-indicator-tokenizer-from-scratch.md).
        self.pre_trained_tokenizer = experiment_config.get('pre_trained_tokenizer', unified_pretrained if unified_pretrained is not None else False)
        self.pre_trained_predictor = experiment_config.get('pre_trained_predictor', unified_pretrained if unified_pretrained is not None else True)

        device_config = self.loader.get_device_config()
        self.use_cuda = device_config.get('use_cuda', True)
        self.device_id = device_config.get('device_id', 0)

        distributed_config = self.loader.get_distributed_config()
        self.use_ddp = distributed_config.get('use_ddp', False)
        self.ddp_backend = distributed_config.get('backend', 'nccl')

        self._compute_full_paths()

    def _compute_full_paths(self):

        self.tokenizer_save_path = os.path.join(self.base_save_path, self.tokenizer_save_name)
        self.tokenizer_best_model_path = os.path.join(self.tokenizer_save_path, 'best_model')

        self.basemodel_save_path = os.path.join(self.base_save_path, self.basemodel_save_name)
        self.basemodel_best_model_path = os.path.join(self.basemodel_save_path, 'best_model')

    def get_tokenizer_config(self):

        return {
            'data_path': self.data_path,
            'data_paths': self.data_paths,
            'feature_list': self.feature_list,
            'enabled_indicators': self.enabled_indicators,
            'd_in': self.d_in,
            'lookback_window': self.lookback_window,
            'predict_window': self.predict_window,
            'max_context': self.max_context,
            'clip': self.clip,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'epochs': self.tokenizer_epochs,
            'batch_size': self.batch_size,
            'log_interval': self.log_interval,
            'num_workers': self.num_workers,
            'seed': self.seed,
            'learning_rate': self.tokenizer_learning_rate,
            'adam_beta1': self.adam_beta1,
            'adam_beta2': self.adam_beta2,
            'adam_weight_decay': self.adam_weight_decay,
            'accumulation_steps': self.accumulation_steps,
            'pretrained_model_path': self.pretrained_tokenizer_path,
            'save_path': self.tokenizer_save_path,
            'use_comet': self.use_comet
        }

    def get_basemodel_config(self):

        return {
            'data_path': self.data_path,
            'data_paths': self.data_paths,
            'feature_list': self.feature_list,
            'enabled_indicators': self.enabled_indicators,
            'd_in': self.d_in,
            'lookback_window': self.lookback_window,
            'predict_window': self.predict_window,
            'max_context': self.max_context,
            'clip': self.clip,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'epochs': self.basemodel_epochs,
            'batch_size': self.batch_size,
            'log_interval': self.log_interval,
            'num_workers': self.num_workers,
            'seed': self.seed,
            'predictor_learning_rate': self.predictor_learning_rate,
            'tokenizer_learning_rate': self.tokenizer_learning_rate,
            'adam_beta1': self.adam_beta1,
            'adam_beta2': self.adam_beta2,
            'adam_weight_decay': self.adam_weight_decay,
            'pretrained_tokenizer_path': self.finetuned_tokenizer_path,
            'pretrained_predictor_path': self.pretrained_predictor_path,
            'save_path': self.basemodel_save_path,
            'use_comet': self.use_comet
        }

    def print_config_summary(self):

        print("=" * 60)
        print("Kronos technical-indicator finetuning configuration summary")
        print("=" * 60)
        print(f"Experiment name: {self.exp_name}")
        print(f"Data path: {self.data_path}")
        print(f"Resolved data files ({len(self.data_paths)}): {self.data_paths}")
        print(f"Enabled indicators: {self.enabled_indicators}")
        print(f"Feature list ({self.d_in} dims): {self.feature_list}")
        print(f"Lookback window: {self.lookback_window}")
        print(f"Predict window: {self.predict_window}")
        print(f"Tokenizer training epochs: {self.tokenizer_epochs}")
        print(f"Basemodel training epochs: {self.basemodel_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Tokenizer learning rate: {self.tokenizer_learning_rate}")
        print(f"Predictor learning rate: {self.predictor_learning_rate}")
        print(f"Train tokenizer: {self.train_tokenizer}")
        print(f"Train basemodel: {self.train_basemodel}")
        print(f"Skip existing: {self.skip_existing}")
        print(f"Use pre-trained tokenizer: {self.pre_trained_tokenizer}")
        print(f"Use pre-trained predictor: {self.pre_trained_predictor}")
        print(f"Base save path: {self.base_save_path}")
        print(f"Tokenizer save path: {self.tokenizer_save_path}")
        print(f"Basemodel save path: {self.basemodel_save_path}")
        print("=" * 60)
