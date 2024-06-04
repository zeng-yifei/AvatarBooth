import os
from dataclasses import dataclass, field
from datetime import datetime

from omegaconf import OmegaConf


from utils.typing import *

# ============ Register OmegaConf Recolvers ============= #
OmegaConf.register_new_resolver(
    "calc_exp_lr_decay_rate", lambda factor, n: factor ** (1.0 / n)
)
OmegaConf.register_new_resolver("add", lambda a, b: a + b)
OmegaConf.register_new_resolver("sub", lambda a, b: a - b)
OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
OmegaConf.register_new_resolver("div", lambda a, b: a / b)
OmegaConf.register_new_resolver("idiv", lambda a, b: a // b)
OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(p))
OmegaConf.register_new_resolver("rmspace", lambda s, sub: s.replace(" ", sub))
# ======================================================= #


@dataclass
class ExperimentConfig:
    name: str = "default"
    description: str = ""
    tag: str = ""
    seed: int = 0
    use_timestamp: bool = True
    timestamp: Optional[str] = None
    exp_root_dir: str = "outputs"

    ### these shouldn't be set manually
    exp_dir: str = "outputs/default"
    trial_name: str = "exp"
    trial_dir: str = "outputs/default/exp"
    n_gpus: int = 1
    ###

    resume: Optional[str] = None

    data_type: str = ""
    data: dict = field(default_factory=dict)

    system_type: str = ""
    system: dict = field(default_factory=dict)

    # accept pytorch-lightning trainer parameters
    # see https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api
    trainer: dict = field(default_factory=dict)

    # accept pytorch-lightning checkpoint callback parameters
    # see https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#modelcheckpoint
    checkpoint: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.tag and not self.use_timestamp:
            raise ValueError("Either tag is specified or use_timestamp is True.")
        self.trial_name = self.tag
        if self.use_timestamp:

                if self.timestamp is None:
                    self.timestamp = datetime.now().strftime("@%Y%m%d-%H%M%S")
                self.trial_name += self.timestamp
        self.exp_dir = os.path.join(self.exp_root_dir, self.name)
        self.trial_dir = os.path.join(self.exp_dir, self.trial_name)


def load_config(*yaml_files: str, cli_args: list = [], **kwargs) -> Any:
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    cli_conf = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(*yaml_confs, cli_conf, kwargs)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    scfg = parse_structured(ExperimentConfig, cfg)
    return scfg


def config_to_primitive(config, resolve: bool = True) -> Any:
    return OmegaConf.to_container(config, resolve=resolve)


def dump_config(path: str, config) -> None:
    with open(path, "w") as fp:
        OmegaConf.save(config=config, f=fp)


def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg
