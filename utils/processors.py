import json
import os
from dataclasses import dataclass

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import logging
from utils.base import BaseObject
from utils.typing import *


def hash_prompt(model: str, prompt: str) -> str:
    import hashlib

    identifier = f"{model}-{prompt}"
    return hashlib.md5(identifier.encode()).hexdigest()


class PromptProcessor(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        prompt: str = "a hamburger"
        negative_prompt: str = ""
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        view_dependent_prompting: bool = True
        overhead_threshold: float = 60.0
        front_threshold: float = 45.0
        back_threshold: float = 45.0
        view_dependent_prompt_front: bool = False
        use_cache: bool = True

    cfg: Config

    @rank_zero_only
    def configure_text_encoder(self) -> None:
        raise NotImplementedError

    @rank_zero_only
    def destroy_text_encoder(self) -> None:
        raise NotImplementedError

    def configure(self, trainer) -> None:
        self.trainer = trainer

        self._cache_dir = ".threestudio_cache/text_embeddings"  # FIXME: hard-coded path

        @dataclass
        class DirectionConfig:
            name: str
            prompt: Callable[[str], str]
            negative_prompt: Callable[[str], str]
            condition: Callable[
                [Float[Tensor, "B"], Float[Tensor, "B"], Float[Tensor, "B"]],
                Float[Tensor, "B"],
            ]

        # view-dependent text embeddings
        self.directions: List[DirectionConfig]
        if self.cfg.view_dependent_prompt_front:
            self.directions = [
                DirectionConfig(
                    "side",
                    lambda s: f"side view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: torch.ones_like(ele, dtype=torch.bool),
                ),
                DirectionConfig(
                    "front",
                    lambda s: f"front view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: (azi > -self.cfg.front_threshold)
                    & (azi < self.cfg.front_threshold),
                ),
                DirectionConfig(
                    "back",
                    lambda s: f"backside view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: (azi > 180 - self.cfg.back_threshold)
                    | (azi < -180 + self.cfg.back_threshold),
                ),
                DirectionConfig(
                    "overhead",
                    lambda s: f"overhead view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: ele > self.cfg.overhead_threshold,
                ),
            ]
        else:
            self.directions = [
                DirectionConfig(
                    "side",
                    lambda s: f"{s}, side view",
                    lambda s: s,
                    lambda ele, azi, dis: torch.ones_like(ele, dtype=torch.bool),
                ),
                DirectionConfig(
                    "front",
                    lambda s: f"{s}, front view",
                    lambda s: s,
                    lambda ele, azi, dis: (azi > -self.cfg.front_threshold)
                    & (azi < self.cfg.front_threshold),
                ),
                DirectionConfig(
                    "back",
                    lambda s: f"{s}, back view",
                    lambda s: s,
                    lambda ele, azi, dis: (azi > 180 - self.cfg.back_threshold)
                    | (azi < -180 + self.cfg.back_threshold),
                ),
                DirectionConfig(
                    "overhead",
                    lambda s: f"{s}, overhead view",
                    lambda s: s,
                    lambda ele, azi, dis: ele > self.cfg.overhead_threshold,
                ),
            ]

        self.direction2idx = {d.name: i for i, d in enumerate(self.directions)}

        with open(os.path.join("load/prompt_library.json"), "r") as f:
            self.prompt_library = json.load(f)
        # use provided prompt or find prompt in library
        self.prompt = self.preprocess_prompt(self.cfg.prompt)
        # use provided negative prompt
        self.negative_prompt = self.cfg.negative_prompt
        self.prompts_vd = [d.prompt(self.prompt) for d in self.directions]
        self.negative_prompts_vd = [
            d.negative_prompt(self.negative_prompt) for d in self.directions
        ]

        self.prepare_text_embeddings()
        self.load_text_embeddings()

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir):
        raise NotImplementedError

    @rank_zero_only
    def prepare_text_embeddings(self):
        os.makedirs(self._cache_dir, exist_ok=True)

        all_prompts = (
            [self.prompt]
            + [self.negative_prompt]
            + self.prompts_vd
            + self.negative_prompts_vd
        )
        prompts_to_process = []
        for prompt in all_prompts:
            if self.cfg.use_cache:
                # some text embeddings are already in cache
                # do not process them
                cache_path = os.path.join(
                    self._cache_dir,
                    f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt",
                )
                
                if os.path.exists(cache_path):
                    logging.debug(
                        f"Text embeddings for model {self.cfg.pretrained_model_name_or_path} and prompt [{prompt}] are already in cache, skip processing."
                    )
                    continue
            prompts_to_process.append(prompt)

        if len(prompts_to_process) > 0:
            ctx = mp.get_context("spawn")
            subprocess = ctx.Process(
                target=self.spawn_func,
                args=(
                    self.cfg.pretrained_model_name_or_path,
                    prompts_to_process,
                    self._cache_dir,
                ),
            )
            subprocess.start()
            subprocess.join()

    def load_text_embeddings(self):
        # synchronize, to ensure the text embeddings have been computed and saved to cache
        self.trainer.strategy.barrier()
        self.text_embeddings = self.load_from_cache(self.prompt)[None, ...]
        self.uncond_text_embeddings = self.load_from_cache(self.negative_prompt)[
            None, ...
        ]
        self.text_embeddings_vd = torch.stack(
            [self.load_from_cache(prompt) for prompt in self.prompts_vd], dim=0
        )
        self.uncond_text_embeddings_vd = torch.stack(
            [self.load_from_cache(prompt) for prompt in self.negative_prompts_vd], dim=0
        )
        logging.debug(f"Loaded text embeddings.")

    def load_from_cache(self, prompt):
        cache_path = os.path.join(
            self._cache_dir,
            f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt",
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Text embedding file {cache_path} for model {self.cfg.pretrained_model_name_or_path} and prompt [{prompt}] not found."
            )
        return torch.load(cache_path, map_location=self.device)

    def preprocess_prompt(self, prompt: str) -> str:
        if prompt.startswith("lib:"):
            # find matches in the library
            candidate = None
            keywords = prompt[4:].lower().split("_")
            for prompt in self.prompt_library["dreamfusion"]:
                if all([k in prompt.lower() for k in keywords]):
                    if candidate is not None:
                        raise ValueError(
                            f"Multiple prompts matched with keywords {keywords} in library"
                        )
                    candidate = prompt
            if candidate is None:
                raise ValueError(
                    f"Cannot find prompt with keywords {keywords} in library"
                )
            return candidate
        else:
            return prompt

    def get_text_embeddings(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Tuple[Float[Tensor, "B ..."], Float[Tensor, "B ..."]]:
        raise NotImplementedError

    def __call__(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        **kwargs,
    ) -> Float[Tensor, "BB ..."]:
        batch_size = elevation.shape[0]

        if self.cfg.view_dependent_prompting:
            # Get direction
            direction_idx = torch.zeros_like(elevation, dtype=torch.long)
            for d in self.directions:
                direction_idx[
                    d.condition(elevation, azimuth, camera_distances)
                ] = self.direction2idx[d.name]

            # Get text embeddings
            text_embeddings = self.text_embeddings_vd[direction_idx]  # type: ignore
            uncond_text_embeddings = self.uncond_text_embeddings_vd[direction_idx]  # type: ignore
        else:
            text_embeddings = self.text_embeddings.expand(batch_size, -1, -1)  # type: ignore
            uncond_text_embeddings = self.uncond_text_embeddings.expand(  # type: ignore
                batch_size, -1, -1
            )

        # IMPORTANT: we return (cond, uncond), which is in different order than other implementations!
        return torch.cat([text_embeddings, uncond_text_embeddings], dim=0)
