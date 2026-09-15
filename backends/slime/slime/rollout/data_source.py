import abc
import copy
import logging
import os
from pathlib import Path

import torch

from slime.utils.data import Dataset
from slime.utils.misc import load_function
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class DataSource(abc.ABC):
    @abc.abstractmethod
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

    @abc.abstractmethod
    def add_samples(self, samples: list[list[Sample]]):
        """
        Add samples to the data source
        """

    @abc.abstractmethod
    def save(self, rollout_id):
        """
        Save the state of the data source
        """

    @abc.abstractmethod
    def load(self, rollout_id=None):
        """
        Load the state of the data source
        """


# TODO may further refactor data-loading part later
class RolloutDataSource(DataSource):
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        # TODO remove this
        self.metadata = {}

        if args.rollout_global_dataset:
            tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
            processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

            # TODO move (during the refactor)
            if (d := args.dump_details) is not None:
                tokenizer.save_pretrained(Path(d) / "tokenizer")
                if processor:
                    processor.save_pretrained(Path(d) / "processor")

            self.dataset = Dataset(
                args.prompt_data,
                tokenizer=tokenizer,
                processor=processor,
                max_length=args.rollout_max_prompt_len,
                prompt_key=args.input_key,
                multimodal_keys=args.multimodal_keys,
                label_key=args.label_key,
                metadata_key=args.metadata_key,
                tool_key=args.tool_key,
                apply_chat_template=args.apply_chat_template,
                apply_chat_template_kwargs=args.apply_chat_template_kwargs,
                seed=args.rollout_seed,
            )
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
        else:
            self.dataset = None

    def get_samples(self, num_samples):
        # TODO further improve code
        if self.dataset is not None:
            if self.sample_offset + num_samples <= len(self.dataset):
                prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
                self.sample_offset += num_samples
            else:
                prompt_samples = self.dataset.samples[self.sample_offset :]
                num_samples -= len(prompt_samples)
                self.epoch_id += 1
                if self.args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
                prompt_samples += self.dataset.samples[:num_samples]
                self.sample_offset = num_samples
        else:
            prompt_samples = [Sample() for _ in range(num_samples)]

        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
        }
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset:
            return

        if self.args.load is None:
            return

        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info(f"Checkpoint {path} does not exist.")
            return

        logger.info(f"load metadata from {path}")
        logger.info(f"load metadata: {self.metadata}")
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})

        if self.args.rollout_global_dataset and self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)


class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = []
        self.buffer_max_size = getattr(args, "buffer_max_size", 1000)
        self.buffer_enabled = getattr(args, "use_buffer", None)
        if self.buffer_enabled is None:
            self.buffer_enabled = getattr(args, "loss_type", "policy_loss") == "decoupled_policy_loss"
        if getattr(args, "buffer_mode", "in_process") == "none":
            self.buffer_enabled = False
        self.current_policy_version = 0
        self.total_added = 0
        self.total_sampled = 0
        if self.buffer_enabled:
            from slime.utils.buffer_sampling_strategies import get_sampling_strategy

            self.sampling_strategy = get_sampling_strategy(args, self.current_policy_version)
            logger.info(
                "Replay buffer enabled: max_size=%s strategy=%s",
                self.buffer_max_size,
                self.sampling_strategy.get_name(),
            )
        else:
            self.sampling_strategy = None
        self.trajectory_replay_enabled = bool(getattr(args, "enable_trajectory_replay", False))
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)
        self.sil_buffer = None
        if self.trajectory_replay_enabled:
            try:
                from slime.utils.sil_buffer import SILBuffer

                self.sil_buffer = SILBuffer(
                    buffer_size=getattr(args, "trajectory_buffer_size", 2048),
                    score_threshold=getattr(args, "trajectory_score_threshold", 1.0),
                    posadv_only=getattr(args, "enable_trajectory_posadv", False),
                    weight_decay=getattr(args, "weight_decay_trajectory_replay", -1.0),
                    baseline_buffer_size=getattr(args, "baseline_buffer_size", 10240),
                    tolerate_steps=getattr(args, "trajectory_tolerate_steps", 10),
                    seed=getattr(args, "seed", 42),
                )
                logger.info("SPEAR SIL buffer enabled: size=%s", getattr(args, "trajectory_buffer_size", 2048))
            except Exception as exc:
                logger.warning("SPEAR SIL buffer init failed; disabling SIL: %s", exc)
                self.sil_buffer = None

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """
        if self.buffer_enabled:
            return super().get_samples(num_samples=num_samples)

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        if self.sampling_strategy is not None:
            if self.sampling_strategy.current_policy_version != self.current_policy_version:
                self.sampling_strategy.current_policy_version = self.current_policy_version
            samples = self.sampling_strategy.sample(self.buffer, num_samples)
        else:
            samples = self.buffer_filter(self.args, self.current_policy_version, self.buffer, num_samples)
        self.total_sampled += len(samples)
        return samples

    def get_training_samples(self, num_samples: int) -> list[list[Sample]]:
        if not self.buffer_enabled:
            return self.get_samples(num_samples)
        samples = self._get_samples_from_buffer(num_samples)
        if len(samples) < num_samples:
            logger.info("Replay buffer returned %d/%d requested training groups", len(samples), num_samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        """
        Add a sample group to buffer.
        """
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        if not isinstance(samples[0], list):
            samples = _group_flat_samples_for_replay(samples, self.args.n_samples_per_prompt)
        if self.sil_buffer is not None:
            self._push_sil_candidates(samples)
        if not self.buffer_enabled and not self.args.partial_rollout:
            return
        for i in range(0, len(samples)):
            assert (
                len(samples[i]) == self.args.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
            group = samples[i]  # type: ignore
            group_is_complete = all(
                getattr(sample, "response", None) not in (None, "")
                for sample in group
            )
            if self.buffer_enabled and not group_is_complete:
                continue
            if self.buffer_enabled:
                for sample in group:
                    if sample.loss_mask is None and sample.response_length > 0:
                        sample.loss_mask = [1] * sample.response_length
            if self.buffer_enabled and not all(
                sample.rollout_log_probs is not None
                and len(sample.rollout_log_probs) == sample.response_length
                and sample.loss_mask is not None
                and len(sample.loss_mask) == sample.response_length
                for sample in group
            ):
                continue
            for sample in group:
                policy_version = getattr(sample, "policy_version", None)
                if policy_version is None or not isinstance(policy_version, int) or policy_version < 0:
                    sample.policy_version = self.current_policy_version
            if getattr(self.args, "enable_dynamic_sampling", False):
                try:
                    from slime.rollout.dynamic_sampling import select_admissible_groups

                    admitted, _, _ = select_admissible_groups([group], self.args)
                    if not admitted:
                        continue
                except Exception as exc:
                    logger.warning("Dynamic sampling gate failed; admitting group: %s", exc)
            self.buffer.append(group)
            self.total_added += 1
        while self.buffer_enabled and len(self.buffer) > self.buffer_max_size:
            self.buffer.pop(0)

    def _push_sil_candidates(self, samples: list[list[Sample]]):
        if self.sil_buffer is None:
            return
        try:
            entries = []
            group_rewards: dict[int, list[float]] = {}
            for sample in _iter_sample_leaves(samples):
                if sample.response_length == 0 or sample.tokens is None:
                    continue
                if sample.rollout_log_probs is None or len(sample.rollout_log_probs) != sample.response_length:
                    continue
                loss_mask = sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length
                if len(loss_mask) != sample.response_length:
                    continue
                reward_value = float(sample.get_reward_value(self.args)) if sample.reward is not None else 0.0
                group_index = getattr(sample, "group_index", None)
                if group_index is not None:
                    group_rewards.setdefault(int(group_index), []).append(reward_value)
                policy_version = getattr(sample, "policy_version", None)
                try:
                    policy_version = int(policy_version)
                except (TypeError, ValueError):
                    policy_version = self.current_policy_version
                if policy_version < 0:
                    policy_version = self.current_policy_version
                entries.append(
                    {
                        "tokens": sample.tokens,
                        "group_index": int(group_index) if group_index is not None else None,
                        "response_length": sample.response_length,
                        "loss_mask": loss_mask,
                        "rollout_log_probs": sample.rollout_log_probs,
                        "policy_version": policy_version,
                        "reward": reward_value,
                        # The official implementation admits positive
                        # group-relative advantages.  Keep this value on the
                        # record so ``enable_trajectory_posadv`` has the same
                        # meaning before the training-side GRPO pass runs.
                        "advantage": reward_value,
                    }
                )
            if entries:
                mean_by_group = {
                    key: sum(values) / len(values) for key, values in group_rewards.items() if values
                }
                for entry in entries:
                    group_index = entry.get("group_index")
                    if group_index in mean_by_group:
                        entry["advantage"] = entry["reward"] - mean_by_group[group_index]
                try:
                    # Policy version advances once per rollout by the driver;
                    # use it as the SPEAR age clock instead of the number of
                    # groups admitted (which depends on batch size).
                    sil_step = int(getattr(self, "current_policy_version", self.total_added))
                    self.sil_buffer.push(
                        entries,
                        current_step=sil_step,
                        group_rewards=mean_by_group.values(),
                    )
                except TypeError as exc:
                    # Keep compatibility with custom/legacy SIL buffers that
                    # only implement the original two-argument ``push`` API.
                    if "group_rewards" not in str(exc):
                        raise
                    self.sil_buffer.push(entries, current_step=sil_step)
        except Exception as exc:
            logger.warning("SPEAR SIL candidate push failed: %s", exc)

    def update_policy_version(self, version: int):
        self.current_policy_version = version
        if self.sampling_strategy is not None:
            self.sampling_strategy.current_policy_version = version

    # TODO remove
    def update_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    # TODO remove
    def get_metadata(self):
        return self.metadata

    def get_buffer_length(self):
        return len(self.buffer)

    def save(self, rollout_id):
        """Persist dataset cursors and replay state together for resumable runs."""
        super().save(rollout_id)
        if not getattr(self.args, "save", None):
            return
        path = Path(self.args.save) / "rollout" / f"replay_buffer_state_{rollout_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": "lightrl_rollout_replay_v1",
            "buffer": list(self.buffer),
            "buffer_max_size": self.buffer_max_size,
            "current_policy_version": self.current_policy_version,
            "total_added": self.total_added,
            "total_sampled": self.total_sampled,
            "sil_buffer": self.sil_buffer.state_dict() if self.sil_buffer is not None else None,
        }
        torch.save(state, path)

    def load(self, rollout_id=None):
        """Restore replay state when a checkpoint contains one."""
        super().load(rollout_id)
        if getattr(self.args, "load", None) is None:
            return
        if rollout_id is None:
            return
        path = Path(self.args.load) / "rollout" / f"replay_buffer_state_{rollout_id}.pt"
        if not path.is_file():
            logger.info("Replay buffer checkpoint does not exist: %s", path)
            return
        state = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise TypeError(f"Expected replay state dict, got {type(state).__name__}")
        self.buffer = list(state.get("buffer") or [])[-self.buffer_max_size :]
        self.current_policy_version = int(state.get("current_policy_version", 0))
        self.total_added = int(state.get("total_added", len(self.buffer)))
        self.total_sampled = int(state.get("total_sampled", 0))
        sil_state = state.get("sil_buffer")
        if self.sil_buffer is not None and isinstance(sil_state, dict):
            self.sil_buffer.load_state_dict(sil_state)


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples


def _group_flat_samples_for_replay(samples: list[Sample], group_size: int) -> list[list[Sample]]:
    groups = []
    for start in range(0, len(samples), group_size):
        group = samples[start : start + group_size]
        if len(group) == group_size:
            groups.append(group)
    return groups


def _iter_sample_leaves(samples):
    """Yield Sample leaves from possibly nested rollout sample containers."""
    if isinstance(samples, Sample):
        yield samples
        return
    if isinstance(samples, (list, tuple)):
        for item in samples:
            yield from _iter_sample_leaves(item)
