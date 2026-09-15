from dataclasses import asdict, dataclass
import math
from harborrl.trajectories.validation import require_rl_ready
from harborrl.types import Interaction


@dataclass(frozen=True)
class ExportConfig:
    version: str = "slime-outcome-v1"
    discount: float = 1.0
    outcome_is_score: bool = False
    penalize_short_response: bool = True
    dapo_overlong_cfg: dict | None = None


class SlimeExporter:
    @staticmethod
    def export(ir, base_sample, config, *, weight_version, group_id):
        require_rl_ready(ir, weight_version=weight_version, group_id=group_id)
        if (
            config.version != "slime-outcome-v1"
            or not math.isfinite(config.discount)
            or config.discount < 0
        ):
            raise ValueError("unsupported or invalid export configuration")
        if str(base_sample.group_index) != ir.group_id:
            raise ValueError("base Sample group does not match trajectory")
        from slime.utils.types import Sample
        from harborrl.rollout.sample_builder import _build_samples

        status = (
            Sample.Status.COMPLETED
            if ir.execution_status == "completed"
            else Sample.Status.TRUNCATED
        )
        samples = _build_samples(
            [Interaction(**turn) for turn in ir.turns],
            base_sample,
            ir.evaluation["raw_reward"],
            status,
            discount=config.discount,
            outcome_is_score=config.outcome_is_score,
            penalize_short_response=config.penalize_short_response,
            dapo_overlong_cfg=config.dapo_overlong_cfg,
        )
        for sample in samples:
            if not math.isfinite(sample.reward["score"]):
                raise ValueError("export produced a nonfinite reward")
            sample.weight_versions = [str(ir.policy["weight_version"])]
            sample.metadata.update(
                trajectory_id=ir.trajectory_id,
                group_id=ir.group_id,
                ir_schema_version=ir.schema_version,
                export_config=asdict(config),
            )
        return samples
