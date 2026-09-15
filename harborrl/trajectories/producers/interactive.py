from copy import deepcopy
from ..schema import Trajectory
from ..validation import validate


class InteractiveProducer:
    @staticmethod
    def build(interactions, tool_records, evaluation, context):
        fields = (
            "turn_idx",
            "input_ids",
            "output_token_ids",
            "output_token_logprobs",
            "output_text",
            "finish_reason",
            "messages",
            "latency_ms",
            "generation_meta",
        )
        ir = Trajectory(**deepcopy(context))
        ir.turns = [
            {key: deepcopy(getattr(turn, key)) for key in fields}
            for turn in interactions
        ]
        ir.events = deepcopy(tool_records)
        ir.evaluation = deepcopy(evaluation)
        ir.integrity_errors = validate(ir)
        ir.readiness = "EVAL_ONLY" if ir.integrity_errors else "RL_READY"
        return ir
