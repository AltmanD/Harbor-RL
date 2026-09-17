"""Thin boundary between the existing turn loop and canonical trajectories."""

import hashlib
import json
import os
from pathlib import Path
from harborrl.trajectories.producers.interactive import InteractiveProducer
from harborrl.trajectories.store import save


def context(plan, clients, sampling_params, expected_version, attempt_id):
    client = clients.sglang_client
    tokenizer = getattr(client, "tokenizer", None)
    template = getattr(tokenizer, "chat_template", None)
    tokenizer_identity = getattr(tokenizer, "_harbor_identity", None)
    if (
        tokenizer is not None
        and tokenizer_identity is None
        and hasattr(tokenizer, "get_vocab")
    ):
        tokenizer_identity = hashlib.sha256(
            json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()
        ).hexdigest()
        tokenizer._harbor_identity = tokenizer_identity
    template_config = {
        "template": template,
        "type": getattr(client, "chat_template_type", None),
        "kwargs": getattr(client, "chat_template_kwargs", None),
        "start": getattr(client, "messages_delimiter_start", None),
        "end": getattr(client, "messages_delimiter_end", None),
    }
    return dict(
        task=plan.task_meta.get("harbor_task_ref", {}),
        trajectory_id=plan.run_ctx.uid,
        attempt_id=attempt_id,
        group_id=str(plan.run_ctx.group_index),
        policy={
            "weight_version": expected_version,
            "tokenizer": tokenizer_identity,
            "tokenizer_name": getattr(tokenizer, "name_or_path", None),
            "chat_template": hashlib.sha256(
                json.dumps(template_config, sort_keys=True).encode()
            ).hexdigest()
            if template
            else None,
            "sampling_params": getattr(client, "sampling_params", sampling_params),
            "logprob_source": os.getenv("HARBOR_LOGPROB_SOURCE", ""),
            "logprob_semantics": os.getenv("HARBOR_LOGPROB_SEMANTICS", "unknown"),
        },
    )


async def serving_version(client):
    import aiohttp

    if os.getenv('HARBORRL_VERIFY_POLICY_POOL') == '1':
        import json
        from pathlib import Path
        from harborrl.platform.policy_pool import validate_pool
        pool = json.loads((Path(os.environ['RUN_DIR']) / 'policy-pool.json').read_text())
        states = []
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            for engine in pool['engines']:
                async with session.get(engine['endpoint'] + '/get_weight_version') as response:
                    response.raise_for_status()
                    states.append({**engine, 'weight_version': (await response.json())['weight_version']})
        return validate_pool(states, pool['weight_version'])

    url = (
        os.getenv("HARBOR_VERSION_ENDPOINT")
        or client.url.rsplit("/", 1)[0] + "/get_weight_version"
    )
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        async with session.get(url) as response:
            response.raise_for_status()
            value = (await response.json())["weight_version"]
    if value is None or str(value) == "":
        raise ValueError("serving did not provide a weight version")
    return str(value)


def build(
    plan,
    clients,
    loop,
    sampling_params,
    expected_version,
    attempt_id,
    status,
    eval_details,
    error,
):
    ctx = context(plan, clients, sampling_params, expected_version, attempt_id)
    ctx.update(
        execution_status=str(getattr(status, "value", status)),
        termination_reason=str(error or getattr(status, "value", status)),
    )
    evaluation = dict(eval_details or {})
    if error:
        evaluation.update(
            error=str(error),
            exception_stage=evaluation.get("exception_stage") or "rollout",
        )
    return InteractiveProducer.build(
        loop.interactions, loop.turn_records, evaluation, ctx
    )


def persist(ir, plan, *, started=False):
    root = Path(os.getenv("HARBOR_IR_ROOT", str(plan.run_ctx.log_dir / "ir")))
    return save(ir, root / "started" if started else root)
