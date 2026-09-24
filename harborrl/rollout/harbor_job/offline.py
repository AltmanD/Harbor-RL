"""Reproducible synthetic transport-to-export check; never produces RL evidence."""
import json
import threading
from dataclasses import replace
from pathlib import Path
from urllib import request

from harborrl.gateway.server import Gateway, make_server
from harborrl.gateway.trace import TraceRegistry
from harborrl.rollout.exporters.native import export_group
from harborrl.trajectories.native import (
    Identity,
    RewardProfile,
    assemble,
    digest,
    publish,
    require_ready,
)

from .audit import audit_session
from .collector import collect


class SyntheticBackend:
    def count_tokens(self, converted):
        return 2

    def generate(self, converted, identity, response_id):
        return {"input_ids": [1, 2], "output_ids": [3, 4], "logprobs": [-.2, -.3],
                "logprob_semantics": "raw_model", "policy_version": identity.policy_version,
                "tokenizer_digest": "synthetic-tokenizer", "template_digest": "synthetic-template",
                "engine_id": "synthetic", "serving_input": {"input_ids": [1, 2]},
                "sampling": converted["sampling"], "finish_reason": "end_turn",
                "evidence_kind": "synthetic", "content": [{"type": "text", "text": "fixture output"}]}


def smoke(output):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    profile = RewardProfile()
    sampling = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": 64, "stop": []}
    base = Identity("offline", "batch", "group", "0", "attempt-0", "trajectory-0", "synthetic-task",
                    "1", "synthetic-harness", profile.digest, digest(sampling))
    registry = TraceRegistry(root / "traces")
    gateway = Gateway(registry, SyntheticBackend(), model="policy")
    server = make_server(gateway)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    opener = request.build_opener(request.ProxyHandler({}))
    trajectories = []
    try:
        for slot, count in enumerate((1, 3)):
            identity = replace(base, slot_id=str(slot), attempt_id=f"attempt-{slot}", trajectory_id=f"trajectory-{slot}")
            trial = f"synthetic-trial-{slot}"
            token = registry.register(identity, trial)
            directory = root / identity.attempt_id
            directory.mkdir()
            session = directory / "session.jsonl"
            events = []
            for _ in range(count):
                payload = {"model": "policy", "max_tokens": 64, "messages": [{"role": "user", "content": "fixture"}]}
                req = request.Request(f"http://127.0.0.1:{server.server_port}/v1/messages",
                                      json.dumps(payload).encode(), {"x-api-key": token})
                with opener.open(req, timeout=5) as response:
                    message = json.load(response)
                # Wait for the server-side delivery record, not just receipt of HTTP bytes.
                with registry.changed:
                    if not registry.changed.wait_for(
                        lambda identity=identity: not registry.attempts[identity.attempt_id]["pending"], timeout=5
                    ):
                        raise TimeoutError("synthetic response delivery did not drain")
                events.append({"sessionId": trial, "type": "assistant", "message": message})
            session.write_text("".join(json.dumps(e) + "\n" for e in events))
            registry.close(identity.attempt_id)
            audit_session(session, registry, identity.attempt_id)
            turns, seal = registry.seal(identity.attempt_id)
            publish(directory / "result.json", {"id": trial, "finished_at": "synthetic", "verifier_environment_mode": "shared",
                                                "verifier_result": {"rewards": {"reward": slot}}})
            publish(directory / "reward.json", {"reward": slot})
            receipt = collect(identity, trial, directory / "result.json", directory, profile, evidence_kind="synthetic")
            publish(directory / "evaluation.json", receipt)
            ir = assemble(identity, trial, turns, seal, receipt)
            require_ready(ir, allow_synthetic=True)
            publish(directory / "trajectory.v2.json", ir)
            trajectories.append(ir)
        exported = export_group(trajectories, 2, allow_synthetic=True)
        publish(root / "export.json", exported)
        registry.drain(5)
        registry.publish_version([{"endpoint": "synthetic", "healthy": True, "weight_version": "2"}], "2")
        report = {"status": "CONTRACT_READY", "evidence_kind": "synthetic", "gpu_used": False,
                  "trajectories": 2, "turns": 4, "group_mean": exported["mean"],
                  "policy_barrier_version": registry.version,
                  "live_harbor_validated": False, "training_backend_integrated": False}
        publish(root / "acceptance.json", report)
        return report
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
