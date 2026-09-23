# ── Start router ─────────────────────────────────────────────────────
ROUTER_PID=""
cleanup() {
  set +e
  if [[ "${HARBORRL_STRUCTURED_LAUNCH:-0}" == "1" && "${RAY_STATUS_STATE:-}" != "unknown" ]]; then
    timeout 120 "${TRAIN_PYTHON}" -m harborrl.platform.run_leases || true
  fi
  if [[ -n "${ROUTER_PID}" ]] && kill -0 "${ROUTER_PID}" 2>/dev/null; then
    kill "${ROUTER_PID}" || true
  fi
}
cancel_run() {
  trap '' INT TERM
  RAY_STATUS_STATE="cancelled"
  if [[ -n "${RAY_LOG_PID:-}" ]]; then
    kill "${RAY_LOG_PID}" 2>/dev/null || true
  fi
  if [[ -n "${RAY_JOB_SUBMISSION_ID:-}" ]]; then
    # Stop only the job submitted by this launcher; shared services stay up.
    timeout 45 ray job stop --address="http://${MASTER_ADDR}:8265" "${RAY_JOB_SUBMISSION_ID}" || true
  fi
  exit "$1"
}
trap cleanup EXIT
trap 'cancel_run 130' INT
trap 'cancel_run 143' TERM

ROUTER_LOG="${RUN_LOG_DIR}/router.log"
require_cmd curl
if [[ "${NEEDS_ENV_ROUTER}" == "1" && "${START_ENV_POOL_SERVER}" == "1" ]]; then
  if [[ "${AUTO_CLOSE_STALE_WORKER_RUNS}" == "1" ]]; then
    log "Pre-cleaning stale worker runs before router readiness check..."
    close_stale_runs_for_all_workers "pre_router_start" || true
  fi
  log "Starting router on ${ROUTER_HOST}:${ROUTER_PORT} -> ${WORKER_URLS} (python=${ROUTER_PYTHON})"
  log "  worker_urls_file=${WORKER_URLS_FILE} reload_interval=${WORKER_URLS_RELOAD_INTERVAL}s"
  log "  forward_timeout=${ROUTER_FORWARD_TIMEOUT}s retries=${ROUTER_FORWARD_RETRIES} backoff=${ROUTER_FORWARD_RETRY_BACKOFF}s pressure_cooldown=${ROUTER_PRESSURE_COOLDOWN}s no_proxy=${NO_PROXY}"
  log "  readiness require_router=${ROUTER_REQUIRE_READY} wait_forever=${ROUTER_READY_WAIT_FOREVER} require_worker=${WORKER_PREFLIGHT_REQUIRE_READY} probe_timeout=${READY_PROBE_TIMEOUT}s worker_timeout=${ROUTER_READYZ_WORKER_TIMEOUT}s auto_close_stale=${AUTO_CLOSE_STALE_WORKER_RUNS}"
  (
    cd "${REPO_ROOT}"
    "${ROUTER_PYTHON}" -m harborrl.platform.router_cli \
      --host "${ROUTER_HOST}" --port "${ROUTER_PORT}" --workers "${WORKER_URLS}" \
      --workers-file "${WORKER_URLS_FILE}" --workers-reload-interval "${WORKER_URLS_RELOAD_INTERVAL}" \
      > "${ROUTER_LOG}" 2>&1 &
    echo $! > "${RUN_LOG_DIR}/router.pid"
  )
  ROUTER_PID="$(cat "${RUN_LOG_DIR}/router.pid")"
  log "Router PID=${ROUTER_PID}, log=${ROUTER_LOG}"

  # Wait for router readiness. /readyz validates at least one env worker; /healthz
  # is only used as fallback for older router implementations.
  ROUTER_READY=0
  i=1
  while true; do
    if probe_ready_endpoint "http://${CHECK_HOST}:${ROUTER_PORT}" "router http://${CHECK_HOST}:${ROUTER_PORT}" "${READY_PROBE_TIMEOUT}"; then
      log "router ready (attempt ${i})"
      ROUTER_READY=1
      break
    fi
    if [[ "${AUTO_CLOSE_STALE_WORKER_RUNS}" == "1" && $((i % STALE_WORKER_CLOSE_INTERVAL)) -eq 0 ]]; then
      close_stale_runs_for_all_workers "router_wait_attempt_${i}" || true
    fi
    if [[ -n "${ROUTER_PID}" ]] && ! kill -0 "${ROUTER_PID}" 2>/dev/null; then
      log "ERROR: router process exited before becoming ready; see ${ROUTER_LOG}"
      break
    fi
    if [[ "${ROUTER_READY_WAIT_FOREVER}" != "1" && "${i}" -ge "${CHECK_WAIT_SECS}" ]]; then
      break
    fi
    sleep 1
    i=$((i + 1))
  done
  if [[ "${ROUTER_READY}" != "1" ]]; then
    log "ERROR: router not ready after ${CHECK_WAIT_SECS}s"
    if [[ "${ROUTER_REQUIRE_READY}" == "1" ]]; then
      exit 1
    fi
  fi
  curl -fsS "http://${CHECK_HOST}:${ROUTER_PORT}/status" || true
  echo
else
  if [[ "${NEEDS_ENV_ROUTER}" == "1" ]]; then
    log "Skipping local terminal env router; using ENV_SERVER_URL=${ENV_SERVER_URL} WORKER_URLS=${WORKER_URLS}"
  else
    log "Skipping terminal env router; Agent-SafetyBench uses local env backend"
  fi
fi

# Pre-flight: sanity check each pool worker before launching training
# (issue #3 §1.X-E: early detection of worker transport flakes).
if [[ "${NEEDS_ENV_ROUTER}" == "1" ]]; then
  # Router startup may wait indefinitely while its workers file is hot-reloaded.
  # Keep the direct preflight aligned with the worker set that made the router
  # ready instead of probing a stale snapshot. In direct-worker mode an
  # explicit WORKER_URLS always wins over a possibly stale local file.
  if [[ "${START_ENV_POOL_SERVER}" == "1" || "${WORKER_URLS_FROM_FILE}" == "1" ]]; then
    _REFRESHED_WORKER_URLS="$(read_worker_urls_from_file "${WORKER_URLS_FILE}")"
    if [[ -n "${_REFRESHED_WORKER_URLS}" && "${_REFRESHED_WORKER_URLS}" != "${WORKER_URLS}" ]]; then
      log "Worker URLs changed during router startup; refreshing preflight targets: ${WORKER_URLS} -> ${_REFRESHED_WORKER_URLS}"
      WORKER_URLS="${_REFRESHED_WORKER_URLS}"
      export WORKER_URLS
    fi
  else
    log "Keeping explicit WORKER_URLS for direct-worker preflight: ${WORKER_URLS}"
  fi
  log "Probing worker endpoints..."
  IFS=',' read -r -a _WORKERS <<< "${WORKER_URLS}"
  READY_WORKERS=0
  for _w in "${_WORKERS[@]}"; do
    if probe_ready_endpoint "${_w}" "${_w}" "${WORKER_PREFLIGHT_TIMEOUT}"; then
      READY_WORKERS=$((READY_WORKERS + 1))
    else
      _probe_rc=$?
      if [[ "${_probe_rc}" == "1" && "${AUTO_CLOSE_STALE_WORKER_RUNS}" == "1" ]]; then
        close_stale_worker_runs "${_w}" "${_w} (worker_preflight)" "${STALE_WORKER_CLOSE_TIMEOUT}" || true
      elif [[ "${_probe_rc}" == "2" ]]; then
        log "  [WARN] ${_w}: skipping stale-run repair because the worker is unreachable"
      fi
    fi
  done
  log "Worker readiness: ${READY_WORKERS}/${#_WORKERS[@]} ready"
  if [[ "${READY_WORKERS}" -le 0 && "${WORKER_PREFLIGHT_REQUIRE_READY}" == "1" ]]; then
    log "ERROR: no ready docker env worker; aborting before Ray job submit"
    exit 1
  fi
fi

# ── NVLink detection ─────────────────────────────────────────────────
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [[ "${NVLINK_COUNT:-0}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-${HAS_NVLINK}}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"

NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NVLS_ENABLE NCCL_P2P_DISABLE NCCL_IB_DISABLE
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-}"
log "HAS_NVLINK=${HAS_NVLINK} NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} NCCL_IB_DISABLE=${NCCL_IB_DISABLE} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-auto} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-auto}"


# ── Dump run config ──────────────────────────────────────────────────
# Preserve the original provenance on checkpoint resume and record each
# recovery launch separately.
RUN_CONFIG_OUTPUT="${RUN_DIR}/config/run_config.json"
if [[ "${RESUME_EXISTING_RUN:-0}" == "1" ]]; then
  RUN_CONFIG_OUTPUT="${RUN_DIR}/config/run_config.resume-${RUN_TIMESTAMP}.json"
fi
cat > "${RUN_CONFIG_OUTPUT}" <<CFGEOF
{
  "run_name": "${RUN_NAME}",
  "run_category": "${RUN_CATEGORY}",
  "timestamp": "${RUN_TIMESTAMP}",
  "run_dataset_tag": "${RUN_DATASET_TAG}",
  "run_algo_tag": "${RUN_ALGO_TAG}",
  "debug_mode": ${DEBUG_MODE},
  "dry_run": "${DRY_RUN}",
  "algo": "${ALGO}",
  "harness_option": "${HARNESS_OPTION}",
  "model": "${MODEL_DISPLAY_NAME}",
  "hf_ckpt": "${HF_CKPT}",
  "ref_load": "${REF_LOAD}",
  "save_ckpt": "${SAVE_CKPT}",
  "save_interval": ${SAVE_INTERVAL},
  "save_first_rollout": "${SAVE_FIRST_ROLLOUT}",
  "num_gpus": ${NUM_GPUS},
  "actor_gpus": ${ACTOR_GPUS},
  "rollout_gpus": ${ROLLOUT_GPUS},
  "tp_size": ${TP_SIZE},
  "rollout_engine_gpus": ${ROLLOUT_NUM_GPUS_PER_ENGINE},
  "seta_execution_profile": "${SETA_EXECUTION_PROFILE:-legacy}",
  "sglang_server_concurrency": "${SGLANG_SERVER_CONCURRENCY:-}",
  "slime_use_fault_tolerance": "${SLIME_USE_FAULT_TOLERANCE:-0}",
  "slime_save_debug_rollout_data": "${SLIME_SAVE_DEBUG_ROLLOUT_DATA:-}",
  "dataset": "${DATASET}",
  "includes_seta": "${INCLUDES_SETA}",
  "includes_safety": "${INCLUDES_SAFETY}",
  "includes_agentharm": "${INCLUDES_AGENTHARM}",
  "includes_swesmith": "${INCLUDES_SWESMITH}",
  "prompt_data": "${ROLLOUT_PROMPT_DATA}",
  "eval_protocol": "${EVAL_PROTOCOL:-legacy}",
  "eval_config": "${EVAL_CONFIG:-}",
  "eval_seed": "${EVAL_SEED:-}",
  "eval_steps": "${EVAL_STEPS:-}",
  "eval_manifest_sha256": "${EVAL_MANIFEST_SHA256:-}",
  "eval_set_sha256": "${EVAL_SET_SHA256:-}",
  "train_set_sha256": "${TRAIN_SET_SHA256:-}",
  "swesmith_source_prompt_data": "${SWESMITH_SOURCE_PROMPT_DATA}",
  "swesmith_artifact_sha256": "${SWESMITH_ARTIFACT_SHA256}",
  "swesmith_conversion_mode": "${SWESMITH_CONVERSION_MODE}",
  "swesmith_dataset_revision": "${SWESMITH_DATASET_REVISION}",
  "swesmith_converter_sha256": "${SWESMITH_CONVERTER_SHA256}",
  "mix_mode": "${MIX_MODE}",
  "num_rollout": ${NUM_ROLLOUT},
  "rollout_batch_size": ${ROLLOUT_BATCH_SIZE},
  "n_samples": ${N_SAMPLES},
  "max_turn": ${MAX_TURN},
  "rollout_max_response_len": ${ROLLOUT_MAX_RESPONSE_LEN},
  "rollout_max_context_len": ${ROLLOUT_MAX_CONTEXT_LEN},
  "rollout_generation_max_retries": "${ROLLOUT_GENERATION_MAX_RETRIES}",
  "rollout_generation_retry_initial_backoff": "${ROLLOUT_GENERATION_RETRY_INITIAL_BACKOFF}",
  "rollout_generation_retry_max_backoff": "${ROLLOUT_GENERATION_RETRY_MAX_BACKOFF}",
  "rollout_generation_skip_on_failure": "${ROLLOUT_GENERATION_SKIP_ON_FAILURE}",
  "rollout_generation_retry_backoff_multiplier": "${ROLLOUT_GENERATION_RETRY_BACKOFF_MULTIPLIER}",
  "rollout_generation_env_storm_max_retries": "${ROLLOUT_GENERATION_ENV_STORM_MAX_RETRIES}",
  "max_tokens_per_gpu": ${MAX_TOKENS_PER_GPU},
  "worker_urls": "${WORKER_URLS}",
  "worker_urls_file": "${WORKER_URLS_FILE}",
  "worker_urls_reload_interval": "${WORKER_URLS_RELOAD_INTERVAL}",
  "env_server_url": "${ENV_SERVER_URL}",
  "needs_env_router": "${NEEDS_ENV_ROUTER}",
  "router_pressure_cooldown": "${ROUTER_PRESSURE_COOLDOWN}",
  "router_require_ready": "${ROUTER_REQUIRE_READY}",
  "router_readyz_worker_timeout": "${ROUTER_READYZ_WORKER_TIMEOUT}",
  "worker_preflight_require_ready": "${WORKER_PREFLIGHT_REQUIRE_READY}",
  "router_ready_wait_forever": "${ROUTER_READY_WAIT_FOREVER}",
  "auto_close_stale_worker_runs": "${AUTO_CLOSE_STALE_WORKER_RUNS}",
  "stale_worker_close_interval": "${STALE_WORKER_CLOSE_INTERVAL}",
  "stale_worker_close_timeout": "${STALE_WORKER_CLOSE_TIMEOUT}",
  "stale_worker_repair_min_age": "${STALE_WORKER_REPAIR_MIN_AGE}",
  "stale_worker_repair_max_repairs": "${STALE_WORKER_REPAIR_MAX_REPAIRS}",
  "env_http_max_retries": "${ENV_HTTP_MAX_RETRIES}",
  "env_allocate_max_retries": "${ENV_ALLOCATE_MAX_RETRIES}",
  "http_retry_log_every_n": "${HTTP_RETRY_LOG_EVERY_N}",
  "http_retry_log_response_chars": "${HTTP_RETRY_LOG_RESPONSE_CHARS}",
  "terminal_rl_generate_failure_traceback": "${TERMINAL_RL_GENERATE_FAILURE_TRACEBACK}",
  "env_reset_http_timeout": "${ENV_RESET_HTTP_TIMEOUT}",
  "env_reset_max_retries": "${ENV_RESET_MAX_RETRIES}",
  "env_reset_lease_max_attempts": "${ENV_RESET_LEASE_MAX_ATTEMPTS}",
  "env_reset_lease_retry_base_sleep": "${ENV_RESET_LEASE_RETRY_BASE_SLEEP}",
  "env_reset_lease_retry_max_sleep": "${ENV_RESET_LEASE_RETRY_MAX_SLEEP}",
  "env_close_http_timeout": "${ENV_CLOSE_HTTP_TIMEOUT}",
  "env_remote_max_active_tasks": "${ENV_REMOTE_MAX_ACTIVE_TASKS}",
  "env_remote_max_active_runs": "${ENV_REMOTE_MAX_ACTIVE_RUNS}",
  "env_remote_max_runs_per_task": "${ENV_REMOTE_MAX_RUNS_PER_TASK}",
  "eval_rollout_max_concurrency": "${EVAL_ROLLOUT_MAX_CONCURRENCY}",
  "agent_safetybench_remote_env": "${AGENT_SAFETYBENCH_REMOTE_ENV}",
  "agentharm_remote_env": "${AGENTHARM_REMOTE_ENV}",
  "seta_safety": "${SETA_SAFETY}",
  "safety_bench_reward": "${SAFETY_BENCH_REWARD}",
  "agentharm_reward": "${AGENTHARM_REWARD}",
  "agentharm_root": "${AGENTHARM_ROOT}",
  "dapo_eps_clip_low": "${DAPO_EPS_CLIP_LOW}",
  "dapo_eps_clip_high": "${DAPO_EPS_CLIP_HIGH}",
  "dapo_calculate_per_token_loss": "${DAPO_CALCULATE_PER_TOKEN_LOSS}",
  "dapo_dynamic_sampling": "${DAPO_DYNAMIC_SAMPLING}",
  "dapo_dynamic_filter_path": "${DAPO_DYNAMIC_FILTER_PATH}",
  "dapo_over_sampling_batch_size": "${DAPO_OVER_SAMPLING_BATCH_SIZE}",
  "dapo_failed_group_abort_min_groups": "${DAPO_FAILED_GROUP_ABORT_MIN_GROUPS}",
  "dapo_failed_group_abort_ratio": "${DAPO_FAILED_GROUP_ABORT_RATIO}",
  "dapo_grpo_std_normalization": "${DAPO_GRPO_STD_NORMALIZATION}",
  "dapo_use_kl_loss": "${DAPO_USE_KL_LOSS}",
  "dapo_kl_loss_coef": "${DAPO_KL_LOSS_COEF}",
  "dapo_overlong_buffer_enable": "${DAPO_OVERLONG_BUFFER_ENABLE}",
  "dapo_overlong_buffer_len": "${DAPO_OVERLONG_BUFFER_LEN}",
  "dapo_overlong_penalty_factor": "${DAPO_OVERLONG_PENALTY_FACTOR}",
  "trajectory_save_interval_env": "${TRAJECTORY_SAVE_INTERVAL}",
  "trajectory_save_interval_seta": "${TRAJECTORY_SAVE_INTERVAL_SETA}",
  "trajectory_save_interval_agent_safetybench": "${TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH}",
  "trajectory_save_interval_agentharm": "${TRAJECTORY_SAVE_INTERVAL_AGENTHARM}",
  "trajectory_save_policy": "${TRAJECTORY_SAVE_POLICY}",
  "trajectory_task_save_interval": "${TRAJECTORY_TASK_SAVE_INTERVAL}",
  "trajectory_task_max_per_step": "${TRAJECTORY_TASK_MAX_PER_STEP}",
  "trajectory_task_max_per_task": "${TRAJECTORY_TASK_MAX_PER_TASK}",
  "trajectory_max_total": "${TRAJECTORY_MAX_TOTAL}",
  "trajectory_save_reward_strata": "${TRAJECTORY_SAVE_REWARD_STRATA}",
  "trajectory_save_log_decisions": "${TRAJECTORY_SAVE_LOG_DECISIONS}",
  "exploration_profile": "${EXPLORATION_PROFILE}",
  "custom_reward_post_process_path": "${CUSTOM_REWARD_POST_PROCESS_PATH:-}",
  "explore_entropy_coef": "${EXPLORE_ENTROPY_COEF}",
  "explore_think_mode": "${EXPLORE_THINK_MODE}",
  "explore_temp_high": "${EXPLORE_TEMP_HIGH}",
  "explore_intrinsic": "${EXPLORE_INTRINSIC}",
  "explore_intrinsic_enabled": "${EXPLORE_INTRINSIC_ENABLED}",
  "explore_intrinsic_coef": "${EXPLORE_INTRINSIC_COEF}",
  "explore_intrinsic_schedule": "${EXPLORE_INTRINSIC_SCHEDULE}",
  "explore_intrinsic_decay_steps": "${EXPLORE_INTRINSIC_DECAY_STEPS}",
  "explore_intrinsic_reducer": "${EXPLORE_INTRINSIC_REDUCER}",
  "explore_intrinsic_granularity": "${EXPLORE_INTRINSIC_GRANULARITY}",
  "explore_intrinsic_scope": "${EXPLORE_INTRINSIC_SCOPE}",
  "explore_score_bonus_components": "${EXPLORE_SCORE_BONUS_COMPONENTS}",
  "explore_safety_filter": "${EXPLORE_SAFETY_FILTER}",
  "explore_safety_filter_enabled": "${EXPLORE_SAFETY_FILTER_ENABLED}",
  "explore_safety_filter_coef": "${EXPLORE_SAFETY_FILTER_COEF}",
  "explore_lprnd": "${EXPLORE_LPRND}",
  "explore_lprnd_enabled": "${EXPLORE_LPRND_ENABLED}",
  "explore_lprnd_coef": "${EXPLORE_LPRND_COEF}",
  "explore_lprnd_schedule": "${EXPLORE_LPRND_SCHEDULE}",
  "explore_lprnd_decay_steps": "${EXPLORE_LPRND_DECAY_STEPS}",
  "explore_lprnd_clip": "${EXPLORE_LPRND_CLIP}",
  "explore_lprnd_warmup": "${EXPLORE_LPRND_WARMUP}",
  "explore_agent57_lite": "${EXPLORE_AGENT57_LITE}",
  "explore_agent57_lite_enabled": "${EXPLORE_AGENT57_LITE_ENABLED}",
  "explore_agent57_k": "${EXPLORE_AGENT57_K}",
  "explore_agent57_arm_betas": "${EXPLORE_AGENT57_ARM_BETAS}",
  "explore_agent57_combine_mode": "${EXPLORE_AGENT57_COMBINE_MODE}",
  "explore_agent57_ngu_mod_clip": "${EXPLORE_AGENT57_NGU_MOD_CLIP}",
  "explore_agent57_ngu_episodic_source": "${EXPLORE_AGENT57_NGU_EPISODIC_SOURCE}",
  "explore_agent57_ngu_episodic_reducer": "${EXPLORE_AGENT57_NGU_EPISODIC_REDUCER}",
  "explore_agent57_ngu_life_mod_mode": "${EXPLORE_AGENT57_NGU_LIFE_MOD_MODE}",
  "explore_agent57_ngu_life_mod_std_clip": "${EXPLORE_AGENT57_NGU_LIFE_MOD_STD_CLIP}",
  "explore_agent57_max_bonus": "${EXPLORE_AGENT57_MAX_BONUS}",
  "explore_agent57_arm_temperatures": "${EXPLORE_AGENT57_ARM_TEMPERATURES}",
  "explore_agent57_arm_temperature_warmup_rollouts": "${EXPLORE_AGENT57_ARM_TEMPERATURE_WARMUP_ROLLOUTS}",
  "explore_agent57_arm_top_ps": "${EXPLORE_AGENT57_ARM_TOP_PS}",
  "explore_agent57_arm_top_ks": "${EXPLORE_AGENT57_ARM_TOP_KS}",
  "explore_agent57_controller": "${EXPLORE_AGENT57_CONTROLLER}",
  "explore_agent57_ucb_c": "${EXPLORE_AGENT57_UCB_C}",
  "explore_agent57_ucb_window": "${EXPLORE_AGENT57_UCB_WINDOW}",
  "explore_agent57_ucb_epsilon": "${EXPLORE_AGENT57_UCB_EPSILON}",
  "explore_agent57_ucb_min_per_arm": "${EXPLORE_AGENT57_UCB_MIN_PER_ARM}",
  "explore_agent57_ucb_value": "${EXPLORE_AGENT57_UCB_VALUE}",
  "explore_agent57_ucb_dataset_aware": "${EXPLORE_AGENT57_UCB_DATASET_AWARE}",
  "explore_agent57_ucb_random_seed": "${EXPLORE_AGENT57_UCB_RANDOM_SEED}",
  "explore_agent57_ucb_seed_salt": "${EXPLORE_AGENT57_UCB_SEED_SALT}",
  "explore_agent57_keep_baseline": "${EXPLORE_AGENT57_KEEP_BASELINE}",
  "episodic_memory_backend": "${EPISODIC_MEMORY_BACKEND}",
  "explore_agent57_episodic_backend": "${EXPLORE_AGENT57_EPISODIC_BACKEND}",
  "explore_agent57_episodic_capacity": "${EXPLORE_AGENT57_EPISODIC_CAPACITY}",
  "explore_agent57_episodic_count_decay": "${EXPLORE_AGENT57_EPISODIC_COUNT_DECAY}",
  "explore_agent57_episodic_clear_on_reset": "${EXPLORE_AGENT57_EPISODIC_CLEAR_ON_RESET}",
  "explore_agent57_episodic_simhash_bits": "${EXPLORE_AGENT57_EPISODIC_SIMHASH_BITS}",
  "explore_agent57_episodic_bucket_capacity": "${EXPLORE_AGENT57_EPISODIC_BUCKET_CAPACITY}",
  "explore_agent57_episodic_k": "${EXPLORE_AGENT57_EPISODIC_K}",
  "explore_agent57_episodic_distance": "${EXPLORE_AGENT57_EPISODIC_DISTANCE}",
  "explore_agent57_episodic_vector_dim": "${EXPLORE_AGENT57_EPISODIC_VECTOR_DIM}",
  "explore_agent57_episodic_random_seed": "${EXPLORE_AGENT57_EPISODIC_RANDOM_SEED}",
  "explore_agent57_episodic_obs_mode": "${EXPLORE_AGENT57_EPISODIC_OBS_MODE}",
  "explore_agent57_episodic_include_turn": "${EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN}",
  "explore_agent57_episodic_turn_mode": "${EXPLORE_AGENT57_EPISODIC_TURN_MODE}",
  "explore_agent57_episodic_multi_probe_radius": "${EXPLORE_AGENT57_EPISODIC_MULTI_PROBE_RADIUS}",
  "explore_agent57_episodic_novelty_floor": "${EXPLORE_AGENT57_EPISODIC_NOVELTY_FLOOR}",
  "explore_agent57_lifelong": "${EXPLORE_AGENT57_LIFELONG}",
  "explore_agent57_lifelong_enabled": "${EXPLORE_AGENT57_LIFELONG_ENABLED}",
  "explore_agent57_lifelong_coef": "${EXPLORE_AGENT57_LIFELONG_COEF}",
  "explore_agent57_lifelong_clip": "${EXPLORE_AGENT57_LIFELONG_CLIP}",
  "explore_agent57_lifelong_warmup": "${EXPLORE_AGENT57_LIFELONG_WARMUP}",
  "explore_agent57_lifelong_count_decay": "${EXPLORE_AGENT57_LIFELONG_COUNT_DECAY}",
  "explore_agent57_lifelong_capacity": "${EXPLORE_AGENT57_LIFELONG_CAPACITY}",
  "explore_agent57_lifelong_backend": "${EXPLORE_AGENT57_LIFELONG_BACKEND}",
  "explore_agent57_lifelong_key_version": "${EXPLORE_AGENT57_LIFELONG_KEY_VERSION}",
  "explore_agent57_lifelong_include_dataset": "${EXPLORE_AGENT57_LIFELONG_INCLUDE_DATASET}",
  "explore_agent57_lifelong_include_task": "${EXPLORE_AGENT57_LIFELONG_INCLUDE_TASK}",
  "explore_agent57_lifelong_include_turn": "${EXPLORE_AGENT57_LIFELONG_INCLUDE_TURN}",
  "explore_agent57_lifelong_obs_mode": "${EXPLORE_AGENT57_LIFELONG_OBS_MODE}",
  "explore_agent57_lifelong_hierarchical": "${EXPLORE_AGENT57_LIFELONG_HIERARCHICAL}",
  "explore_agent57_lifelong_task_weight": "${EXPLORE_AGENT57_LIFELONG_TASK_WEIGHT}",
  "explore_agent57_lifelong_skill_weight": "${EXPLORE_AGENT57_LIFELONG_SKILL_WEIGHT}",
  "explore_agent57_lifelong_global_weight": "${EXPLORE_AGENT57_LIFELONG_GLOBAL_WEIGHT}",
  "explore_agent57_sqlite_busy_timeout_ms": "${EXPLORE_AGENT57_SQLITE_BUSY_TIMEOUT_MS}",
  "explore_agent57_sqlite_wal": "${EXPLORE_AGENT57_SQLITE_WAL}",
  "explore_agent57_trust_gate": "${EXPLORE_AGENT57_TRUST_GATE}",
  "explore_agent57_trust_completed": "${EXPLORE_AGENT57_TRUST_COMPLETED}",
  "explore_agent57_trust_truncated": "${EXPLORE_AGENT57_TRUST_TRUNCATED}",
  "explore_agent57_trust_failed": "${EXPLORE_AGENT57_TRUST_FAILED}",
  "explore_agent57_trust_parse_error": "${EXPLORE_AGENT57_TRUST_PARSE_ERROR}",
  "explore_agent57_trust_warmup": "${EXPLORE_AGENT57_TRUST_WARMUP}",
  "explore_agent57_state_path": "${EXPLORE_AGENT57_STATE_PATH}",
  "explore_agent57_success_threshold": "${EXPLORE_AGENT57_SUCCESS_THRESHOLD}",
  "explore_advantage_bonus": "${EXPLORE_ADVANTAGE_BONUS}",
  "explore_advantage_bonus_enabled": "${EXPLORE_ADVANTAGE_BONUS_ENABLED}",
  "explore_advantage_bonus_mode": "${EXPLORE_ADVANTAGE_BONUS_MODE}",
  "explore_advantage_bonus_components": "${EXPLORE_ADVANTAGE_BONUS_COMPONENTS}",
  "explore_advantage_bonus_coef": "${EXPLORE_ADVANTAGE_BONUS_COEF}",
  "explore_advantage_bonus_clip": "${EXPLORE_ADVANTAGE_BONUS_CLIP}",
  "explore_advantage_intrinsic_key": "${EXPLORE_ADVANTAGE_INTRINSIC_KEY}",
  "explore_advantage_lambda": "${EXPLORE_ADVANTAGE_LAMBDA}",
  "explore_advantage_lambda_schedule": "${EXPLORE_ADVANTAGE_LAMBDA_SCHEDULE}",
  "explore_advantage_lambda_decay_steps": "${EXPLORE_ADVANTAGE_LAMBDA_DECAY_STEPS}",
  "explore_advantage_arm_weight_mode": "${EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE}",
  "explore_advantage_trust_key": "${EXPLORE_ADVANTAGE_TRUST_KEY}",
  "explore_advantage_gate_mode": "${EXPLORE_ADVANTAGE_GATE_MODE}",
  "explore_advantage_outcome_key": "${EXPLORE_ADVANTAGE_OUTCOME_KEY}",
  "explore_advantage_completed_floor": "${EXPLORE_ADVANTAGE_COMPLETED_FLOOR}",
  "explore_advantage_truncated_floor": "${EXPLORE_ADVANTAGE_TRUNCATED_FLOOR}",
  "explore_advantage_failed_floor": "${EXPLORE_ADVANTAGE_FAILED_FLOOR}",
  "explore_advantage_aborted_floor": "${EXPLORE_ADVANTAGE_ABORTED_FLOOR}",
  "explore_advantage_truncated_intrinsic_scale": "${EXPLORE_ADVANTAGE_TRUNCATED_INTRINSIC_SCALE}",
  "explore_advantage_failed_intrinsic_scale": "${EXPLORE_ADVANTAGE_FAILED_INTRINSIC_SCALE}",
  "explore_truncation_penalty": "${EXPLORE_TRUNCATION_PENALTY}",
  "explore_advantage_truncation_penalty": "${EXPLORE_ADVANTAGE_TRUNCATION_PENALTY}",
  "explore_truncation_penalty_outcome_aware": "${EXPLORE_TRUNCATION_PENALTY_OUTCOME_AWARE}",
  "slime_skip_zero_trainable_rollout": "${SLIME_SKIP_ZERO_TRAINABLE_ROLLOUT}",
  "slime_skip_zero_trainable_train": "${SLIME_SKIP_ZERO_TRAINABLE_TRAIN}",
  "explore_cde_actor": "${EXPLORE_CDE_ACTOR}",
  "explore_cde_actor_enabled": "${EXPLORE_CDE_ACTOR_ENABLED}",
  "explore_cde_actor_omega": "${EXPLORE_CDE_ACTOR_OMEGA}",
  "explore_cde_actor_kappa": "${EXPLORE_CDE_ACTOR_KAPPA}",
  "explore_cde_actor_alpha": "${EXPLORE_CDE_ACTOR_ALPHA}",
  "explore_cde_actor_reward_gate": "${EXPLORE_CDE_ACTOR_REWARD_GATE}",
  "explore_cde_actor_decay_steps": "${EXPLORE_CDE_ACTOR_DECAY_STEPS}",
  "explore_retry_attempts": "${EXPLORE_RETRY_ATTEMPTS}",
  "explore_retry_traj_gamma": "${EXPLORE_RETRY_TRAJ_GAMMA}",
  "terminal_structured_metrics": "${TERMINAL_STRUCTURED_METRICS}",
  "terminal_metrics_jsonl": "${TERMINAL_METRICS_JSONL}",
  "terminal_wandb_metric_profile": "${TERMINAL_WANDB_METRIC_PROFILE}",
  "train_python": "${TRAIN_PYTHON}",
  "claude_code_cli": "${CLAUDE_CODE_CLI}",
  "claude_code_llm_backend": "${CLAUDE_CODE_LLM_BACKEND}",
  "claude_code_model": "${CLAUDE_CODE_MODEL}",
  "claude_code_qwen_gateway_model": "${CLAUDE_CODE_QWEN_GATEWAY_MODEL}",
  "claude_code_local_run_root": "${CLAUDE_CODE_LOCAL_RUN_ROOT}",
  "claude_code_workspace_root_compat": "${CLAUDE_CODE_WORKSPACE_ROOT}",
  "claude_code_max_tool_rounds": "${CLAUDE_CODE_MAX_TOOL_ROUNDS}",
  "claude_code_tool_timeout_ms": "${CLAUDE_CODE_TOOL_TIMEOUT_MS}",
  "claude_code_turn_timeout_sec": "${CLAUDE_CODE_TURN_TIMEOUT_SEC}",
  "claude_code_output_format": "${CLAUDE_CODE_OUTPUT_FORMAT}",
  "claude_code_permission_mode": "${CLAUDE_CODE_PERMISSION_MODE}",
  "claude_code_allowed_tools": "${CLAUDE_CODE_ALLOWED_TOOLS}",
  "claude_code_mark_non_trainable": "${CLAUDE_CODE_MARK_NON_TRAINABLE}",
  "slime_ray_placement_gpu_probe": "${SLIME_RAY_PLACEMENT_GPU_PROBE}",
  "log_dir": "${RUN_LOG_DIR}"
}
CFGEOF

if [[ "${FORMAL_CAPTURE_SOURCE_STATE:-0}" == "1" ]]; then
  log "Capturing formal-run source state..."
  "${TRAIN_PYTHON}" "${REPO_ROOT}/tools/infra/capture_formal_run_source.py" \
    --run-dir "${RUN_DIR}"
fi

# ── Start Ray head ───────────────────────────────────────────────────
_ray_num_cpus="${RAY_NUM_CPUS:-$(nproc)}"
# Cap the plasma object store: the default (30% of shm/RAM) consumes
# ~100-300 GiB of the pod's memory cgroup, which colocated GLM-5.1
# workloads need for the engine/training CPU weight backups.
log "ray start --head ..."
ray start --head \
  --node-ip-address "${NODE_IP}" \
  --num-cpus "${_ray_num_cpus}" \
  --num-gpus "${NUM_GPUS}" \
  --object-store-memory "${RAY_OBJECT_STORE_MEMORY:-8000000000}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --temp-dir "${RAY_TMPDIR}"
unset _ray_num_cpus

log "Waiting for Ray dashboard http://${MASTER_ADDR}:8265 ..."
for i in {1..40}; do
  if curl -fsS --max-time 3 "http://${MASTER_ADDR}:8265/api/version" >/dev/null 2>&1; then
    log "Ray dashboard ready (attempt $i)"
    break
  fi
  sleep 3
done

# ── Build runtime env ────────────────────────────────────────────────
# Only add import roots. Adding ${SCRIPT_DIR} (the harborrl package directory)
# makes harborrl/platform shadow Python's stdlib `platform` module and causes
# every runtime-env Ray worker to crash before registration.
RUNTIME_PYTHONPATH="${MEGATRON_DIR}:${REPO_ROOT}:${SLIME_DIR}"
# A harness overlay may expose selected pure-Python packages that the prepared
# RJob image lacks.  It must never expose a whole conda site-packages tree,
# which could replace the image's binary torch/vLLM stack.
if [[ -n "${LIGHTRL_RUNTIME_HARNESS_OVERLAY:-}" ]]; then
  [[ -d "${LIGHTRL_RUNTIME_HARNESS_OVERLAY}" ]] || {
    log "ERROR: missing runtime harness overlay: ${LIGHTRL_RUNTIME_HARNESS_OVERLAY}"
    exit 1
  }
  RUNTIME_PYTHONPATH="${RUNTIME_PYTHONPATH}:${LIGHTRL_RUNTIME_HARNESS_OVERLAY}"
fi
if ! PYTHONPATH="${RUNTIME_PYTHONPATH}" "${TRAIN_PYTHON}" - <<'PY'
import platform

if not callable(getattr(platform, "system", None)):
    raise RuntimeError(f"stdlib platform module was shadowed by {platform.__file__}")
PY
then
  log "ERROR: Ray runtime PYTHONPATH shadows a Python standard-library module: ${RUNTIME_PYTHONPATH}"
  exit 1
fi

RUNTIME_ENV_XTRACE_WAS_ON=0
if [[ "${HARNESS_OPTION}" == "claude-code" && "$-" == *x* ]]; then
  RUNTIME_ENV_XTRACE_WAS_ON=1
  set +x
fi


CLAUDE_RUNTIME_ENV_JSON=""
if [[ "${HARNESS_OPTION}" == "claude-code" ]]; then
  CLAUDE_RUNTIME_ENV_JSON=",
    \"CLAUDE_CODE_CLI\": \"${CLAUDE_CODE_CLI}\",
    \"CLAUDE_CODE_LLM_BACKEND\": \"${CLAUDE_CODE_LLM_BACKEND}\",
    \"CLAUDE_CODE_MODEL\": \"${CLAUDE_CODE_MODEL}\",
    \"CLAUDE_CODE_QWEN_GATEWAY_MODEL\": \"${CLAUDE_CODE_QWEN_GATEWAY_MODEL}\",
    \"CLAUDE_CODE_LOCAL_RUN_ROOT\": \"${CLAUDE_CODE_LOCAL_RUN_ROOT}\",
    \"CLAUDE_CODE_WORKSPACE_ROOT\": \"${CLAUDE_CODE_WORKSPACE_ROOT}\",
    \"CLAUDE_CODE_TURN_TIMEOUT_SEC\": \"${CLAUDE_CODE_TURN_TIMEOUT_SEC}\",
    \"CLAUDE_CODE_TOOL_TIMEOUT_MS\": \"${CLAUDE_CODE_TOOL_TIMEOUT_MS}\",
    \"CLAUDE_CODE_MAX_TOOL_ROUNDS\": \"${CLAUDE_CODE_MAX_TOOL_ROUNDS}\",
    \"CLAUDE_CODE_OUTPUT_FORMAT\": \"${CLAUDE_CODE_OUTPUT_FORMAT}\",
    \"CLAUDE_CODE_PERMISSION_MODE\": \"${CLAUDE_CODE_PERMISSION_MODE}\",
    \"CLAUDE_CODE_ALLOWED_TOOLS\": \"${CLAUDE_CODE_ALLOWED_TOOLS}\",
    \"CLAUDE_CODE_DISALLOWED_TOOLS\": \"${CLAUDE_CODE_DISALLOWED_TOOLS}\",
    \"CLAUDE_CODE_EXTRA_ARGS\": \"${CLAUDE_CODE_EXTRA_ARGS}\",
    \"CLAUDE_CODE_SYSTEM_PROMPT\": \"${CLAUDE_CODE_SYSTEM_PROMPT}\",
    \"CLAUDE_CODE_MCP_PYTHON\": \"${CLAUDE_CODE_MCP_PYTHON}\",
    \"CLAUDE_CODE_HTTP_MAX_RETRIES\": \"${CLAUDE_CODE_HTTP_MAX_RETRIES}\",
    \"CLAUDE_CODE_HTTP_RETRY_DELAY\": \"${CLAUDE_CODE_HTTP_RETRY_DELAY}\",
    \"CLAUDE_CODE_MARK_NON_TRAINABLE\": \"${CLAUDE_CODE_MARK_NON_TRAINABLE}\",
    \"ANTHROPIC_API_KEY\": \"${ANTHROPIC_API_KEY}\",
    \"ANTHROPIC_AUTH_TOKEN\": \"${ANTHROPIC_AUTH_TOKEN}\",
    \"ANTHROPIC_BASE_URL\": \"${ANTHROPIC_BASE_URL}\",
    \"ANTHROPIC_API_URL\": \"${ANTHROPIC_API_URL}\""
fi

# Behavior flags exported by the launcher (SGLang switches and NCCL
# flight-recorder settings from e.g. the GLM-5.1 smoke script) must reach
# engine and training actors on EVERY node.  Ray merges the job-level
# runtime env into all actors, while an actor's own raylet env does not
# propagate — without this passthrough, workers run with stock SGLang
# defaults and deadlock against the head in mismatched collectives.
EXTRA_ENV_PASSTHROUGH_JSON=""
for _passthrough_var in \
  HARBORRL_VERIFY_POLICY_POOL \
  HARBORRL_GPU_LAYOUT \
  HARBORRL_SKIP_GLOBAL_CLEANUP \
  HARBORRL_NATIVE_ROLLOUT \
  HARBORRL_NATIVE_SLIME_CONTRACT \
  HARBORRL_NATIVE_CATALOG \
  HARBORRL_NATIVE_PROFILE \
  HARBORRL_NATIVE_RUNNER_PYTHON \
  HARBORRL_NATIVE_RUNNER_WORKERS \
  HARBORRL_NATIVE_GATEWAY_HOST \
  HARBORRL_NATIVE_GATEWAY_URL \
  HARBORRL_NATIVE_GATEWAY_PORT \
  HARBORRL_NATIVE_TOKENIZER_DIGEST \
  HARBORRL_NATIVE_TEMPLATE_DIGEST \
  HARBORRL_NATIVE_AUDITED_LOGPROBS \
  HARBORRL_NATIVE_MAX_ATTEMPTS \
  HARBORRL_NATIVE_DRAIN_TIMEOUT \
  LD_LIBRARY_PATH \
  HARBOR_IR_ROOT \
  HARBOR_LOGPROB_SOURCE \
  HARBOR_LOGPROB_SEMANTICS \
  HARBOR_VERSION_ENDPOINT \
  SGLANG_RETURN_ORIGINAL_LOGPROB \
  SGLANG_USE_MESSAGE_QUEUE_BROADCASTER \
  SGLANG_DISABLE_NSA_DP_ATTENTION \
  SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS \
  SGLANG_GLM51_SAFE_MLA_CONCAT \
  SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT \
  SGLANG_DISABLE_CUDA_GRAPH \
  SGLANG_ENABLE_WEIGHTS_CPU_BACKUP \
  SGLANG_ATTENTION_BACKEND \
  SGLANG_CHUNKED_PREFILL_SIZE \
  SGLANG_MEM_FRACTION_STATIC \
  LIGHTRL_SGLANG_SERVER_PYTHON \
  MEGATRON_BRIDGE_LOAD_LOG_EVERY \
  TORCH_NCCL_TRACE_BUFFER_SIZE \
  TORCH_NCCL_DUMP_ON_TIMEOUT \
  TORCH_NCCL_DEBUG_INFO_TEMP_FILE; do
  append_env_passthrough "${_passthrough_var}"
done
unset _passthrough_var

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PATH\": \"${PATH}\",
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH:-}\",
    \"PYTHONPATH\": \"${RUNTIME_PYTHONPATH}\",
    \"REPO_ROOT\": \"${REPO_ROOT}\",
    \"DATASET_DIR\": \"${DATASET_DIR:-}\",
    \"LIGHTRL_SETA_SUPPORT_SITE_PATHS\": \"${LIGHTRL_SETA_SUPPORT_SITE_PATHS:-}\",
    \"LIGHTRL_SETA_FALLBACK_SITE_PATHS\": \"${LIGHTRL_SETA_FALLBACK_SITE_PATHS:-}\",
    \"LIGHTRL_SETA_FALLBACK_MODULES\": \"${LIGHTRL_SETA_FALLBACK_MODULES:-}\",
    \"LIGHTRL_MEGATRON_BRIDGE_OVERLAY\": \"${LIGHTRL_MEGATRON_BRIDGE_OVERLAY:-}\",
    \"PYTHONUNBUFFERED\": \"1\",
    \"PYTHONFAULTHANDLER\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${NCCL_NVLS_ENABLE}\",
    \"NCCL_P2P_DISABLE\": \"${NCCL_P2P_DISABLE}\",

    \"NCCL_IB_DISABLE\": \"${NCCL_IB_DISABLE}\",
    \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\",
    \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME}\",
    \"SLIME_RAY_PLACEMENT_GPU_PROBE\": \"${SLIME_RAY_PLACEMENT_GPU_PROBE}\",
    \"SLIME_SKIP_ZERO_TRAINABLE_ROLLOUT\": \"${SLIME_SKIP_ZERO_TRAINABLE_ROLLOUT}\",
    \"SLIME_SKIP_ZERO_TRAINABLE_TRAIN\": \"${SLIME_SKIP_ZERO_TRAINABLE_TRAIN}\",
    \"SETA_EXECUTION_PROFILE\": \"${SETA_EXECUTION_PROFILE:-legacy}\",
    \"SGLANG_SERVER_CONCURRENCY\": \"${SGLANG_SERVER_CONCURRENCY:-}\",
    \"SLIME_USE_FAULT_TOLERANCE\": \"${SLIME_USE_FAULT_TOLERANCE:-0}\",
    \"SLIME_SAVE_DEBUG_ROLLOUT_DATA\": \"${SLIME_SAVE_DEBUG_ROLLOUT_DATA:-}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"SLIME_HOST_IP\": \"${SLIME_HOST_IP:-}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",${EXTRA_ENV_PASSTHROUGH_JSON}
    \"USE_REMOTE_ENV\": \"${USE_REMOTE_ENV}\",
    \"ENV_SERVER_URL\": \"${ENV_SERVER_URL}\",
    \"ENV_HTTP_MAX_RETRIES\": \"${ENV_HTTP_MAX_RETRIES}\",
    \"ENV_ALLOCATE_MAX_RETRIES\": \"${ENV_ALLOCATE_MAX_RETRIES}\",
    \"ENV_ALLOCATE_RETRY_BASE_DELAY\": \"${ENV_ALLOCATE_RETRY_BASE_DELAY}\",
    \"ENV_ALLOCATE_RETRY_MAX_DELAY\": \"${ENV_ALLOCATE_RETRY_MAX_DELAY}\",
    \"ENV_ALLOCATE_RETRY_BACKOFF\": \"${ENV_ALLOCATE_RETRY_BACKOFF}\",
    \"ENV_ALLOCATE_RETRY_JITTER\": \"${ENV_ALLOCATE_RETRY_JITTER}\",
    \"HTTP_RETRY_LOG_EVERY_N\": \"${HTTP_RETRY_LOG_EVERY_N}\",
    \"HTTP_RETRY_LOG_RESPONSE_CHARS\": \"${HTTP_RETRY_LOG_RESPONSE_CHARS}\",
    \"TERMINAL_RL_GENERATE_FAILURE_TRACEBACK\": \"${TERMINAL_RL_GENERATE_FAILURE_TRACEBACK}\",
    \"ENV_EVALUATE_MAX_RETRIES\": \"${ENV_EVALUATE_MAX_RETRIES}\",
    \"ENV_CLOSE_MAX_RETRIES\": \"${ENV_CLOSE_MAX_RETRIES}\",
    \"ENV_EXEC_TOOL_MAX_RETRIES\": \"${ENV_EXEC_TOOL_MAX_RETRIES}\",
    \"ENV_ALLOCATE_HTTP_TIMEOUT\": \"${ENV_ALLOCATE_HTTP_TIMEOUT}\",
    \"ENV_RESET_HTTP_TIMEOUT\": \"${ENV_RESET_HTTP_TIMEOUT}\",
    \"ENV_RESET_MAX_RETRIES\": \"${ENV_RESET_MAX_RETRIES}\",
    \"ENV_RESET_LEASE_MAX_ATTEMPTS\": \"${ENV_RESET_LEASE_MAX_ATTEMPTS}\",
    \"ENV_RESET_LEASE_RETRY_BASE_SLEEP\": \"${ENV_RESET_LEASE_RETRY_BASE_SLEEP}\",
    \"ENV_RESET_LEASE_RETRY_MAX_SLEEP\": \"${ENV_RESET_LEASE_RETRY_MAX_SLEEP}\",
    \"ENV_CLOSE_HTTP_TIMEOUT\": \"${ENV_CLOSE_HTTP_TIMEOUT}\",
    \"ENSURE_IMAGE_TIMEOUT\": \"${ENSURE_IMAGE_TIMEOUT:-1200}\",
    \"RESET_SESSION_TIMEOUT\": \"${RESET_SESSION_TIMEOUT:-600}\",
    \"CLOSE_SESSION_TIMEOUT\": \"${CLOSE_SESSION_TIMEOUT:-60}\",
    \"EVAL_TIMEOUT\": \"${EVAL_TIMEOUT:-600}\",
    \"ENV_REMOTE_MAX_ACTIVE_TASKS\": \"${ENV_REMOTE_MAX_ACTIVE_TASKS}\",
    \"ENV_REMOTE_MAX_ACTIVE_RUNS\": \"${ENV_REMOTE_MAX_ACTIVE_RUNS}\",
    \"ENV_REMOTE_MAX_RUNS_PER_TASK\": \"${ENV_REMOTE_MAX_RUNS_PER_TASK}\",
    \"ENV_REMOTE_ADMISSION_TIMEOUT\": \"${ENV_REMOTE_ADMISSION_TIMEOUT}\",
    \"ENV_REMOTE_ADMISSION_LOG_INTERVAL\": \"${ENV_REMOTE_ADMISSION_LOG_INTERVAL}\",
    \"ENV_REMOTE_MAX_CONCURRENT_CLOSES\": \"${ENV_REMOTE_MAX_CONCURRENT_CLOSES}\",
    \"EVAL_ROLLOUT_MAX_CONCURRENCY\": \"${EVAL_ROLLOUT_MAX_CONCURRENCY}\",
    \"AGENT_SAFETYBENCH_REMOTE_ENV\": \"${AGENT_SAFETYBENCH_REMOTE_ENV}\",
    \"AGENTHARM_REMOTE_ENV\": \"${AGENTHARM_REMOTE_ENV}\",
    \"NO_PROXY\": \"${NO_PROXY}\",
    \"no_proxy\": \"${NO_PROXY}\",
    \"SETA_SAFETY\": \"${SETA_SAFETY}\",
    \"SAFETY_BENCH_REWARD\": \"${SAFETY_BENCH_REWARD}\",
    \"AGENT_SAFETYBENCH_ROOT\": \"${AGENT_SAFETYBENCH_ROOT}\",
    \"AGENTHARM_REWARD\": \"${AGENTHARM_REWARD}\",
    \"AGENTHARM_ROOT\": \"${AGENTHARM_ROOT}\",
    \"TERMINAL_SAVE_TRAJ_DIR\": \"${TERMINAL_SAVE_TRAJ_DIR}\",
    \"TRAJECTORY_SAVE_INTERVAL\": \"${TRAJECTORY_SAVE_INTERVAL}\",
    \"TRAJECTORY_SAVE_INTERVAL_SETA\": \"${TRAJECTORY_SAVE_INTERVAL_SETA}\",
    \"TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH\": \"${TRAJECTORY_SAVE_INTERVAL_AGENT_SAFETYBENCH}\",
    \"TRAJECTORY_SAVE_INTERVAL_AGENTHARM\": \"${TRAJECTORY_SAVE_INTERVAL_AGENTHARM}\",
    \"TRAJECTORY_SAVE_POLICY\": \"${TRAJECTORY_SAVE_POLICY}\",
    \"TRAJECTORY_TASK_SAVE_INTERVAL\": \"${TRAJECTORY_TASK_SAVE_INTERVAL}\",
    \"TRAJECTORY_TASK_MAX_PER_STEP\": \"${TRAJECTORY_TASK_MAX_PER_STEP}\",
    \"TRAJECTORY_TASK_MAX_PER_TASK\": \"${TRAJECTORY_TASK_MAX_PER_TASK}\",
    \"TRAJECTORY_MAX_TOTAL\": \"${TRAJECTORY_MAX_TOTAL}\",
    \"TRAJECTORY_SAVE_REWARD_STRATA\": \"${TRAJECTORY_SAVE_REWARD_STRATA}\",
    \"TRAJECTORY_SAVE_LOG_DECISIONS\": \"${TRAJECTORY_SAVE_LOG_DECISIONS}\",
    \"MIX_MODE\": \"${MIX_MODE}\",
    \"RUN_DIR\": \"${RUN_DIR}\",
    \"RUN_CATEGORY\": \"${RUN_CATEGORY}\",
    \"RUN_ID\": \"${RUN_ID}\",
    \"RUN_NAME\": \"${RUN_NAME}\",
    \"TBENCH_OUTPUT_ROOT\": \"${TBENCH_OUTPUT_ROOT}\",
    \"RUN_LOG_DIR\": \"${RUN_LOG_DIR}\",
    \"TERMINAL_STRUCTURED_METRICS\": \"${TERMINAL_STRUCTURED_METRICS}\",
    \"TERMINAL_METRICS_JSONL\": \"${TERMINAL_METRICS_JSONL}\",
    \"TERMINAL_WANDB_METRIC_PROFILE\": \"${TERMINAL_WANDB_METRIC_PROFILE}\",
    \"HARNESS_OPTION\": \"${HARNESS_OPTION}\",
    \"DATASET\": \"${DATASET}\",
    \"ALGO\": \"${ALGO}\",
    \"DAPO_OVERLONG_BUFFER_ENABLE\": \"${DAPO_OVERLONG_BUFFER_ENABLE}\",
    \"DAPO_OVERLONG_BUFFER_LEN\": \"${DAPO_OVERLONG_BUFFER_LEN}\",
    \"DAPO_OVERLONG_PENALTY_FACTOR\": \"${DAPO_OVERLONG_PENALTY_FACTOR}\",
    \"DAPO_MAX_RESPONSE_LEN\": \"${ROLLOUT_MAX_RESPONSE_LEN}\",
    \"EXPLORATION_PROFILE\": \"${EXPLORATION_PROFILE}\",
    \"EXPLORE_ENTROPY_COEF\": \"${EXPLORE_ENTROPY_COEF}\",
    \"EXPLORE_THINK_MODE\": \"${EXPLORE_THINK_MODE}\",
    \"EXPLORE_TEMP_HIGH\": \"${EXPLORE_TEMP_HIGH}\",
    \"EXPLORE_INTRINSIC\": \"${EXPLORE_INTRINSIC}\",
    \"EXPLORE_INTRINSIC_ENABLED\": \"${EXPLORE_INTRINSIC_ENABLED}\",
    \"EXPLORE_INTRINSIC_COEF\": \"${EXPLORE_INTRINSIC_COEF}\",
    \"EXPLORE_INTRINSIC_SCHEDULE\": \"${EXPLORE_INTRINSIC_SCHEDULE}\",
    \"EXPLORE_INTRINSIC_DECAY_STEPS\": \"${EXPLORE_INTRINSIC_DECAY_STEPS}\",
    \"EXPLORE_INTRINSIC_REDUCER\": \"${EXPLORE_INTRINSIC_REDUCER}\",
    \"EXPLORE_INTRINSIC_GRANULARITY\": \"${EXPLORE_INTRINSIC_GRANULARITY}\",
    \"EXPLORE_INTRINSIC_SCOPE\": \"${EXPLORE_INTRINSIC_SCOPE}\",
    \"EXPLORE_SCORE_BONUS_COMPONENTS\": \"${EXPLORE_SCORE_BONUS_COMPONENTS}\",
    \"EXPLORE_SAFETY_FILTER\": \"${EXPLORE_SAFETY_FILTER}\",
    \"EXPLORE_SAFETY_FILTER_ENABLED\": \"${EXPLORE_SAFETY_FILTER_ENABLED}\",
    \"EXPLORE_SAFETY_FILTER_COEF\": \"${EXPLORE_SAFETY_FILTER_COEF}\",
    \"EXPLORE_LPRND\": \"${EXPLORE_LPRND}\",
    \"EXPLORE_LPRND_ENABLED\": \"${EXPLORE_LPRND_ENABLED}\",
    \"EXPLORE_LPRND_COEF\": \"${EXPLORE_LPRND_COEF}\",
    \"EXPLORE_LPRND_SCHEDULE\": \"${EXPLORE_LPRND_SCHEDULE}\",
    \"EXPLORE_LPRND_DECAY_STEPS\": \"${EXPLORE_LPRND_DECAY_STEPS}\",
    \"EXPLORE_LPRND_CLIP\": \"${EXPLORE_LPRND_CLIP}\",
    \"EXPLORE_LPRND_WARMUP\": \"${EXPLORE_LPRND_WARMUP}\",
    \"EXPLORE_AGENT57_LITE\": \"${EXPLORE_AGENT57_LITE}\",
    \"EXPLORE_AGENT57_LITE_ENABLED\": \"${EXPLORE_AGENT57_LITE_ENABLED}\",
    \"EXPLORE_AGENT57_K\": \"${EXPLORE_AGENT57_K}\",
    \"EXPLORE_AGENT57_ARM_BETAS\": \"${EXPLORE_AGENT57_ARM_BETAS}\",
    \"EXPLORE_AGENT57_COMBINE_MODE\": \"${EXPLORE_AGENT57_COMBINE_MODE}\",
    \"EXPLORE_AGENT57_NGU_MOD_CLIP\": \"${EXPLORE_AGENT57_NGU_MOD_CLIP}\",
    \"EXPLORE_AGENT57_NGU_EPISODIC_SOURCE\": \"${EXPLORE_AGENT57_NGU_EPISODIC_SOURCE}\",
    \"EXPLORE_AGENT57_NGU_EPISODIC_REDUCER\": \"${EXPLORE_AGENT57_NGU_EPISODIC_REDUCER}\",
    \"EXPLORE_AGENT57_NGU_LIFE_MOD_MODE\": \"${EXPLORE_AGENT57_NGU_LIFE_MOD_MODE}\",
    \"EXPLORE_AGENT57_NGU_LIFE_MOD_STD_CLIP\": \"${EXPLORE_AGENT57_NGU_LIFE_MOD_STD_CLIP}\",
    \"EXPLORE_AGENT57_MAX_BONUS\": \"${EXPLORE_AGENT57_MAX_BONUS}\",
    \"EXPLORE_AGENT57_ARM_TEMPERATURES\": \"${EXPLORE_AGENT57_ARM_TEMPERATURES}\",
    \"EXPLORE_AGENT57_ARM_TEMPERATURE_WARMUP_ROLLOUTS\": \"${EXPLORE_AGENT57_ARM_TEMPERATURE_WARMUP_ROLLOUTS}\",
    \"EXPLORE_AGENT57_ARM_TOP_PS\": \"${EXPLORE_AGENT57_ARM_TOP_PS}\",
    \"EXPLORE_AGENT57_ARM_TOP_KS\": \"${EXPLORE_AGENT57_ARM_TOP_KS}\",
    \"EXPLORE_AGENT57_CONTROLLER\": \"${EXPLORE_AGENT57_CONTROLLER}\",
    \"EXPLORE_AGENT57_UCB_C\": \"${EXPLORE_AGENT57_UCB_C}\",
    \"EXPLORE_AGENT57_UCB_WINDOW\": \"${EXPLORE_AGENT57_UCB_WINDOW}\",
    \"EXPLORE_AGENT57_UCB_EPSILON\": \"${EXPLORE_AGENT57_UCB_EPSILON}\",
    \"EXPLORE_AGENT57_UCB_MIN_PER_ARM\": \"${EXPLORE_AGENT57_UCB_MIN_PER_ARM}\",
    \"EXPLORE_AGENT57_UCB_VALUE\": \"${EXPLORE_AGENT57_UCB_VALUE}\",
    \"EXPLORE_AGENT57_UCB_DATASET_AWARE\": \"${EXPLORE_AGENT57_UCB_DATASET_AWARE}\",
    \"EXPLORE_AGENT57_UCB_RANDOM_SEED\": \"${EXPLORE_AGENT57_UCB_RANDOM_SEED}\",
    \"EXPLORE_AGENT57_UCB_SEED_SALT\": \"${EXPLORE_AGENT57_UCB_SEED_SALT}\",
    \"EXPLORE_AGENT57_KEEP_BASELINE\": \"${EXPLORE_AGENT57_KEEP_BASELINE}\",
    \"EPISODIC_MEMORY_BACKEND\": \"${EPISODIC_MEMORY_BACKEND}\",
    \"EXPLORE_AGENT57_EPISODIC_BACKEND\": \"${EXPLORE_AGENT57_EPISODIC_BACKEND}\",
    \"EXPLORE_AGENT57_EPISODIC_CAPACITY\": \"${EXPLORE_AGENT57_EPISODIC_CAPACITY}\",
    \"EXPLORE_AGENT57_EPISODIC_COUNT_DECAY\": \"${EXPLORE_AGENT57_EPISODIC_COUNT_DECAY}\",
    \"EXPLORE_AGENT57_EPISODIC_CLEAR_ON_RESET\": \"${EXPLORE_AGENT57_EPISODIC_CLEAR_ON_RESET}\",
    \"EXPLORE_AGENT57_EPISODIC_SIMHASH_BITS\": \"${EXPLORE_AGENT57_EPISODIC_SIMHASH_BITS}\",
    \"EXPLORE_AGENT57_EPISODIC_BUCKET_CAPACITY\": \"${EXPLORE_AGENT57_EPISODIC_BUCKET_CAPACITY}\",
    \"EXPLORE_AGENT57_EPISODIC_K\": \"${EXPLORE_AGENT57_EPISODIC_K}\",
    \"EXPLORE_AGENT57_EPISODIC_DISTANCE\": \"${EXPLORE_AGENT57_EPISODIC_DISTANCE}\",
    \"EXPLORE_AGENT57_EPISODIC_VECTOR_DIM\": \"${EXPLORE_AGENT57_EPISODIC_VECTOR_DIM}\",
    \"EXPLORE_AGENT57_EPISODIC_RANDOM_SEED\": \"${EXPLORE_AGENT57_EPISODIC_RANDOM_SEED}\",
    \"EXPLORE_AGENT57_EPISODIC_OBS_MODE\": \"${EXPLORE_AGENT57_EPISODIC_OBS_MODE}\",
    \"EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN\": \"${EXPLORE_AGENT57_EPISODIC_INCLUDE_TURN}\",
    \"EXPLORE_AGENT57_EPISODIC_TURN_MODE\": \"${EXPLORE_AGENT57_EPISODIC_TURN_MODE}\",
    \"EXPLORE_AGENT57_EPISODIC_MULTI_PROBE_RADIUS\": \"${EXPLORE_AGENT57_EPISODIC_MULTI_PROBE_RADIUS}\",
    \"EXPLORE_AGENT57_EPISODIC_NOVELTY_FLOOR\": \"${EXPLORE_AGENT57_EPISODIC_NOVELTY_FLOOR}\",
    \"EXPLORE_AGENT57_LIFELONG\": \"${EXPLORE_AGENT57_LIFELONG}\",
    \"EXPLORE_AGENT57_LIFELONG_ENABLED\": \"${EXPLORE_AGENT57_LIFELONG_ENABLED}\",
    \"EXPLORE_AGENT57_LIFELONG_COEF\": \"${EXPLORE_AGENT57_LIFELONG_COEF}\",
    \"EXPLORE_AGENT57_LIFELONG_CLIP\": \"${EXPLORE_AGENT57_LIFELONG_CLIP}\",
    \"EXPLORE_AGENT57_LIFELONG_WARMUP\": \"${EXPLORE_AGENT57_LIFELONG_WARMUP}\",
    \"EXPLORE_AGENT57_LIFELONG_COUNT_DECAY\": \"${EXPLORE_AGENT57_LIFELONG_COUNT_DECAY}\",
    \"EXPLORE_AGENT57_LIFELONG_CAPACITY\": \"${EXPLORE_AGENT57_LIFELONG_CAPACITY}\",
    \"EXPLORE_AGENT57_LIFELONG_BACKEND\": \"${EXPLORE_AGENT57_LIFELONG_BACKEND}\",
    \"EXPLORE_AGENT57_LIFELONG_KEY_VERSION\": \"${EXPLORE_AGENT57_LIFELONG_KEY_VERSION}\",
    \"EXPLORE_AGENT57_LIFELONG_INCLUDE_DATASET\": \"${EXPLORE_AGENT57_LIFELONG_INCLUDE_DATASET}\",
    \"EXPLORE_AGENT57_LIFELONG_INCLUDE_TASK\": \"${EXPLORE_AGENT57_LIFELONG_INCLUDE_TASK}\",
    \"EXPLORE_AGENT57_LIFELONG_INCLUDE_TURN\": \"${EXPLORE_AGENT57_LIFELONG_INCLUDE_TURN}\",
    \"EXPLORE_AGENT57_LIFELONG_OBS_MODE\": \"${EXPLORE_AGENT57_LIFELONG_OBS_MODE}\",
    \"EXPLORE_AGENT57_LIFELONG_HIERARCHICAL\": \"${EXPLORE_AGENT57_LIFELONG_HIERARCHICAL}\",
    \"EXPLORE_AGENT57_LIFELONG_TASK_WEIGHT\": \"${EXPLORE_AGENT57_LIFELONG_TASK_WEIGHT}\",
    \"EXPLORE_AGENT57_LIFELONG_SKILL_WEIGHT\": \"${EXPLORE_AGENT57_LIFELONG_SKILL_WEIGHT}\",
    \"EXPLORE_AGENT57_LIFELONG_GLOBAL_WEIGHT\": \"${EXPLORE_AGENT57_LIFELONG_GLOBAL_WEIGHT}\",
    \"EXPLORE_AGENT57_SQLITE_BUSY_TIMEOUT_MS\": \"${EXPLORE_AGENT57_SQLITE_BUSY_TIMEOUT_MS}\",
    \"EXPLORE_AGENT57_SQLITE_WAL\": \"${EXPLORE_AGENT57_SQLITE_WAL}\",
    \"EXPLORE_AGENT57_TRUST_GATE\": \"${EXPLORE_AGENT57_TRUST_GATE}\",
    \"EXPLORE_AGENT57_TRUST_COMPLETED\": \"${EXPLORE_AGENT57_TRUST_COMPLETED}\",
    \"EXPLORE_AGENT57_TRUST_TRUNCATED\": \"${EXPLORE_AGENT57_TRUST_TRUNCATED}\",
    \"EXPLORE_AGENT57_TRUST_FAILED\": \"${EXPLORE_AGENT57_TRUST_FAILED}\",
    \"EXPLORE_AGENT57_TRUST_PARSE_ERROR\": \"${EXPLORE_AGENT57_TRUST_PARSE_ERROR}\",
    \"EXPLORE_AGENT57_TRUST_WARMUP\": \"${EXPLORE_AGENT57_TRUST_WARMUP}\",
    \"EXPLORE_AGENT57_STATE_PATH\": \"${EXPLORE_AGENT57_STATE_PATH}\",
    \"EXPLORE_AGENT57_SUCCESS_THRESHOLD\": \"${EXPLORE_AGENT57_SUCCESS_THRESHOLD}\",
    \"EXPLORE_ADVANTAGE_BONUS\": \"${EXPLORE_ADVANTAGE_BONUS}\",
    \"EXPLORE_ADVANTAGE_BONUS_ENABLED\": \"${EXPLORE_ADVANTAGE_BONUS_ENABLED}\",
    \"EXPLORE_ADVANTAGE_BONUS_MODE\": \"${EXPLORE_ADVANTAGE_BONUS_MODE}\",
    \"EXPLORE_ADVANTAGE_BONUS_COMPONENTS\": \"${EXPLORE_ADVANTAGE_BONUS_COMPONENTS}\",
    \"EXPLORE_ADVANTAGE_BONUS_COEF\": \"${EXPLORE_ADVANTAGE_BONUS_COEF}\",
    \"EXPLORE_ADVANTAGE_BONUS_CLIP\": \"${EXPLORE_ADVANTAGE_BONUS_CLIP}\",
    \"EXPLORE_ADVANTAGE_INTRINSIC_KEY\": \"${EXPLORE_ADVANTAGE_INTRINSIC_KEY}\",
    \"EXPLORE_ADVANTAGE_LAMBDA\": \"${EXPLORE_ADVANTAGE_LAMBDA}\",
    \"EXPLORE_ADVANTAGE_LAMBDA_SCHEDULE\": \"${EXPLORE_ADVANTAGE_LAMBDA_SCHEDULE}\",
    \"EXPLORE_ADVANTAGE_LAMBDA_DECAY_STEPS\": \"${EXPLORE_ADVANTAGE_LAMBDA_DECAY_STEPS}\",
    \"EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE\": \"${EXPLORE_ADVANTAGE_ARM_WEIGHT_MODE}\",
    \"EXPLORE_ADVANTAGE_TRUST_KEY\": \"${EXPLORE_ADVANTAGE_TRUST_KEY}\",
    \"EXPLORE_ADVANTAGE_GATE_MODE\": \"${EXPLORE_ADVANTAGE_GATE_MODE}\",
    \"EXPLORE_ADVANTAGE_OUTCOME_KEY\": \"${EXPLORE_ADVANTAGE_OUTCOME_KEY}\",
    \"EXPLORE_ADVANTAGE_COMPLETED_FLOOR\": \"${EXPLORE_ADVANTAGE_COMPLETED_FLOOR}\",
    \"EXPLORE_ADVANTAGE_TRUNCATED_FLOOR\": \"${EXPLORE_ADVANTAGE_TRUNCATED_FLOOR}\",
    \"EXPLORE_ADVANTAGE_FAILED_FLOOR\": \"${EXPLORE_ADVANTAGE_FAILED_FLOOR}\",
    \"EXPLORE_ADVANTAGE_ABORTED_FLOOR\": \"${EXPLORE_ADVANTAGE_ABORTED_FLOOR}\",
    \"EXPLORE_ADVANTAGE_TRUNCATED_INTRINSIC_SCALE\": \"${EXPLORE_ADVANTAGE_TRUNCATED_INTRINSIC_SCALE}\",
    \"EXPLORE_ADVANTAGE_FAILED_INTRINSIC_SCALE\": \"${EXPLORE_ADVANTAGE_FAILED_INTRINSIC_SCALE}\",
    \"EXPLORE_TRUNCATION_PENALTY\": \"${EXPLORE_TRUNCATION_PENALTY}\",
    \"EXPLORE_ADVANTAGE_TRUNCATION_PENALTY\": \"${EXPLORE_ADVANTAGE_TRUNCATION_PENALTY}\",
    \"EXPLORE_TRUNCATION_PENALTY_OUTCOME_AWARE\": \"${EXPLORE_TRUNCATION_PENALTY_OUTCOME_AWARE}\",
    \"EXPLORE_CDE_ACTOR\": \"${EXPLORE_CDE_ACTOR}\",
    \"EXPLORE_CDE_ACTOR_ENABLED\": \"${EXPLORE_CDE_ACTOR_ENABLED}\",
    \"EXPLORE_CDE_ACTOR_OMEGA\": \"${EXPLORE_CDE_ACTOR_OMEGA}\",
    \"EXPLORE_CDE_ACTOR_KAPPA\": \"${EXPLORE_CDE_ACTOR_KAPPA}\",
    \"EXPLORE_CDE_ACTOR_ALPHA\": \"${EXPLORE_CDE_ACTOR_ALPHA}\",
    \"EXPLORE_CDE_ACTOR_REWARD_GATE\": \"${EXPLORE_CDE_ACTOR_REWARD_GATE}\",
    \"EXPLORE_CDE_ACTOR_DECAY_STEPS\": \"${EXPLORE_CDE_ACTOR_DECAY_STEPS}\",
    \"EXPLORE_RETRY_ATTEMPTS\": \"${EXPLORE_RETRY_ATTEMPTS}\",
    \"EXPLORE_RETRY_TRAJ_GAMMA\": \"${EXPLORE_RETRY_TRAJ_GAMMA}\",
    \"WANDB_MODE\": \"${WANDB_MODE:-offline}\"
    ${CLAUDE_RUNTIME_ENV_JSON}
  }
}"

if [[ "${RUNTIME_ENV_XTRACE_WAS_ON}" == "1" ]]; then
  set -x
fi

RAY_JOB_SUBMISSION_ID="${RAY_JOB_SUBMISSION_ID:-terminal_rl_8b_${NUM_GPUS}gpu_$(date +%Y%m%d_%H%M%S)}"
SLIME_ENTRYPOINT="${SLIME_ENTRYPOINT:-${SLIME_DIR}/train_async.py}"
CASE_STUDY_ON_EXIT="${CASE_STUDY_ON_EXIT:-0}"
CASE_STUDY_ON_FAILURE="${CASE_STUDY_ON_FAILURE:-0}"
CASE_STUDY_CONFIG="${CASE_STUDY_CONFIG:-${REPO_ROOT}/tools/analysis/case_study_samples.yaml}"

run_case_study_if_requested() {
  local phase="$1"
  if [[ "${CASE_STUDY_ON_EXIT}" != "1" ]]; then
    return 0
  fi
  if [[ "${phase}" != "success" && "${CASE_STUDY_ON_FAILURE}" != "1" ]]; then
    return 0
  fi
  if [[ ! -d "${RUN_DIR}/trajectories" ]]; then
    log "Case-study skipped: ${RUN_DIR}/trajectories not found"
    return 0
  fi
  log "Running case-study analysis (${phase})"
  if ! CASE_STUDY_CONFIG="${CASE_STUDY_CONFIG}" \
       bash "${REPO_ROOT}/tools/analysis/run_case_study.sh" "${RUN_DIR}"; then
    log "Case-study analysis failed; training exit status is unchanged"
  fi
}

log "Submitting Ray job ${RAY_JOB_SUBMISSION_ID}"
RAY_SUBMIT_XTRACE_WAS_ON=0
if [[ "${HARNESS_OPTION}" == "claude-code" && "$-" == *x* ]]; then
  RAY_SUBMIT_XTRACE_WAS_ON=1
  set +x
fi
ray job submit --address="http://${MASTER_ADDR}:8265" \
  --submission-id "${RAY_JOB_SUBMISSION_ID}" \
  --no-wait \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${TRAIN_PYTHON}" -u "${SLIME_ENTRYPOINT}" \
  "${TRAIN_ARGS[@]}"
if [[ "${RAY_SUBMIT_XTRACE_WAS_ON}" == "1" ]]; then
  set -x
fi

set +e
ray job logs --address="http://${MASTER_ADDR}:8265" "${RAY_JOB_SUBMISSION_ID}" -f --log-style=record &
RAY_LOG_PID=$!
wait "${RAY_LOG_PID}"
RAY_LOG_EXIT=$?
RAY_LOG_PID=""
RAY_STATUS_OUTPUT=""
RAY_STATUS_STATE="unknown"
RAY_STATUS_RETRIES="${RAY_JOB_STATUS_RETRIES:-12}"
RAY_STATUS_INTERVAL="${RAY_JOB_STATUS_RETRY_DELAY:-10}"
# `ray job logs -f` can return when the dashboard briefly drops its HTTP
# connection even though the submitted job is still RUNNING.  A single status
# request at that point used to turn a healthy run into the failure/cleanup
# path.  Require an explicit terminal state and tolerate transient status
# errors before deciding what to do.
for ((status_attempt = 1; status_attempt <= RAY_STATUS_RETRIES; status_attempt++)); do
  RAY_STATUS_OUTPUT=$(ray job status --address="http://${MASTER_ADDR}:8265" "${RAY_JOB_SUBMISSION_ID}" --log-style=record 2>&1)
  echo "${RAY_STATUS_OUTPUT}"
  RAY_STATUS_LOWER=$(echo "${RAY_STATUS_OUTPUT}" | tr '[:upper:]' '[:lower:]')
  if [[ ( "${RAY_STATUS_LOWER}" == *"status for job"* || "${RAY_STATUS_LOWER}" == *"job '${RAY_JOB_SUBMISSION_ID}' succeeded"* ) && "${RAY_STATUS_LOWER}" == *"succeeded"* ]]; then
    RAY_STATUS_STATE="succeeded"
    break
  fi
  if [[ ( "${RAY_STATUS_LOWER}" == *"status for job"* || "${RAY_STATUS_LOWER}" == *"job '${RAY_JOB_SUBMISSION_ID}' failed"* || "${RAY_STATUS_LOWER}" == *"job '${RAY_JOB_SUBMISSION_ID}' stopped"* || "${RAY_STATUS_LOWER}" == *"job '${RAY_JOB_SUBMISSION_ID}' was stopped"* ) && ( "${RAY_STATUS_LOWER}" == *"failed"* || "${RAY_STATUS_LOWER}" == *"stopped"* ) ]]; then
    RAY_STATUS_STATE="failed"
    break
  fi
  if [[ "${RAY_STATUS_LOWER}" == *"currently running"* || "${RAY_STATUS_LOWER}" == *"pending"* ]]; then
    RAY_STATUS_STATE="running"
  fi
  if (( status_attempt < RAY_STATUS_RETRIES )); then
    log "Ray job status is transient/unavailable (attempt ${status_attempt}/${RAY_STATUS_RETRIES}); preserving the run and retrying in ${RAY_STATUS_INTERVAL}s"
    sleep "${RAY_STATUS_INTERVAL}"
  fi
done
if [[ "${RAY_STATUS_STATE}" == "running" ]]; then
  # A still-running job after the bounded poll is not a failure.  Keep the
  # cluster alive so an operator can inspect it or resume log streaming.
  RAY_STATUS_STATE="unknown"
fi
set -e

# Checkpoint pruning happens transactionally in checkpoint_utils.py.  Do not
# delete iter_* directories here: a newer directory can be an incomplete save,
# while the commit marker still points at the older recoverable checkpoint.

if [[ "${RAY_STATUS_STATE}" == "succeeded" ]]; then
  run_case_study_if_requested success
  log "Ray job succeeded"
  exit 0
fi

if [[ "${RAY_STATUS_STATE}" == "unknown" ]]; then
  log "Ray job state remains unknown after ${RAY_STATUS_RETRIES} status attempts; leaving Ray/SGLang processes untouched for diagnosis"
  exit 2
fi

# ── Failure auto-capture ─────────────────────────────────────────────
# Generate two condensed artifacts under the run-contained log mirror:
#   gpu_tail.log : last 300 lines (often enough to see the actual stack)
#   gpu_err.log  : grep-filtered "real" error lines (CUDA/OOM/Exception/etc.)
log "Ray job failed (logs exit: ${RAY_LOG_EXIT}). Writing condensed artifacts..."
tail -n 300 "${RUN_LOG}" > "${GPU_TAIL_LOG}" 2>/dev/null || true
grep -E "Error|Exception|Traceback|CUDA|OOM|invalid device|FAILED|CheckpointException|ENOSPC|PermissionError|Connect call failed|ConnectorError|500|502" \
     "${RUN_LOG}" 2>/dev/null \
  | grep -v "FutureWarning" \
  | grep -v "DeprecationWarning" \
  | tail -n 200 \
  > "${GPU_ERR_LOG}" 2>/dev/null || true

cat <<EOF
========================================
  Run failed. Inspect:
    full   : ${GPU_RUN_LOG}
    errors : ${GPU_ERR_LOG}
    tail   : ${GPU_TAIL_LOG}
    latest : ${TMP_DOC_LATEST}/
========================================
EOF
run_case_study_if_requested failure
exit 1
