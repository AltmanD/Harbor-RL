#!/usr/bin/env python3
"""Docker data-root garbage collector for HarborRL CPU workers.

The script is intentionally Docker-CLI based so it can run on the CPU servers
without the Docker Python SDK. It performs progressive cleanup. Old tagged
images are diagnosed by default and are removed only when explicitly enabled.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import fnmatch
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


GB = 1024**3


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class Logger:
    def __init__(self, log_file: str | None = None):
        self._fh = None
        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()

    def log(self, msg: str) -> None:
        line = f"[{dt.datetime.now().strftime('%F %T')}] {msg}"
        print(line, flush=True)
        if self._fh is not None:
            self._fh.write(line + "\n")
            self._fh.flush()


@dataclass
class FsStats:
    path: str
    total_bytes: int
    used_bytes: int
    avail_bytes: int
    used_pct: float
    inode_pct: float | None = None

    @property
    def avail_gb(self) -> float:
        return self.avail_bytes / GB

    @property
    def used_gb(self) -> float:
        return self.used_bytes / GB

    @property
    def total_gb(self) -> float:
        return self.total_bytes / GB


@dataclass
class ImageInfo:
    image_id: str
    created: dt.datetime
    size_bytes: int
    repo_tags: list[str] = field(default_factory=list)
    repo_digests: list[str] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def display_tags(self) -> str:
        values = self.repo_tags or self.repo_digests or ["<none>:<none>"]
        return ",".join(values)


def run_cmd(
    args: list[str],
    *,
    timeout: int,
    check: bool = False,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=check,
    )


def docker_cmd(args: list[str], *, timeout: int, check: bool = False) -> subprocess.CompletedProcess[str]:
    return run_cmd(["docker", *args], timeout=timeout, check=check)


def chunked(items: list[str], n: int) -> list[list[str]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def parse_time(value: str | None) -> dt.datetime:
    if not value:
        return dt.datetime.fromtimestamp(0, tz=dt.timezone.utc)
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:
        return dt.datetime.fromtimestamp(0, tz=dt.timezone.utc)


def docker_root_from_daemon(timeout: int) -> str | None:
    try:
        cp = docker_cmd(["info", "--format", "{{.DockerRootDir}}"], timeout=timeout, check=True)
    except Exception:
        return None
    root = cp.stdout.strip()
    return root or None


def fs_stats(path: str) -> FsStats:
    usage = shutil.disk_usage(path)
    used = usage.total - usage.free
    inode_pct: float | None = None
    try:
        cp = run_cmd(["df", "-Pi", path], timeout=10)
        if cp.returncode == 0:
            parts = cp.stdout.strip().splitlines()[-1].split()
            inode_pct = float(parts[4].rstrip("%"))
    except Exception:
        inode_pct = None
    return FsStats(
        path=path,
        total_bytes=usage.total,
        used_bytes=used,
        avail_bytes=usage.free,
        used_pct=(used / usage.total * 100.0) if usage.total else 0.0,
        inode_pct=inode_pct,
    )


def stat_line(stats: FsStats) -> str:
    inode = "?" if stats.inode_pct is None else f"{stats.inode_pct:.1f}%"
    return (
        f"{stats.path}: used={stats.used_pct:.1f}% "
        f"({stats.used_gb:.1f}G/{stats.total_gb:.1f}G) "
        f"free={stats.avail_gb:.1f}G inode={inode}"
    )


def target_reached(stats: FsStats, target_used_pct: float, min_free_gb: float | None) -> bool:
    if stats.used_pct > target_used_pct:
        return False
    if min_free_gb is not None and stats.avail_gb < min_free_gb:
        return False
    return True


def trigger_exceeded(stats: FsStats, trigger_used_pct: float, min_free_gb: float | None) -> bool:
    if stats.used_pct >= trigger_used_pct:
        return True
    if min_free_gb is not None and stats.avail_gb < min_free_gb:
        return True
    return False


def read_keep_patterns(values: list[str], keep_file: str | None) -> list[str]:
    patterns: list[str] = []
    for item in values:
        for part in item.replace("\n", ",").split(","):
            part = part.strip()
            if part:
                patterns.append(part)
    if keep_file:
        for line in Path(keep_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return patterns


def image_matches_keep(img: ImageInfo, patterns: list[str]) -> bool:
    candidates = [img.image_id, img.image_id.replace("sha256:", "")]
    candidates.extend(img.repo_tags)
    candidates.extend(img.repo_digests)
    for tag in img.repo_tags:
        repo = tag.rsplit(":", 1)[0] if ":" in tag else tag
        candidates.append(repo)
    for pattern in patterns:
        if any(fnmatch.fnmatchcase(value, pattern) for value in candidates):
            return True
    return False


def labels_match(img: ImageInfo, label_rules: list[str]) -> bool:
    if not label_rules:
        return False
    for rule in label_rules:
        if "=" in rule:
            key, value = rule.split("=", 1)
            if img.labels.get(key) == value:
                return True
        elif rule in img.labels:
            return True
    return False


def docker_object_ids(kind: str, timeout: int) -> list[str]:
    if kind == "containers":
        args = ["ps", "-aq", "--no-trunc"]
    elif kind == "running_containers":
        args = ["ps", "-q", "--no-trunc"]
    elif kind == "images":
        args = ["image", "ls", "-q", "--no-trunc"]
    else:
        raise ValueError(kind)
    cp = docker_cmd(args, timeout=timeout)
    if cp.returncode != 0:
        return []
    return list(dict.fromkeys(x.strip() for x in cp.stdout.splitlines() if x.strip()))


def protected_image_ids(timeout: int, protect_all_containers: bool) -> set[str]:
    kind = "containers" if protect_all_containers else "running_containers"
    container_ids = docker_object_ids(kind, timeout)
    protected: set[str] = set()
    if not container_ids:
        return protected
    for group in chunked(container_ids, 50):
        cp = docker_cmd(["inspect", *group], timeout=timeout)
        if cp.returncode != 0:
            continue
        try:
            data = json.loads(cp.stdout)
        except json.JSONDecodeError:
            continue
        for item in data:
            image_id = str(item.get("Image") or "")
            if image_id:
                protected.add(image_id)
                protected.add(image_id.replace("sha256:", ""))
    return protected


def list_images(timeout: int, logger: Logger) -> list[ImageInfo]:
    image_ids = docker_object_ids("images", timeout)
    images: list[ImageInfo] = []
    for group in chunked(image_ids, 50):
        cp = docker_cmd(["image", "inspect", *group], timeout=timeout)
        if cp.returncode != 0:
            logger.log(f"WARN: docker image inspect failed for {len(group)} images: {cp.stderr.strip()[:300]}")
            continue
        try:
            data = json.loads(cp.stdout)
        except json.JSONDecodeError as exc:
            logger.log(f"WARN: docker image inspect returned invalid JSON: {exc}")
            continue
        for item in data:
            cfg = item.get("Config") if isinstance(item.get("Config"), dict) else {}
            labels = cfg.get("Labels") if isinstance(cfg.get("Labels"), dict) else {}
            images.append(
                ImageInfo(
                    image_id=str(item.get("Id") or ""),
                    created=parse_time(item.get("Created")),
                    size_bytes=int(item.get("Size") or 0),
                    repo_tags=list(item.get("RepoTags") or []),
                    repo_digests=list(item.get("RepoDigests") or []),
                    labels={str(k): str(v) for k, v in labels.items()},
                )
            )
    return images


def run_cleanup_command(
    name: str,
    cmd: list[str],
    *,
    args: argparse.Namespace,
    logger: Logger,
) -> None:
    before = fs_stats(args.docker_root)
    logger.log(f"STEP {name}: before {stat_line(before)}")
    printable = " ".join(shlex.quote(x) for x in cmd)
    if args.dry_run:
        logger.log(f"DRY_RUN: would run: {printable}")
        return
    try:
        cp = run_cmd(cmd, timeout=args.prune_timeout)
    except subprocess.TimeoutExpired:
        logger.log(f"WARN: step timed out after {args.prune_timeout}s: {printable}")
        return
    if cp.stdout.strip():
        for line in cp.stdout.strip().splitlines()[-12:]:
            logger.log(f"  {line}")
    if cp.stderr.strip():
        for line in cp.stderr.strip().splitlines()[-12:]:
            logger.log(f"  STDERR: {line}")
    if cp.returncode != 0:
        logger.log(f"WARN: step exited rc={cp.returncode}: {printable}")
    after = fs_stats(args.docker_root)
    freed = before.used_bytes - after.used_bytes
    logger.log(f"STEP {name}: after  {stat_line(after)} freed={freed / GB:.2f}G")


def progressive_builtin_cleanup(args: argparse.Namespace, logger: Logger) -> None:
    steps: list[tuple[str, list[str]]] = [
        ("container-prune", ["docker", "container", "prune", "-f"]),
    ]
    if args.prune_volumes:
        steps.append(("volume-prune", ["docker", "volume", "prune", "-f"]))
    else:
        logger.log("STEP volume-prune: disabled (set DOCKER_GC_PRUNE_VOLUMES=1 or --prune-volumes to enable)")
    steps.extend(
        [
            ("dangling-image-prune", ["docker", "image", "prune", "-f"]),
            (
                "builder-prune",
                ["docker", "builder", "prune", "-af", "--filter", f"until={args.builder_cache_until}"],
            ),
        ]
    )
    for name, cmd in steps:
        if target_reached(fs_stats(args.docker_root), args.target_used_pct, args.min_free_gb):
            logger.log(f"Target reached before {name}; stopping builtin cleanup.")
            return
        run_cleanup_command(name, cmd, args=args, logger=logger)


def docker_count(command: list[str], timeout: int) -> str:
    cp = docker_cmd(command, timeout=timeout)
    if cp.returncode != 0:
        return "?"
    return str(len([x for x in cp.stdout.splitlines() if x.strip()]))


def shell_count(command: list[str], timeout: int) -> str:
    try:
        cp = run_cmd(command, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout"
    if cp.returncode != 0:
        return "?"
    return cp.stdout.strip() or "0"


def run_diagnostics(args: argparse.Namespace, logger: Logger) -> None:
    logger.log("Diagnostics: Docker object counts")
    logger.log(
        "  "
        f"containers={docker_count(['ps', '-aq'], args.docker_timeout)} "
        f"running={docker_count(['ps', '-q'], args.docker_timeout)} "
        f"images={docker_count(['image', 'ls', '-q'], args.docker_timeout)} "
        f"volumes={docker_count(['volume', 'ls', '-q'], args.docker_timeout)} "
        f"networks={docker_count(['network', 'ls', '-q'], args.docker_timeout)}"
    )
    overlay_root = Path(args.docker_root) / "overlay2"
    if overlay_root.is_dir():
        count = shell_count(
            [
                "find",
                str(overlay_root),
                "-mindepth",
                "1",
                "-maxdepth",
                "1",
                "-type",
                "d",
                "!",
                "-name",
                "l",
            ],
            args.docker_timeout,
        )
        if count not in {"?", "timeout"}:
            count = str(len(count.splitlines()))
        logger.log(f"Diagnostics: overlay2 directory count={count}")
    if args.run_docker_df:
        logger.log(f"Diagnostics: docker system df -v (timeout={args.docker_timeout}s)")
        try:
            cp = docker_cmd(["system", "df", "-v"], timeout=args.docker_timeout)
            text = (cp.stdout or cp.stderr).strip()
        except subprocess.TimeoutExpired:
            text = "timeout"
        for line in text.splitlines()[:120]:
            logger.log(f"  {line}")
    else:
        logger.log("Diagnostics: skipping docker system df -v; pass --run-docker-df if needed.")
    if args.run_du:
        logger.log(f"Diagnostics: top-level {args.docker_root} usage (timeout={args.du_timeout}s)")
        try:
            cp = run_cmd(["du", "-xhd1", args.docker_root], timeout=args.du_timeout)
            lines = (cp.stdout or cp.stderr).splitlines()
        except subprocess.TimeoutExpired:
            lines = ["timeout"]
        for line in lines[-40:]:
            logger.log(f"  {line}")
    else:
        logger.log("Diagnostics: skipping du scan; pass --run-du for slower top-level size accounting.")


def remove_old_images(args: argparse.Namespace, logger: Logger, keep_patterns: list[str]) -> None:
    stats = fs_stats(args.docker_root)
    if target_reached(stats, args.target_used_pct, args.min_free_gb):
        logger.log("Target already reached; skipping old-image LRU cleanup.")
        return

    protected = protected_image_ids(args.docker_timeout, args.protect_all_container_images)
    logger.log(f"Protected image ids from containers: {len(protected)}")

    now = dt.datetime.now(dt.timezone.utc)
    min_age = dt.timedelta(hours=args.min_image_age_hours)
    images = list_images(args.docker_timeout, logger)
    candidates: list[ImageInfo] = []
    skipped = {
        "protected": 0,
        "keep_pattern": 0,
        "keep_label": 0,
        "too_new": 0,
        "invalid": 0,
    }
    for img in images:
        short_id = img.image_id.replace("sha256:", "")
        if not img.image_id:
            skipped["invalid"] += 1
            continue
        if img.image_id in protected or short_id in protected:
            skipped["protected"] += 1
            continue
        if image_matches_keep(img, keep_patterns):
            skipped["keep_pattern"] += 1
            continue
        if labels_match(img, args.keep_label):
            skipped["keep_label"] += 1
            continue
        if img.created.tzinfo is None:
            created = img.created.replace(tzinfo=dt.timezone.utc)
        else:
            created = img.created.astimezone(dt.timezone.utc)
        if now - created < min_age:
            skipped["too_new"] += 1
            continue
        candidates.append(img)

    candidates.sort(key=lambda x: (x.created, x.image_id))
    virtual_size_gb = sum(x.size_bytes for x in candidates) / GB
    logger.log(
        "Old-image candidates: "
        f"{len(candidates)} virtual_size={virtual_size_gb:.1f}G skipped={skipped}"
    )
    if args.dry_run:
        for img in candidates[: args.max_log_items]:
            logger.log(
                "DRY_RUN: would remove image "
                f"{img.image_id[:19]} created={img.created.isoformat()} "
                f"size={img.size_bytes / GB:.2f}G tags={img.display_tags}"
            )
        if len(candidates) > args.max_log_items:
            logger.log(f"DRY_RUN: suppressed {len(candidates) - args.max_log_items} more candidates")
        return
    if not args.delete_old_images:
        logger.log(
            "Old-image LRU deletion is disabled. "
            "Set DOCKER_GC_DELETE_OLD_IMAGES=1 or pass --delete-old-images to enable it."
        )
        for img in candidates[: min(args.max_log_items, 20)]:
            logger.log(
                "Candidate not removed: "
                f"{img.image_id[:19]} created={img.created.isoformat()} "
                f"size={img.size_bytes / GB:.2f}G tags={img.display_tags}"
            )
        if len(candidates) > min(args.max_log_items, 20):
            logger.log(f"Suppressed {len(candidates) - min(args.max_log_items, 20)} more old-image candidates")
        return

    removed = 0
    failed = 0
    for img in candidates:
        if args.max_old_images > 0 and removed >= args.max_old_images:
            logger.log(f"Reached max old-image removals: {args.max_old_images}")
            break
        stats = fs_stats(args.docker_root)
        if target_reached(stats, args.target_used_pct, args.min_free_gb):
            logger.log(f"Target reached during old-image cleanup: {stat_line(stats)}")
            break
        cmd = ["docker", "image", "rm"]
        if args.force_rmi:
            cmd.append("--force")
        cmd.append(img.image_id)
        logger.log(
            f"Removing old image {img.image_id[:19]} created={img.created.isoformat()} "
            f"size={img.size_bytes / GB:.2f}G tags={img.display_tags}"
        )
        try:
            cp = run_cmd(cmd, timeout=args.prune_timeout)
        except subprocess.TimeoutExpired:
            failed += 1
            logger.log(f"WARN: docker image rm timed out for {img.image_id[:19]}")
            continue
        if cp.returncode == 0:
            removed += 1
            if cp.stdout.strip():
                for line in cp.stdout.strip().splitlines()[-6:]:
                    logger.log(f"  {line}")
        else:
            failed += 1
            logger.log(
                f"WARN: failed to remove {img.image_id[:19]} rc={cp.returncode}: "
                f"{(cp.stderr or cp.stdout).strip()[:500]}"
            )
    logger.log(f"Old-image cleanup complete: removed={removed} failed={failed}")


def acquire_lock(path: str, logger: Logger):
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open("w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.log(f"Another docker GC instance is running (lock={path}); exiting.")
        sys.exit(0)
    return fh


def parse_args() -> argparse.Namespace:
    env_keep = os.getenv("DOCKER_GC_KEEP_PATTERNS", "")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-root", default=os.getenv("DOCKER_DATA_ROOT") or os.getenv("DOCKER_ROOT"))
    parser.add_argument("--trigger-used-pct", type=float, default=env_float("DOCKER_GC_TRIGGER_USED_PCT", 85.0))
    parser.add_argument("--target-used-pct", type=float, default=env_float("DOCKER_GC_TARGET_USED_PCT", 70.0))
    parser.add_argument("--min-free-gb", type=float, default=None if os.getenv("DOCKER_GC_MIN_FREE_GB") in {None, ""} else env_float("DOCKER_GC_MIN_FREE_GB", 0.0))
    parser.add_argument("--dry-run", action="store_true", default=env_bool("DOCKER_GC_DRY_RUN", False))
    parser.add_argument("--force-cleanup", action="store_true", default=env_bool("DOCKER_GC_FORCE_CLEANUP", False))
    parser.add_argument("--diagnose", action="store_true", default=env_bool("DOCKER_GC_DIAGNOSE", False))
    parser.add_argument("--diagnose-only", action="store_true", default=env_bool("DOCKER_GC_DIAGNOSE_ONLY", False))
    parser.add_argument("--run-du", action="store_true", default=env_bool("DOCKER_GC_RUN_DU", False))
    parser.add_argument("--run-docker-df", action="store_true", default=env_bool("DOCKER_GC_RUN_DOCKER_DF", False))
    parser.add_argument("--keep", action="append", default=[env_keep] if env_keep else [], help="fnmatch pattern for repo:tag, repo, digest, or image id. May be repeated or comma-separated.")
    parser.add_argument("--keep-file", default=os.getenv("DOCKER_GC_KEEP_FILE"))
    parser.add_argument("--keep-label", action="append", default=[], help="Protect images with label key or key=value.")
    parser.add_argument("--prune-volumes", dest="prune_volumes", action="store_true", default=env_bool("DOCKER_GC_PRUNE_VOLUMES", True))
    parser.add_argument("--no-prune-volumes", dest="prune_volumes", action="store_false")
    parser.add_argument("--builder-cache-until", default=os.getenv("DOCKER_GC_BUILDER_CACHE_UNTIL", "24h"))
    parser.add_argument("--delete-old-images", action="store_true", default=env_bool("DOCKER_GC_DELETE_OLD_IMAGES", False))
    parser.add_argument("--min-image-age-hours", type=float, default=env_float("DOCKER_GC_MIN_IMAGE_AGE_HOURS", 24.0))
    parser.add_argument("--max-old-images", type=int, default=env_int("DOCKER_GC_MAX_OLD_IMAGES", 0), help="0 means unlimited until target is reached.")
    parser.add_argument("--max-log-items", type=int, default=env_int("DOCKER_GC_MAX_LOG_ITEMS", 100))
    parser.add_argument("--docker-timeout", type=int, default=env_int("DOCKER_GC_DOCKER_TIMEOUT", 30))
    parser.add_argument("--prune-timeout", type=int, default=env_int("DOCKER_GC_PRUNE_TIMEOUT", 180))
    parser.add_argument("--du-timeout", type=int, default=env_int("DOCKER_GC_DU_TIMEOUT", 180))
    parser.add_argument("--force-rmi", action="store_true", default=env_bool("DOCKER_GC_FORCE_RMI", False))
    parser.add_argument("--protect-running-only", dest="protect_all_container_images", action="store_false", default=True)
    parser.add_argument("--log-file", default=os.getenv("DOCKER_GC_LOG_FILE"))
    parser.add_argument("--summary-json", default=os.getenv("DOCKER_GC_SUMMARY_JSON"))
    parser.add_argument("--lock-file", default=os.getenv("DOCKER_GC_LOCK_FILE", "/tmp/openclaw_docker_storage_gc.lock"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = Logger(args.log_file)
    lock_fh = None
    started = time.time()
    try:
        lock_fh = acquire_lock(args.lock_file, logger)
        if not args.docker_root:
            args.docker_root = docker_root_from_daemon(args.docker_timeout) or "/var/lib/docker"
        if not Path(args.docker_root).exists():
            logger.log(f"ERROR: docker root does not exist: {args.docker_root}")
            return 1
        try:
            docker_cmd(["info"], timeout=args.docker_timeout, check=True)
        except Exception as exc:
            logger.log(f"ERROR: docker daemon is not available: {exc}")
            return 1

        keep_patterns = read_keep_patterns(args.keep, args.keep_file)
        before = fs_stats(args.docker_root)
        logger.log("Docker storage GC starting")
        logger.log(f"docker_root={args.docker_root} dry_run={int(args.dry_run)}")
        logger.log(
            f"trigger_used_pct={args.trigger_used_pct:.1f} target_used_pct={args.target_used_pct:.1f} "
            f"min_free_gb={args.min_free_gb if args.min_free_gb is not None else '<none>'}"
        )
        logger.log(
            f"keep_patterns={keep_patterns or '<none>'} prune_volumes={int(args.prune_volumes)} "
            f"delete_old_images={int(args.delete_old_images)} min_image_age_hours={args.min_image_age_hours} "
            f"force_rmi={int(args.force_rmi)}"
        )
        logger.log(f"Before: {stat_line(before)}")
        if args.diagnose or args.diagnose_only:
            run_diagnostics(args, logger)
        if args.diagnose_only:
            return 0

        if not args.force_cleanup and not trigger_exceeded(before, args.trigger_used_pct, args.min_free_gb):
            logger.log("Below trigger threshold; no cleanup needed.")
            return 0

        progressive_builtin_cleanup(args, logger)
        remove_old_images(args, logger, keep_patterns)

        after = fs_stats(args.docker_root)
        freed = before.used_bytes - after.used_bytes
        logger.log(f"After:  {stat_line(after)}")
        logger.log(f"Freed: {freed / GB:.2f}G elapsed={time.time() - started:.1f}s")
        ok = target_reached(after, args.target_used_pct, args.min_free_gb)
        logger.log(f"Target reached: {int(ok)}")
        if args.summary_json:
            Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.summary_json).write_text(
                json.dumps(
                    {
                        "docker_root": args.docker_root,
                        "dry_run": args.dry_run,
                        "before": before.__dict__,
                        "after": after.__dict__,
                        "freed_bytes": freed,
                        "target_reached": ok,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        if args.dry_run:
            return 0
        return 0 if ok else 2
    finally:
        if lock_fh is not None:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            lock_fh.close()
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
