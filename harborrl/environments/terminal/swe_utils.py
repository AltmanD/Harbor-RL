from __future__ import annotations

SWE_TASK_PREFIXES = ("swesmith_env/", "sweverified_env/")


def is_swe_task_path(task_path: str) -> bool:
    return task_path.startswith(SWE_TASK_PREFIXES)


def build_swe_user_message(
    *, task_name: str, task_path: str, instruction: str
) -> str:
    lines = [f"Task name:{task_name}"]
    if is_swe_task_path(task_path):
        lines.extend(
            [
                "Workspace: The target repository is already checked out at /testbed "
                "at the required base commit.",
                "Work directly in /testbed. Do not clone, replace, or initialize another "
                "repository; the final patch is collected only from /testbed.",
            ]
        )
    lines.append(f"Task instruction: {instruction}")
    return "\n".join(lines)
