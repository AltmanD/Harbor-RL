import base64
import io
import json
import logging
from pathlib import Path

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    ProcessorMixin,
)

logger = logging.getLogger(__name__)

# Default image patch size for vision-language models
# Note: Qwen3-VL uses 16, Qwen2.5-VL uses 14
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/README.md
DEFAULT_PATCH_SIZE = 14


def load_tokenizer(name_or_path: str, **kwargs):
    try:
        return AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    except ValueError as exc:
        # GLM-5.1 tokenizer_config declares ``TokenizersBackend``, a class
        # introduced after the transformers version shipped in the runtime
        # image (4.57).  The underlying tokenizer.json is standard tokenizers
        # format, so construct the fast wrapper directly and preserve the
        # model's chat template/special-token metadata.
        if "TokenizersBackend" not in str(exc):
            raise
        root = Path(name_or_path)
        tok = PreTrainedTokenizerFast(
            tokenizer_file=str(root / "tokenizer.json"),
            **{k: kwargs[k] for k in ("trust_remote_code",) if k in kwargs},
        )
        cfg_path = root / "tokenizer_config.json"
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            for key in ("chat_template", "bos_token", "eos_token", "pad_token", "unk_token"):
                value = cfg.get(key)
                if value is not None and key == "chat_template":
                    tok.chat_template = value
                elif value is not None and getattr(tok, key, None) is None:
                    setattr(tok, key, value)
        logger.warning("Using PreTrainedTokenizerFast fallback for %s: %s", name_or_path, exc)
        return tok


def load_processor(name_or_path: str, **kwargs):
    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load processor from {name_or_path}: {e}")
        proc = None

    # If HF returned a tokenizer, discard it.
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        proc = None

    return proc


def process_vision_info(prompt, processor):
    # temporary solution, will write image utils for slime later
    from qwen_vl_utils import process_vision_info

    if hasattr(processor.image_processor, "patch_size"):
        image_patch_size = processor.image_processor.patch_size
    else:
        logger.info(f"Using default patch size: {DEFAULT_PATCH_SIZE}")
        image_patch_size = DEFAULT_PATCH_SIZE
    images, videos = process_vision_info(prompt, image_patch_size=image_patch_size)
    multimodal_inputs = {"images": images, "videos": videos}
    return multimodal_inputs


def encode_image_for_rollout_engine(image) -> str:
    """Load an image from path, ensure RGB, encode as PNG base64 string."""
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
