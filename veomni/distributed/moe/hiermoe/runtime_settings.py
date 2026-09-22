# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
from contextlib import nullcontext

import torch


try:
    from torch.distributed._tensor import DTensor
except ImportError:  # pragma: no cover - older torch fallback
    DTensor = ()  # type: ignore[assignment]


from ....utils import logging
from .placemoe.runtime import (
    PlaceMoERuntimeConfig,
)


logger = logging.get_logger(__name__)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s.", name, raw, default)
        return default


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s.", name, raw, default)
        return default


def _env_candidate_shards(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.lower() == "auto":
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Invalid %s=%r; using automatic candidate sharding.", name, raw)
        return 0


_MAX_SWAP_BUCKET_BYTES = _env_int("VEOMNI_HIERMOE_SWAP_BUCKET_MIB", 1024) * 1024 * 1024
_MAX_SWAP_WAVE_BYTES = _env_int("VEOMNI_HIERMOE_SWAP_WAVE_MIB", 2048) * 1024 * 1024
_EXACT_SINGLE_SWAP_MAX_EXPERTS = 256
_EXACT_SINGLE_SWAP_MAX_STATS_BYTES = 64 * 1024 * 1024
_EXACT_P1_ROUTE_SAMPLE_SIZE = _env_int(
    "VEOMNI_HIERMOE_EXACT_P1_ROUTE_SAMPLE_SIZE",
    0,
    minimum=0,
)
_SWAP_COST_CHUNK_CANDIDATES = _env_int("VEOMNI_HIERMOE_SWAP_COST_CHUNK_CANDIDATES", 96)
_GREEDY_LAYER_PARALLEL_STREAMS = _env_int("VEOMNI_HIERMOE_GREEDY_LAYER_STREAMS", 8)
_GREEDY_ADAPTIVE_TOPK_INITIAL = _env_int("VEOMNI_HIERMOE_GREEDY_ADAPTIVE_TOPK_INITIAL", 32)
_GREEDY_EXACT_PRIMITIVE_TOPK = _env_int(
    "VEOMNI_HIERMOE_GREEDY_EXACT_PRIMITIVE_TOPK",
    0,
    minimum=0,
)
_ALL_CANDIDATE_PAIR_CACHE: dict[tuple[str, int], torch.Tensor] = {}


def _env_flag(name: str) -> bool:
    raw = os.environ.get(name)
    return raw is not None and raw.lower() in {"1", "true", "yes", "on", "y"}


_USE_FAST_2D_SELECTOR = not _env_flag("VEOMNI_HIERMOE_SWAP_DISABLE_FAST_2D")
_USE_GLOBAL_2D_SELECTOR = not _env_flag("VEOMNI_HIERMOE_SWAP_DISABLE_GLOBAL_2D")
_USE_GLOBAL_HIERARCHY_SELECTOR = not _env_flag("VEOMNI_HIERMOE_SWAP_DISABLE_GLOBAL_HIERARCHY")
_SWAP_CANDIDATE_SHARDS = _env_candidate_shards("VEOMNI_HIERMOE_SWAP_CANDIDATE_SHARDS")
_FIXED_R2_LAYOUT = _env_flag("VEOMNI_HIERMOE_FIXED_R2_LAYOUT")
_FORCE_FIXED_R2_MIRRORED_REMAP = _env_flag("VEOMNI_HIERMOE_FORCE_FIXED_R2_MIRRORED_REMAP")
_GREEDY_ADAPTIVE_TOPK = _env_flag("VEOMNI_HIERMOE_GREEDY_ADAPTIVE_TOPK")
_GREEDY_ADAPTIVE_TOPK_STRICT = _env_flag("VEOMNI_HIERMOE_GREEDY_ADAPTIVE_TOPK_STRICT")
_GREEDY_POST_SHORTLIST_COMPACT_PAIR = _env_flag("VEOMNI_HIERMOE_GREEDY_POST_SHORTLIST_COMPACT_PAIR")
_GREEDY_EXACT_PRIMITIVE_MAX_ONLY = _env_flag("VEOMNI_HIERMOE_GREEDY_EXACT_PRIMITIVE_MAX_ONLY")
_PIPELINE_STAGE_TIMING = _env_flag("VEOMNI_HIERMOE_PIPELINE_STAGE_TIMING")
_PIPELINE_PREPARE_SUBSTAGES = (
    "planner_setup",
    "context",
    "route_hash",
    "baseline_route",
    "occupancy",
    "candidate_routes",
    "pair_events",
    "unary_statistics",
    "unary_scoring",
    "pair_statistics",
    "pair_interaction",
    "candidate_pack",
    "collective_pack",
)
_PIPELINE_PREPARE_CUT_POINTS = (2, 2, 5, 7, 10, 13)
# Event.wait() still wakes immediately on the normal path.  The timeout only
# controls how often a blocked planner checks for an exceptional future or
# shutdown.  A 1 ms timeout makes 48 layer workers contend for the GIL and the
# manager lock tens of thousands of times per second while they are supposed
# to be dormant between fixed pipeline windows.
_PIPELINE_HOST_EVENT_POLL_SECONDS = 0.05
_PIPELINE_PLAN_WORKERS = _env_int("VEOMNI_HIERMOE_PIPELINE_PLAN_WORKERS", 64)
_ABLATION_REPLAY_PATH = os.environ.get("VEOMNI_HIERMOE_ABLATION_REPLAY_PATH", "").strip()
_ABLATION_REPLAY_MODE = os.environ.get("VEOMNI_HIERMOE_ABLATION_REPLAY_MODE", "off").strip().lower()
_ABLATION_MIGRATION_MODE = os.environ.get("VEOMNI_HIERMOE_ABLATION_MIGRATION_MODE", "hidden").strip().lower()
_ABLATION_GRAD_MODE = os.environ.get("VEOMNI_HIERMOE_ABLATION_GRAD_MODE", "hidden").strip().lower()
_INITIAL_LAYOUT_PATH = os.environ.get("VEOMNI_HIERMOE_INITIAL_LAYOUT", "").strip()
_PLACEMOE_RUNTIME_CONFIG = PlaceMoERuntimeConfig.from_environment()
if _PLACEMOE_RUNTIME_CONFIG.source_path:
    _INITIAL_LAYOUT_PATH = _PLACEMOE_RUNTIME_CONFIG.initial_artifact
    _ABLATION_REPLAY_PATH = _PLACEMOE_RUNTIME_CONFIG.initial_artifact
_CPU_PLANNER_MODE = os.environ.get("VEOMNI_HIERMOE_CPU_PLANNER_MODE", "off").strip().lower()
_CPU_TRAIN_CORES_PER_RANK = _env_int("VEOMNI_HIERMOE_CPU_TRAIN_CORES_PER_RANK", 8)
_NPU_LAYER_OWNER_BLOCKING = _env_flag("VEOMNI_HIERMOE_NPU_LAYER_OWNER_BLOCKING")
_NPU_LAYER_OWNER_COLLECTIVE = (
    os.environ.get(
        "VEOMNI_HIERMOE_NPU_LAYER_OWNER_COLLECTIVE",
        "reduce_scatter",
    )
    .strip()
    .lower()
)
_HOT_UPDATE = _PLACEMOE_RUNTIME_CONFIG.hot_update.enabled
_HOT_UPDATE_WORK_ROOT = _PLACEMOE_RUNTIME_CONFIG.hot_update.work_root
_HOT_UPDATE_BUILDER = _PLACEMOE_RUNTIME_CONFIG.hot_update.planner_path
_HOT_UPDATE_RESOURCES = _PLACEMOE_RUNTIME_CONFIG.resources
_HOT_UPDATE_LAST_STEP = _PLACEMOE_RUNTIME_CONFIG.hot_update.last_update_step
_HOT_UPDATE_LAYOUT_INTERVAL = _PLACEMOE_RUNTIME_CONFIG.hot_update.layout_interval_steps
_HOT_UPDATE_MAPPING_INTERVAL = _PLACEMOE_RUNTIME_CONFIG.hot_update.mapping_interval_steps
_HOT_UPDATE_INTER_MS_PER_BYTE = _PLACEMOE_RUNTIME_CONFIG.calibration.inter_ms_per_byte
_HOT_UPDATE_INTRA_MS_PER_BYTE = _PLACEMOE_RUNTIME_CONFIG.calibration.intra_ms_per_byte
_HOT_UPDATE_ROUTE_MS_PER_ASSIGNMENT = _PLACEMOE_RUNTIME_CONFIG.calibration.route_ms_per_assignment
_HOT_UPDATE_COMMUNICATION_MULTIPLIER = _PLACEMOE_RUNTIME_CONFIG.calibration.communication_multiplier
_HOT_UPDATE_COMPUTE_MS_PER_ASSIGNMENT = _PLACEMOE_RUNTIME_CONFIG.calibration.compute_ms_per_assignment
_HOT_UPDATE_COMPUTE_MULTIPLIER = _PLACEMOE_RUNTIME_CONFIG.calibration.compute_multiplier


def configure_placemoe_runtime(config: PlaceMoERuntimeConfig) -> None:
    """Install the canonical runtime config before creating a manager.

    Environment variables remain a compatibility input at import time, but
    production training passes the nested VeOmni configuration explicitly.
    """

    global _PLACEMOE_RUNTIME_CONFIG
    global _INITIAL_LAYOUT_PATH, _ABLATION_REPLAY_PATH
    global _HOT_UPDATE, _HOT_UPDATE_WORK_ROOT, _HOT_UPDATE_BUILDER, _HOT_UPDATE_RESOURCES
    global _HOT_UPDATE_LAST_STEP, _HOT_UPDATE_LAYOUT_INTERVAL, _HOT_UPDATE_MAPPING_INTERVAL
    global _HOT_UPDATE_INTER_MS_PER_BYTE, _HOT_UPDATE_INTRA_MS_PER_BYTE
    global _HOT_UPDATE_ROUTE_MS_PER_ASSIGNMENT, _HOT_UPDATE_COMMUNICATION_MULTIPLIER
    global _HOT_UPDATE_COMPUTE_MS_PER_ASSIGNMENT, _HOT_UPDATE_COMPUTE_MULTIPLIER

    config.validate()
    _PLACEMOE_RUNTIME_CONFIG = config
    _INITIAL_LAYOUT_PATH = config.initial_artifact
    if config.initial_artifact:
        _ABLATION_REPLAY_PATH = config.initial_artifact
    _HOT_UPDATE = config.hot_update.enabled
    _HOT_UPDATE_WORK_ROOT = config.hot_update.work_root
    _HOT_UPDATE_BUILDER = config.hot_update.planner_path
    _HOT_UPDATE_RESOURCES = config.resources
    _HOT_UPDATE_LAST_STEP = config.hot_update.last_update_step
    _HOT_UPDATE_LAYOUT_INTERVAL = config.hot_update.layout_interval_steps
    _HOT_UPDATE_MAPPING_INTERVAL = config.hot_update.mapping_interval_steps
    _HOT_UPDATE_INTER_MS_PER_BYTE = config.calibration.inter_ms_per_byte
    _HOT_UPDATE_INTRA_MS_PER_BYTE = config.calibration.intra_ms_per_byte
    _HOT_UPDATE_ROUTE_MS_PER_ASSIGNMENT = config.calibration.route_ms_per_assignment
    _HOT_UPDATE_COMMUNICATION_MULTIPLIER = config.calibration.communication_multiplier
    _HOT_UPDATE_COMPUTE_MS_PER_ASSIGNMENT = config.calibration.compute_ms_per_assignment
    _HOT_UPDATE_COMPUTE_MULTIPLIER = config.calibration.compute_multiplier


_ONLINE_FREEZE_COST_MODE = os.environ.get("VEOMNI_HIERMOE_ONLINE_FREEZE_COST_MODE", "off").strip().lower()
_ONLINE_FREEZE_CALIBRATION_STEP = _env_int(
    "VEOMNI_HIERMOE_ONLINE_FREEZE_CALIBRATION_STEP",
    1,
    minimum=0,
)
_ONLINE_FREEZE_COMMUNICATION_RATIO = _env_float(
    "VEOMNI_HIERMOE_ONLINE_FREEZE_COMMUNICATION_RATIO",
    3.1,
)
_ONLINE_FREEZE_COMPUTE_RATIO = _env_float(
    "VEOMNI_HIERMOE_ONLINE_FREEZE_COMPUTE_RATIO",
    4.19,
)
_ONLINE_FREEZE_INTER_MS_PER_BYTE = _env_float(
    "VEOMNI_HIERMOE_ONLINE_FREEZE_INTER_MS_PER_BYTE",
    6.765449326279194e-08,
)
_ONLINE_FREEZE_INTRA_MS_PER_BYTE = _env_float(
    "VEOMNI_HIERMOE_ONLINE_FREEZE_INTRA_MS_PER_BYTE",
    5.02482606728045e-09,
)
_ONLINE_FREEZE_ROUTE_MS_PER_ASSIGNMENT = _env_float(
    "VEOMNI_HIERMOE_ONLINE_FREEZE_ROUTE_MS_PER_ASSIGNMENT",
    8.746548178958447e-05,
)
_ONLINE_FREEZE_TRAFFIC_INTERCEPT_MS = _env_float(
    "VEOMNI_HIERMOE_ONLINE_FREEZE_TRAFFIC_INTERCEPT_MS",
    16.771503695343263,
)
_COST_MODEL_VERIFY = _env_flag("VEOMNI_HIERMOE_COST_MODEL_VERIFY")
_EXPORT_COST_MODEL_SAMPLES = _env_flag("VEOMNI_HIERMOE_EXPORT_COST_MODEL_SAMPLES")
_COST_MODEL_VALIDATION_STEPS = _env_int(
    "VEOMNI_HIERMOE_COST_MODEL_VALIDATION_STEPS",
    1,
    minimum=1,
)
_ONLINE_LUT_UPDATE = _env_flag("VEOMNI_HIERMOE_ONLINE_LUT_UPDATE")
_ONLINE_LUT_START_STEP = _env_int(
    "VEOMNI_HIERMOE_ONLINE_LUT_START_STEP",
    1,
    minimum=0,
)
_ONLINE_LUT_MIN_GAIN = _env_float(
    "VEOMNI_HIERMOE_ONLINE_LUT_MIN_GAIN",
    0.0,
)
_FORWARD_REUSE_COVER = _env_flag("VEOMNI_HIERMOE_FORWARD_REUSE_COVER")
_FORWARD_REUSE_COVER_COMPUTE_WEIGHT = _env_float(
    "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_COMPUTE_WEIGHT",
    1.0,
)
_FORWARD_REUSE_COVER_COMPUTE_MS_PER_ASSIGNMENT = _env_float(
    "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_COMPUTE_MS_PER_ASSIGNMENT",
    2.82807e-05,
)
_FORWARD_REUSE_COVER_MIN_GAIN = _env_float(
    "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_MIN_GAIN",
    0.0,
)
_FORWARD_REUSE_COVER_PATCH_REMAP = _env_flag("VEOMNI_HIERMOE_FORWARD_REUSE_COVER_PATCH_REMAP")
_FORWARD_REUSE_COVER_FAST = _env_flag("VEOMNI_HIERMOE_FORWARD_REUSE_COVER_FAST")
_FORWARD_REUSE_COVER_ROUNDS = _env_int(
    "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_ROUNDS",
    1,
    minimum=1,
)
_FORWARD_REUSE_COVER_ONLY_STEP = _env_int(
    "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_ONLY_STEP",
    -1,
    minimum=-1,
)
_FORWARD_REUSE_COVER_VICTIM_MODE = (
    os.environ.get("VEOMNI_HIERMOE_FORWARD_REUSE_COVER_VICTIM_MODE", "minimum").strip().lower()
)
_FORWARD_REUSE_COVER_SERVICE_SCOPE = (
    os.environ.get("VEOMNI_HIERMOE_FORWARD_REUSE_COVER_SERVICE_SCOPE", "rank").strip().lower()
)
_FORWARD_REUSE_COVER_CONFIRM_SAMPLES = _env_int(
    "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_CONFIRM_SAMPLES",
    1,
    minimum=1,
)
_FORWARD_REUSE_COVER_AGGREGATE_SERVICE_GROUP = _env_flag("VEOMNI_HIERMOE_FORWARD_REUSE_COVER_AGGREGATE_SERVICE_GROUP")
_FORWARD_REUSE_COVER_PROPOSAL_TOPK = _env_int(
    "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_PROPOSAL_TOPK",
    1,
    minimum=1,
)
_FORWARD_REUSE_COVER_EMPTY_SEEDING = _env_flag("VEOMNI_HIERMOE_FORWARD_REUSE_COVER_EMPTY_SEEDING")


def _full_timing_range(section: str):
    if not _env_flag("VEOMNI_FULL_PROFILE_ENABLE"):
        return nullcontext()
    try:
        from ....utils.full_timing_profiler import get_active_full_timing_profiler

        profiler = get_active_full_timing_profiler()
    except Exception:
        profiler = None
    if profiler is None:
        return nullcontext()
    return profiler.cuda_range(section)


def _placement_timing_range(prefix: str | None, phase: str):
    if prefix is None:
        return nullcontext()
    return _full_timing_range(f"{prefix}_{phase}")
