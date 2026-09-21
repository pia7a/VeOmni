"""PlaceMoE runtime lifecycle and model/optimizer binding."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Any

import torch
import torch.distributed as dist

from . import runtime_settings as settings
from .core_planner import CoReMoEPlanner
from .greedy_planner import GreedyCommunicationPlanner
from .perf_model import HierMoEPerfModel
from .placemoe.runtime import HotUpdateController, PlaceMoECalibration
from .planner import CurrentRoutePlanner, PlacementPlan
from .runtime_artifact import ArtifactMixin
from .runtime_calibration import CalibrationMixin
from .runtime_checkpoint import CheckpointMixin
from .runtime_gradients import GradientsMixin
from .runtime_hot_update import HotUpdateMixin
from .runtime_migration import MigrationMixin
from .runtime_pipeline import PipelineMixin
from .runtime_planning import PlanningMixin
from .runtime_routing import RoutingMixin

# Stable integration entry points; helper modules import their dependencies directly.
from .runtime_settings import _full_timing_range
from .runtime_settings import configure_placemoe_runtime as configure_placemoe_runtime
from .runtime_tensors import (
    _build_optimizer_param_bindings,
    _create_expert_swap_process_group,
    _local_tensor_view,
    _optimizer_state_slot_tensors,
    _optimizer_state_slot_tensors_from_bindings,
)
from .runtime_tensors import expand_redundant_expert_slots as expand_redundant_expert_slots
from .runtime_types import (
    ExpertLayerState,
    _OptimizerParamBinding,
    _PendingLayerSwap,
    _PendingPipelinePlan,
    _PipelineGradResult,
    _PipelineMigrationResult,
    _PipelinePlannerWindows,
    _PipelinePlanResult,
    _SwapStagingBuffer,
)
from .topology import Hierarchy


class ExpertSwapManager(
    ArtifactMixin,
    PlanningMixin,
    PipelineMixin,
    HotUpdateMixin,
    MigrationMixin,
    GradientsMixin,
    RoutingMixin,
    CalibrationMixin,
    CheckpointMixin,
):
    """PlaceMoE runtime lifecycle and model/optimizer binding."""

    def __init__(
        self,
        *,
        ep_group: dist.ProcessGroup | None,
        ep_size: int,
        ep_rank: int,
        expert_swap_interval: int,
        expert_swap_max_pairs_per_layer: int,
        redundant_slot_increment_per_device: int,
        max_replica_rounds: int,
        smooth_max_gamma: float,
        hierarchy: Hierarchy,
        perf_model: HierMoEPerfModel,
        expert_swap_mode: str = "step",
        expert_swap_selector: str = "current_joint",
        activation_checkpointing_enabled: bool = False,
        gradient_bytes_per_element: int = 4,
        configured_max_replica_rounds: int | None = None,
        replica_slot_capacity: int | None = None,
        planner_route_sample_size: int = 1024,
        fixed_pipeline_overlap: bool = False,
        greedy_max_copies_per_expert: int = 4,
        runtime_perf_model_path: str = "",
        debug_validate: bool = False,
    ) -> None:
        if expert_swap_selector == "legacy_batched" or settings._NPU_LAYER_OWNER_BLOCKING:
            raise ValueError("Historical batched/layer-owner selectors were removed; use PlaceMoE hot_update.")
        if settings._FORWARD_REUSE_COVER:
            raise ValueError("Forward-cover experiments were removed; use PlaceMoE hot_update.")
        if settings._ONLINE_LUT_UPDATE:
            raise ValueError("Online LUT experiments were removed; use PlaceMoE mapping updates.")
        if expert_swap_selector == "hiermoe_exact_p1":
            raise ValueError("Historical exact-pair search was removed; use PlaceMoE hot_update.")
        if settings._CPU_PLANNER_MODE != "off":
            raise ValueError("CPU planner experiments were removed; use PlaceMoE hot_update.")
        self.ep_group = ep_group
        self.ep_size = int(ep_size)
        self.ep_rank = int(ep_rank)
        self.expert_swap_interval = int(expert_swap_interval)
        self.expert_swap_max_pairs_per_layer = max(0, int(expert_swap_max_pairs_per_layer))
        self.redundant_slot_increment_per_device = max(0, int(redundant_slot_increment_per_device))
        self.max_replica_rounds = max(0, int(max_replica_rounds))
        self.configured_max_replica_rounds = (
            None if configured_max_replica_rounds is None else max(0, int(configured_max_replica_rounds))
        )
        self.replica_slot_capacity = (
            self.redundant_slot_increment_per_device * self.ep_size
            if replica_slot_capacity is None
            else max(0, int(replica_slot_capacity))
        )
        if planner_route_sample_size <= 0:
            raise ValueError("planner_route_sample_size must be positive.")
        self.planner_route_sample_size = int(planner_route_sample_size)
        if not 1 <= greedy_max_copies_per_expert <= 8:
            raise ValueError("greedy_max_copies_per_expert must be between 1 and 8.")
        self.greedy_max_copies_per_expert = int(greedy_max_copies_per_expert)
        self.smooth_max_gamma = float(smooth_max_gamma)
        self.hierarchy = hierarchy
        self.perf_model = perf_model
        self.expert_swap_mode = str(expert_swap_mode)
        self.expert_swap_selector = str(expert_swap_selector)
        auto_calibration = settings._PLACEMOE_RUNTIME_CONFIG.calibration.auto_generate
        cost_model_verify = bool(settings._COST_MODEL_VERIFY or auto_calibration)
        if self.expert_swap_selector not in {
            "current_joint",
            "hiermoe_greedy_cover_p1",
        }:
            raise ValueError("expert_swap_selector must be current_joint or hiermoe_greedy_cover_p1.")
        if self.expert_swap_selector == "hiermoe_greedy_cover_p1":
            if self.expert_swap_max_pairs_per_layer > 1:
                raise ValueError("hiermoe_greedy_cover_p1 supports at most one steady-state swap per layer.")
            if self.redundant_slot_increment_per_device <= 0:
                raise ValueError("hiermoe_greedy_cover_p1 requires redundant expert slots.")
        self.fixed_pipeline_overlap = bool(fixed_pipeline_overlap)
        if self.fixed_pipeline_overlap and (
            self.expert_swap_mode != "step" or self.expert_swap_selector != "hiermoe_greedy_cover_p1"
        ):
            raise ValueError("fixed_pipeline_overlap requires step mode with the hiermoe_greedy_cover_p1 selector.")
        if settings._ABLATION_REPLAY_MODE not in {"off", "static", "step"}:
            raise ValueError(
                f"VEOMNI_HIERMOE_ABLATION_REPLAY_MODE must be off, static, or step, got {settings._ABLATION_REPLAY_MODE!r}."
            )
        if settings._ABLATION_MIGRATION_MODE not in {"hidden", "blocking"}:
            raise ValueError(
                f"VEOMNI_HIERMOE_ABLATION_MIGRATION_MODE must be hidden or blocking, got {settings._ABLATION_MIGRATION_MODE!r}."
            )
        if settings._ABLATION_GRAD_MODE not in {"hidden", "blocking"}:
            raise ValueError(
                f"VEOMNI_HIERMOE_ABLATION_GRAD_MODE must be hidden or blocking, got {settings._ABLATION_GRAD_MODE!r}."
            )
        if settings._NPU_LAYER_OWNER_COLLECTIVE not in {"reduce_scatter", "all_to_all"}:
            raise ValueError(
                "VEOMNI_HIERMOE_NPU_LAYER_OWNER_COLLECTIVE must be reduce_scatter or all_to_all, "
                f"got {settings._NPU_LAYER_OWNER_COLLECTIVE!r}."
            )
        if settings._ONLINE_FREEZE_COST_MODE not in {"off", "communication", "joint"}:
            raise ValueError(
                "VEOMNI_HIERMOE_ONLINE_FREEZE_COST_MODE must be off, communication, or joint, "
                f"got {settings._ONLINE_FREEZE_COST_MODE!r}."
            )
        if settings._ONLINE_FREEZE_COST_MODE != "off" and (
            not self.fixed_pipeline_overlap
            or self.expert_swap_mode != "step"
            or self.expert_swap_selector != "hiermoe_greedy_cover_p1"
            or not settings._FIXED_R2_LAYOUT
            or self.expert_swap_max_pairs_per_layer != 0
            or self.max_replica_rounds != self.replica_slot_capacity
        ):
            raise ValueError(
                "The online freeze experiment requires fixed R2, fixed-pipeline step mode, "
                "the hiermoe_greedy_cover_p1 selector, zero swaps, and one initialization "
                "round for every redundant slot."
            )
        if cost_model_verify and (
            self.expert_swap_mode != "step"
            or self.expert_swap_max_pairs_per_layer != 0
            or settings._ONLINE_FREEZE_COST_MODE != "off"
        ):
            raise ValueError(
                "Cost-model verification requires step mode, zero swaps, and all placement experiments disabled."
            )
        if settings._HOT_UPDATE and (
            self.expert_swap_mode != "step"
            or settings._ABLATION_REPLAY_MODE not in {"off", "static"}
            or settings._ONLINE_FREEZE_COST_MODE != "off"
            or (settings._COST_MODEL_VERIFY and not auto_calibration)
        ):
            raise ValueError("PlaceMoE hot updates require step mode and all other online planners disabled.")
        if settings._HOT_UPDATE and not settings._HOT_UPDATE_WORK_ROOT:
            raise ValueError("PlaceMoE hot-update work root must not be empty.")
        if settings._FORWARD_REUSE_COVER_EMPTY_SEEDING and settings._FIXED_R2_LAYOUT:
            raise ValueError("Empty-seeding Forward Cover requires VEOMNI_HIERMOE_FIXED_R2_LAYOUT=0.")
        if settings._FORWARD_REUSE_COVER_PATCH_REMAP and not settings._FORWARD_REUSE_COVER:
            raise ValueError("Forward-reuse patch remapping requires VEOMNI_HIERMOE_FORWARD_REUSE_COVER=1.")
        if settings._FORWARD_REUSE_COVER_FAST and not settings._FORWARD_REUSE_COVER_PATCH_REMAP:
            raise ValueError("Fast Forward-reuse Cover requires VEOMNI_HIERMOE_FORWARD_REUSE_COVER_PATCH_REMAP=1.")
        if settings._FORWARD_REUSE_COVER_FAST and settings._FORWARD_REUSE_COVER_CONFIRM_SAMPLES > 1:
            raise ValueError("Multi-sample Cover confirmation requires exact global validation.")
        if settings._FORWARD_REUSE_COVER_FAST and settings._FORWARD_REUSE_COVER_PROPOSAL_TOPK > 1:
            raise ValueError("Top-K Cover proposals require exact global validation.")
        if settings._FORWARD_REUSE_COVER_SERVICE_SCOPE not in {"rank", "node"}:
            raise ValueError(
                "VEOMNI_HIERMOE_FORWARD_REUSE_COVER_SERVICE_SCOPE must be rank or node, "
                f"got {settings._FORWARD_REUSE_COVER_SERVICE_SCOPE!r}."
            )
        if settings._ABLATION_REPLAY_MODE == "step" and not self.fixed_pipeline_overlap:
            raise ValueError("Step-by-step HierMoE ablation replay requires fixed_pipeline_overlap=true.")
        if settings._ABLATION_REPLAY_MODE != "off" and not settings._ABLATION_REPLAY_PATH:
            raise ValueError("VEOMNI_HIERMOE_ABLATION_REPLAY_PATH is required when ablation replay is enabled.")
        self._ablation_replay_mode = settings._ABLATION_REPLAY_MODE
        self._initial_layout_path = settings._INITIAL_LAYOUT_PATH
        self._ablation_migration_mode = settings._ABLATION_MIGRATION_MODE
        self._ablation_grad_mode = settings._ABLATION_GRAD_MODE
        self._online_freeze_cost_mode = settings._ONLINE_FREEZE_COST_MODE
        self._auto_calibration = bool(auto_calibration)
        self._auto_calibration_finalized = not self._auto_calibration
        self._auto_calibration_runtime_perf_model_path = str(runtime_perf_model_path)
        self._online_freeze_calibration_step = (
            int(settings._PLACEMOE_RUNTIME_CONFIG.calibration.warmup_steps)
            if self._auto_calibration
            else settings._ONLINE_FREEZE_CALIBRATION_STEP
        )
        self._online_freeze_communication_ratio = settings._ONLINE_FREEZE_COMMUNICATION_RATIO
        self._online_freeze_compute_ratio = settings._ONLINE_FREEZE_COMPUTE_RATIO
        self._online_freeze_inter_ms_per_byte = settings._ONLINE_FREEZE_INTER_MS_PER_BYTE
        self._online_freeze_intra_ms_per_byte = settings._ONLINE_FREEZE_INTRA_MS_PER_BYTE
        self._online_freeze_route_ms_per_assignment = settings._ONLINE_FREEZE_ROUTE_MS_PER_ASSIGNMENT
        self._online_freeze_traffic_intercept_ms = settings._ONLINE_FREEZE_TRAFFIC_INTERCEPT_MS
        self._cost_model_verify = cost_model_verify
        self._cost_model_validation_steps = (
            int(settings._PLACEMOE_RUNTIME_CONFIG.calibration.validation_steps)
            if self._auto_calibration
            else settings._COST_MODEL_VALIDATION_STEPS
        )
        self._export_cost_model_samples = bool(settings._EXPORT_COST_MODEL_SAMPLES or self._auto_calibration)
        self._cost_model_reports: dict[int, dict[str, Any]] = {}
        self._auto_calibration_compute_mape = 0.0
        self._auto_calibration_communication_mape = 0.0
        self._auto_calibration_joint_mape = 0.0
        self._cost_model_verify_coefficients: tuple[float, float, float, float] | None = None
        self._cost_model_verify_receive_only_coefficients: tuple[float, float] | None = None
        self._cost_model_verify_feature_coefficients: (
            dict[
                str,
                dict[str, tuple[tuple[float, ...], float]],
            ]
            | None
        ) = None
        self._cost_model_verify_complete = False
        self._hot_update = settings._HOT_UPDATE
        self._hot_update_layout_interval = int(settings._HOT_UPDATE_LAYOUT_INTERVAL)
        self._hot_update_mapping_interval = int(settings._HOT_UPDATE_MAPPING_INTERVAL)
        self._hot_update_controller = HotUpdateController(
            layout_interval_steps=self._hot_update_layout_interval,
            mapping_interval_steps=self._hot_update_mapping_interval,
            last_update_step=settings._HOT_UPDATE_LAST_STEP,
            failure_policy=settings._PLACEMOE_RUNTIME_CONFIG.hot_update.failure_policy,
        )
        self._hot_update_updates = 0
        self._hot_update_layout_updates = 0
        self._hot_update_mapping_updates = 0
        self._hot_update_last_source_step = -1
        self._hot_update_last_apply_step = -1
        self._hot_update_last_staleness_steps = -1
        self._hot_update_last_snapshot_ms = 0.0
        self._hot_update_last_planner_ms = 0.0
        self._hot_update_last_migration_ms = 0.0
        self._hot_update_last_moved_slots = 0
        if settings._FORWARD_REUSE_COVER_SERVICE_SCOPE == "rank":
            service_group_size = 1
        else:
            proper_group_sizes = [
                int(group_size)
                for group_size in self.hierarchy.group_sizes
                if 1 < int(group_size) < self.ep_size and self.ep_size % int(group_size) == 0
            ]
            if not proper_group_sizes:
                raise ValueError("Node-scoped Forward-reuse Cover requires a proper hierarchy group size.")
            service_group_size = min(proper_group_sizes)
        # Preserve historical metric keys without retaining retired planner state.
        self._retired_planner_metrics = {
            "hiermoe/cpu_planner_mode": settings._CPU_PLANNER_MODE,
            "hiermoe/online_lut_update": int(settings._ONLINE_LUT_UPDATE),
            "hiermoe/online_lut_start_step": settings._ONLINE_LUT_START_STEP,
            "hiermoe/online_lut_min_gain": settings._ONLINE_LUT_MIN_GAIN,
            "hiermoe/forward_reuse_cover": int(settings._FORWARD_REUSE_COVER),
            "hiermoe/forward_reuse_cover_patch_remap": int(settings._FORWARD_REUSE_COVER_PATCH_REMAP),
            "hiermoe/forward_reuse_cover_fast": int(settings._FORWARD_REUSE_COVER_FAST),
            "hiermoe/forward_reuse_cover_compute_weight": settings._FORWARD_REUSE_COVER_COMPUTE_WEIGHT,
            "hiermoe/forward_reuse_cover_compute_ms_per_assignment": settings._FORWARD_REUSE_COVER_COMPUTE_MS_PER_ASSIGNMENT,
            "hiermoe/forward_reuse_cover_min_gain": settings._FORWARD_REUSE_COVER_MIN_GAIN,
            "hiermoe/forward_reuse_cover_rounds": settings._FORWARD_REUSE_COVER_ROUNDS,
            "hiermoe/forward_reuse_cover_only_step": settings._FORWARD_REUSE_COVER_ONLY_STEP,
            "hiermoe/forward_reuse_cover_victim_mode": settings._FORWARD_REUSE_COVER_VICTIM_MODE,
            "hiermoe/forward_reuse_cover_service_scope": settings._FORWARD_REUSE_COVER_SERVICE_SCOPE,
            "hiermoe/forward_reuse_cover_service_group_size": service_group_size,
            "hiermoe/forward_reuse_cover_aggregate_service_group": int(
                settings._FORWARD_REUSE_COVER_AGGREGATE_SERVICE_GROUP
            ),
            "hiermoe/forward_reuse_cover_proposal_topk": settings._FORWARD_REUSE_COVER_PROPOSAL_TOPK,
            "hiermoe/forward_reuse_cover_empty_seeding": int(settings._FORWARD_REUSE_COVER_EMPTY_SEEDING),
            "hiermoe/forward_reuse_cover_confirm_samples": settings._FORWARD_REUSE_COVER_CONFIRM_SAMPLES,
            "hiermoe/forward_reuse_cover_pending": 0,
        }
        self._ablation_actions_by_step: dict[int, dict[str, tuple[tuple[str, str], ...]]] = {}
        self._ablation_expected_layouts: dict[str, tuple[int, ...]] = {}
        self._ablation_expected_owner_slots: dict[str, tuple[int, ...]] = {}
        self._ablation_expected_source_luts: dict[str, tuple[tuple[int, ...], ...]] = {}
        self._ablation_initial_layout = ""
        layout_metadata_path = self._initial_layout_path or (
            settings._ABLATION_REPLAY_PATH if self._ablation_replay_mode != "off" else ""
        )
        if layout_metadata_path:
            self._load_ablation_replay(layout_metadata_path)

        self.activation_checkpointing_enabled = bool(activation_checkpointing_enabled)
        self.gradient_overlap_enabled = bool(
            self._ablation_grad_mode == "hidden"
            and self.redundant_slot_increment_per_device > 0
            and ep_group is not None
        )
        self._swap_group = (
            _create_expert_swap_process_group(ep_group, self.ep_size)
            if self.expert_swap_max_pairs_per_layer > 0 and not self.fixed_pipeline_overlap
            else None
        )
        # Planner and migration collectives are serialized and can reuse the
        # training EP group. Hidden replica-gradient P2P is launched on a
        # separate accelerator stream while backward/FSDP collectives are active.
        # Arbitrary partial-capacity layouts require multiple peer waves, so
        # sharing the training group can violate cross-stream HCCL ordering.
        # Give only that path a dedicated group.
        self._pipeline_background_group = ep_group if self.fixed_pipeline_overlap else None
        self._pipeline_planner_group = self._pipeline_background_group
        self._pipeline_migration_group = self._pipeline_background_group
        self._owns_pipeline_grad_group = self.gradient_overlap_enabled
        self._pipeline_grad_group = (
            _create_expert_swap_process_group(
                ep_group,
                self.ep_size,
                group_desc="hiermoe_pipeline_grad",
            )
            if self._owns_pipeline_grad_group
            else self._pipeline_background_group
        )
        self.gradient_bytes_per_element = max(1, int(gradient_bytes_per_element))
        self.debug_validate = bool(debug_validate)
        self.layers: dict[str, ExpertLayerState] = {}
        self.module_id_to_key: dict[int, str] = {}
        self.param_id_to_key: dict[int, str] = {}
        self.optimizer: Any = None
        self._optimizer_param_bindings: dict[int, tuple[_OptimizerParamBinding, ...]] = {}
        # Replica-gradient waves are executed layer by layer and synchronously
        # waited. Reuse one manager-wide staging pool instead of retaining a
        # send/receive pair for every layer.
        self._replica_grad_buffers: dict[tuple[str, int, str, str], torch.Tensor] = {}
        self._swap_staging_buffers: dict[tuple[torch.device, torch.dtype], _SwapStagingBuffer] = {}
        self._swap_comm_streams: dict[torch.device, Any] = {}
        self._pending_layer_swaps: dict[str, _PendingLayerSwap] = {}
        self._exact_candidate_pair_cache: dict[tuple[int, torch.device], torch.Tensor] = {}
        self.latest_pair: str = "none"
        self._pending_state: dict[str, Any] | None = None
        self._placement_metrics: dict[str, float | int | str] = {}
        self._metrics_step = -1

        self._pipeline_lock = Lock()
        self._pipeline_grad_submit_lock = Lock()
        self._pipeline_plan_worker_capacity = max(1, settings._PIPELINE_PLAN_WORKERS)
        self._pipeline_plan_executor = (
            ThreadPoolExecutor(
                max_workers=self._pipeline_plan_worker_capacity,
                thread_name_prefix="hiermoe-plan",
            )
            if self.fixed_pipeline_overlap
            else None
        )
        self._pipeline_collective_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="hiermoe-collective")
            if self.fixed_pipeline_overlap
            else None
        )
        self._pipeline_migration_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="hiermoe-move")
            if self.fixed_pipeline_overlap
            else None
        )
        # NCCL host launches stay on the autograd thread; GPU work overlaps on its dedicated stream.
        self._pipeline_grad_executor = None
        self._pipeline_streams: dict[tuple[str, torch.device], Any] = {}
        self._pipeline_plan_futures: dict[str, Future[_PipelinePlanResult]] = {}
        self._pipeline_planner_windows: dict[str, _PipelinePlannerWindows] = {}
        self._pipeline_planner_dispatch_events: dict[str, Any] = {}
        self._pipeline_planner_compute_events: dict[str, Any] = {}
        self._pipeline_pending_plans: dict[str, _PendingPipelinePlan] = {}
        self._pipeline_migration_futures: dict[str, Future[_PipelineMigrationResult]] = {}
        self._pipeline_grad_futures: dict[str, Future[_PipelineGradResult]] = {}
        self._pipeline_grad_ready: dict[str, set[int]] = defaultdict(set)
        self._pipeline_grad_ready_events: dict[str, dict[int, Any]] = defaultdict(dict)
        self._pipeline_grad_dispatch_complete: set[str] = set()
        self._pipeline_grad_window_waited: set[str] = set()
        self._pipeline_grad_comm_blocked = False
        self._pipeline_grad_window_exposed_ms = 0.0
        self._pipeline_grad_hook_handles: list[Any] = []
        self._pipeline_grad_hook_params: set[int] = set()
        self._cpu_training_affinity: tuple[int, ...] = ()
        self._cpu_planner_affinity: tuple[int, ...] = ()
        self._hot_update_resources = settings._HOT_UPDATE_RESOURCES
        self._hot_update_calibration = PlaceMoECalibration(
            inter_ms_per_byte=settings._HOT_UPDATE_INTER_MS_PER_BYTE,
            intra_ms_per_byte=settings._HOT_UPDATE_INTRA_MS_PER_BYTE,
            route_ms_per_assignment=settings._HOT_UPDATE_ROUTE_MS_PER_ASSIGNMENT,
            communication_multiplier=settings._HOT_UPDATE_COMMUNICATION_MULTIPLIER,
            compute_ms_per_assignment=settings._HOT_UPDATE_COMPUTE_MS_PER_ASSIGNMENT,
            compute_multiplier=settings._HOT_UPDATE_COMPUTE_MULTIPLIER,
        )
        self._hot_update_affinity_automatic = False
        self._hot_update_planner_physical_cores = 0
        self._pipeline_step = -1
        self._pipeline_micro_step = 0
        self._pipeline_num_micro_steps = 1
        self._pipeline_next_migration_index = 0
        self._pipeline_next_grad_index = 0
        self._pipeline_layer_order: tuple[str, ...] = ()
        self._pipeline_shutdown = False

    def layer_calibration_enabled(self) -> bool:
        if self._cost_model_verify:
            return not self._cost_model_verify_complete
        if self.expert_swap_selector == "current_joint":
            return True
        if self.expert_swap_selector == "hiermoe_greedy_cover_p1":
            return any(layer.planner_calibration is None for layer in self.layers.values())
        return False

    def placement_metrics(self) -> dict[str, float | int | str]:
        return dict(self._placement_metrics)

    def _begin_metrics_step(self, step: int) -> None:
        if self._metrics_step == int(step):
            return
        self._metrics_step = int(step)
        self._placement_metrics = {
            **self._retired_planner_metrics,
            "hiermoe/placement_replica_rounds_configured": (
                "auto" if self.configured_max_replica_rounds is None else self.configured_max_replica_rounds
            ),
            "hiermoe/placement_replica_slot_capacity": self.replica_slot_capacity,
            "hiermoe/placement_replica_rounds_effective": self.max_replica_rounds,
            "hiermoe/placement_route_sample_size": self.planner_route_sample_size,
            "hiermoe/placement_runtime_cost_model": self.perf_model.runtime_cost_status,
            "hiermoe/cost_model_verify": int(self._cost_model_verify),
            "hiermoe/expert_swap_selector": self.expert_swap_selector,
            "hiermoe/fixed_pipeline_overlap": int(self.fixed_pipeline_overlap),
            "hiermoe/gradient_overlap_enabled": int(self.gradient_overlap_enabled),
            "hiermoe/cpu_training_affinity_cores": len(self._cpu_training_affinity),
            "hiermoe/cpu_planner_affinity_cores": len(self._cpu_planner_affinity),
            "hiermoe/ablation_replay_mode": self._ablation_replay_mode,
            "hiermoe/ablation_migration_mode": self._ablation_migration_mode,
            "hiermoe/ablation_grad_mode": self._ablation_grad_mode,
            "hiermoe/online_freeze_cost_mode": self._online_freeze_cost_mode,
            "hiermoe/online_freeze_inter_ms_per_byte": self._online_freeze_inter_ms_per_byte,
            "hiermoe/online_freeze_intra_ms_per_byte": self._online_freeze_intra_ms_per_byte,
            "hiermoe/online_freeze_route_ms_per_assignment": self._online_freeze_route_ms_per_assignment,
            "hiermoe/online_freeze_traffic_intercept_ms": self._online_freeze_traffic_intercept_ms,
            "hiermoe/fixed_r2_mirrored_remap": int(settings._FORCE_FIXED_R2_MIRRORED_REMAP),
            "placemoe/hot_update_enabled": int(self._hot_update),
            "placemoe/calibration_compute_mape_percent": self._auto_calibration_compute_mape,
            "placemoe/calibration_communication_mape_percent": self._auto_calibration_communication_mape,
            "placemoe/calibration_joint_mape_percent": self._auto_calibration_joint_mape,
            "placemoe/cpu_affinity_automatic": int(self._hot_update_affinity_automatic),
            "placemoe/planner_physical_cores": self._hot_update_planner_physical_cores,
            "placemoe/planner_workers": self._hot_update_resources.workers,
            "placemoe/planner_candidate_workers": self._hot_update_resources.candidate_workers,
            "placemoe/planner_worker_threads": self._hot_update_resources.worker_threads,
            "placemoe/layout_interval_steps": self._hot_update_layout_interval,
            "placemoe/mapping_interval_steps": self._hot_update_mapping_interval,
            "placemoe/hot_update_running": int(self._hot_update_controller.active_job is not None),
            "placemoe/layout_updates": self._hot_update_layout_updates,
            "placemoe/mapping_updates": self._hot_update_mapping_updates,
            "placemoe/last_source_step": self._hot_update_last_source_step,
            "placemoe/last_apply_step": self._hot_update_last_apply_step,
            "placemoe/last_staleness_steps": self._hot_update_last_staleness_steps,
            "placemoe/last_snapshot_ms": self._hot_update_last_snapshot_ms,
            "placemoe/last_planner_ms": self._hot_update_last_planner_ms,
            "placemoe/last_migration_ms": self._hot_update_last_migration_ms,
            "placemoe/last_moved_slots": self._hot_update_last_moved_slots,
            "hiermoe/pipeline_planner_backend": (
                self._planner_collective_backend(self._pipeline_planner_group)
                if self._pipeline_planner_group is not None
                else "none"
            ),
        }

    def _accumulate_metric(self, key: str, value: float | int | str) -> None:
        if isinstance(value, str):
            self._placement_metrics[key] = value
        elif isinstance(value, int):
            self._placement_metrics[key] = int(self._placement_metrics.get(key, 0)) + value
        else:
            self._placement_metrics[key] = float(self._placement_metrics.get(key, 0.0)) + float(value)

    def bind_optimizer(self, optimizer: Any) -> None:
        self.optimizer = optimizer
        self._optimizer_param_bindings = _build_optimizer_param_bindings(optimizer)

    def _planner_collective_backend(self, process_group: dist.ProcessGroup | None = None) -> str | None:
        process_group = self.ep_group if process_group is None else process_group
        if process_group is None or self.ep_size <= 1:
            return None
        return str(dist.get_backend(process_group)).lower().rsplit(".", maxsplit=1)[-1]

    def _planner_reduce_sum(
        self,
        tensor: torch.Tensor,
        process_group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        process_group = self.ep_group if process_group is None else process_group
        if process_group is not None and self.ep_size > 1:
            backend = self._planner_collective_backend(process_group)
            if backend == "gloo" and tensor.device.type != "cpu":
                reduced = tensor.detach().to(device="cpu")
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=process_group)
                tensor.copy_(reduced.to(device=tensor.device))
            else:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=process_group)
        return tensor

    def _planner_gather_fixed(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.ep_group is None or self.ep_size <= 1:
            return tensor.unsqueeze(0)
        backend = self._planner_collective_backend()
        local = tensor.contiguous()
        if backend == "gloo" and local.device.type != "cpu":
            local = local.to(device="cpu")
        gathered = torch.empty(
            (self.ep_size * local.numel(),),
            dtype=local.dtype,
            device=local.device,
        )
        dist.all_gather_into_tensor(gathered, local, group=self.ep_group)
        result = gathered.view(self.ep_size, local.numel())
        return result if result.device == tensor.device else result.to(device=tensor.device)

    def _expert_payload_bytes(self, layer: ExpertLayerState) -> tuple[tuple[int, ...], tuple[int, ...]]:
        state_bytes = 0
        gradient_bytes = 0
        for param in layer.expert_parameters:
            local = _local_tensor_view(param)
            slot_numel = int(local[0].numel())
            state_bytes += slot_numel * int(local.element_size())
            gradient_bytes += slot_numel * self.gradient_bytes_per_element
            state_tensors = _optimizer_state_slot_tensors_from_bindings(
                self._optimizer_param_bindings.get(id(param), ()), param
            )
            if not state_tensors:
                state_tensors = _optimizer_state_slot_tensors(self.optimizer, param)
            for tensor in state_tensors:
                local_state = _local_tensor_view(tensor)
                state_bytes += int(local_state[0].numel()) * int(local_state.element_size())
        return (
            (state_bytes,) * layer.num_experts,
            (gradient_bytes,) * layer.num_experts,
        )

    def _planner_for_layer(
        self,
        layer: ExpertLayerState,
        *,
        communication_scale: float,
        forward_compute_per_assignment: float,
        forward_compute_constant: float = 0.0,
        process_group: dist.ProcessGroup | None = None,
    ) -> CurrentRoutePlanner | GreedyCommunicationPlanner:
        planner_group = self.ep_group if process_group is None else process_group
        if self.expert_swap_selector == "hiermoe_greedy_cover_p1":
            return GreedyCommunicationPlanner(
                hierarchy=self.hierarchy,
                perf_model=self.perf_model,
                hidden_size=layer.latest_hidden_size,
                bytes_per_element=layer.latest_bytes_per_element,
                slots_per_rank=layer.num_local_experts,
                communication_scale=communication_scale,
                forward_compute_per_assignment=forward_compute_per_assignment,
                forward_compute_constant=forward_compute_constant,
                smooth_max_gamma=self.smooth_max_gamma,
                reducer=lambda tensor: self._planner_reduce_sum(tensor, planner_group),
                candidate_chunk_size=settings._SWAP_COST_CHUNK_CANDIDATES,
                process_group=planner_group,
                max_copies=self.greedy_max_copies_per_expert,
                assume_unique_routes=True,
                layer_parallel_streams=settings._GREEDY_LAYER_PARALLEL_STREAMS,
                adaptive_topk=(
                    settings._GREEDY_ADAPTIVE_TOPK
                    and not self.fixed_pipeline_overlap
                    and settings._GREEDY_EXACT_PRIMITIVE_TOPK == 0
                ),
                adaptive_topk_initial=settings._GREEDY_ADAPTIVE_TOPK_INITIAL,
                adaptive_topk_strict_certificate=settings._GREEDY_ADAPTIVE_TOPK_STRICT,
                exact_primitive_topk=(0 if self.fixed_pipeline_overlap else settings._GREEDY_EXACT_PRIMITIVE_TOPK),
                post_shortlist_compact_pair=(
                    not self.fixed_pipeline_overlap and settings._GREEDY_POST_SHORTLIST_COMPACT_PAIR
                ),
                exact_primitive_max_only=(
                    not self.fixed_pipeline_overlap and settings._GREEDY_EXACT_PRIMITIVE_MAX_ONLY
                ),
                traffic_inter_ms_per_byte=(
                    self._online_freeze_inter_ms_per_byte if self._online_freeze_cost_mode != "off" else None
                ),
                traffic_intra_ms_per_byte=(
                    self._online_freeze_intra_ms_per_byte if self._online_freeze_cost_mode != "off" else None
                ),
                traffic_route_ms_per_assignment=(
                    self._online_freeze_route_ms_per_assignment if self._online_freeze_cost_mode == "joint" else 0.0
                ),
                traffic_communication_phase_multiplier=(
                    self._online_freeze_communication_ratio if self._online_freeze_cost_mode != "off" else 1.0
                ),
                traffic_compute_phase_multiplier=(
                    self._online_freeze_compute_ratio if self._online_freeze_cost_mode == "joint" else 1.0
                ),
            )
        if self.expert_swap_mode != "layer":
            return CurrentRoutePlanner(
                hierarchy=self.hierarchy,
                perf_model=self.perf_model,
                hidden_size=layer.latest_hidden_size,
                bytes_per_element=layer.latest_bytes_per_element,
                slots_per_rank=layer.num_local_experts,
                communication_scale=communication_scale,
                forward_compute_per_assignment=forward_compute_per_assignment,
                reducer=self._planner_reduce_sum,
                candidate_chunk_size=settings._SWAP_COST_CHUNK_CANDIDATES,
            )
        expert_state_bytes, expert_gradient_bytes = self._expert_payload_bytes(layer)
        return CoReMoEPlanner(
            hierarchy=self.hierarchy,
            perf_model=self.perf_model,
            hidden_size=layer.latest_hidden_size,
            bytes_per_element=layer.latest_bytes_per_element,
            slots_per_rank=layer.num_local_experts,
            communication_scale=communication_scale,
            forward_compute_per_assignment=forward_compute_per_assignment,
            reducer=self._planner_reduce_sum,
            gather_fixed=self._planner_gather_fixed,
            collective_backend=self._planner_collective_backend(),
            route_sample_size=self.planner_route_sample_size,
            expert_state_bytes=expert_state_bytes,
            expert_gradient_bytes=expert_gradient_bytes,
        )

    def _record_plan_metrics(self, plan: PlacementPlan) -> None:
        values: dict[str, float | int] = {
            "hiermoe/placement_planning_ms": plan.planning_ms,
            "hiermoe/placement_route_stats_ms": plan.route_stats_ms,
            "hiermoe/placement_swap_ms": plan.swap_ms,
            "hiermoe/placement_replica_ms": plan.replica_ms,
            "hiermoe/placement_swap_score_ms": plan.swap_score_ms,
            "hiermoe/placement_swap_update_ms": plan.swap_update_ms,
            "hiermoe/placement_swap_collective_ms": plan.swap_collective_ms,
            "hiermoe/placement_replica_score_ms": plan.replica_score_ms,
            "hiermoe/placement_replica_update_ms": plan.replica_update_ms,
            "hiermoe/placement_replica_collective_ms": plan.replica_collective_ms,
            "hiermoe/placement_decision_sync_ms": plan.decision_sync_ms,
            "hiermoe/placement_finalization_ms": plan.finalization_ms,
            "hiermoe/placement_swap_count": plan.swap_rounds,
            "hiermoe/placement_replica_count": plan.replica_rounds,
            "hiermoe/placement_predicted_communication_ms": plan.final_cost.communication,
            "hiermoe/placement_predicted_compute_ms": plan.final_cost.compute,
            "hiermoe/placement_predicted_state_move_ms": plan.final_cost.state_move_exposed,
            "hiermoe/placement_predicted_gradient_sync_ms": plan.final_cost.gradient_sync,
            "hiermoe/placement_predicted_total_ms": plan.final_cost.total,
            "hiermoe/placement_baseline_total_ms": plan.baseline_cost.total,
        }
        for key, value in values.items():
            self._accumulate_metric(key, value)

    @torch.no_grad()
    def maybe_swap(self, step: int) -> str:
        self._begin_metrics_step(step)
        if self._auto_calibration and not self._auto_calibration_finalized:
            if self._hot_update:
                self._hot_update_controller.observe_step(int(step) + 1)
            layers = [self.layers[layer_key] for layer_key in sorted(self.layers)]
            return self._run_cost_model_verification(layers, int(step))
        if self._hot_update:
            return self._run_hot_update_step(int(step))
        if self._ablation_replay_mode != "off":
            return self._queue_ablation_replay_step(int(step))
        if self._cost_model_verify:
            layers = [self.layers[layer_key] for layer_key in sorted(self.layers)]
            return self._run_cost_model_verification(layers, int(step))
        if self._online_freeze_cost_mode != "off":
            return self._run_online_freeze_step(int(step))
        if self.fixed_pipeline_overlap:
            if any(bool((self._layer_layout(layer) < 0).any().item()) for layer in self.layers.values()):
                layers = [self.layers[layer_key] for layer_key in self.layers]
                committed = self._plan_historical_layers(layers, int(step))
                self.latest_pair = ",".join(committed) if committed else "none"
                return self.latest_pair
            return self._collect_pipeline_plans(int(step))

        if (
            self.layers
            and self.expert_swap_max_pairs_per_layer == 0
            and self.max_replica_rounds == 0
            and all(layer.fixed_r2_layout for layer in self.layers.values())
        ):
            self.latest_pair = "none"
            return self.latest_pair

        self.prepare_calibrations(step)
        minimum_step = 0 if self.expert_swap_selector == "hiermoe_greedy_cover_p1" else 1
        if int(step) < minimum_step or self.expert_swap_interval <= 0 or int(step) % self.expert_swap_interval != 0:
            self.latest_pair = "none"
            return self.latest_pair
        committed = []
        with _full_timing_range("hiermoe_current_route_plan"):
            layers = [self.layers[layer_key] for layer_key in sorted(self.layers)]
            if self.expert_swap_selector == "hiermoe_greedy_cover_p1" and self.expert_swap_mode == "step":
                committed.extend(self._plan_historical_layers(layers, int(step)))
            else:
                for layer in layers:
                    committed.extend(self._plan_current_layer(layer, int(step)))
        self.latest_pair = ",".join(committed) if committed else "none"
        return self.latest_pair

    @torch.no_grad()
    def maybe_swap_layer_on_routing(
        self,
        *,
        layer_key: str,
        selected_experts: torch.Tensor,
        hidden_size: int,
        bytes_per_element: int,
        step: int,
    ) -> str:
        self.record_routing(
            layer_key=layer_key,
            selected_experts=selected_experts,
            hidden_size=hidden_size,
            bytes_per_element=bytes_per_element,
            step=step,
        )
        self._begin_metrics_step(step)
        minimum_step = 0 if self.expert_swap_selector == "hiermoe_greedy_cover_p1" else 1
        if int(step) < minimum_step or self.expert_swap_interval <= 0 or int(step) % self.expert_swap_interval != 0:
            self.latest_pair = "none"
            return self.latest_pair

        layer = self.layers.get(layer_key)
        if layer is None:
            return self.latest_pair
        if layer.last_planned_step == int(step):
            return self.latest_pair
        layer.last_planned_step = int(step)
        with _full_timing_range("hiermoe_current_route_layer_plan"):
            committed = self._plan_current_layer(layer, int(step))
        self.latest_pair = ",".join(committed) if committed else "none"
        return self.latest_pair
