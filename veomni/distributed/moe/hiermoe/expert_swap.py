"""PlaceMoE runtime lifecycle and model/optimizer binding."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Future
from threading import Lock
from typing import Any

import torch
import torch.distributed as dist

from . import runtime_settings as settings
from .perf_model import HierMoEPerfModel
from .placemoe.runtime import HotUpdateController, PlaceMoECalibration
from .runtime_artifact import ArtifactMixin
from .runtime_calibration import CalibrationMixin
from .runtime_checkpoint import CheckpointMixin
from .runtime_gradients import GradientsMixin
from .runtime_hot_update import HotUpdateMixin
from .runtime_pipeline import PipelineMixin
from .runtime_routing import RoutingMixin

# Stable integration entry points; helper modules import their dependencies directly.
from .runtime_settings import configure_placemoe_runtime as configure_placemoe_runtime
from .runtime_tensors import (
    _build_optimizer_param_bindings,
    _create_expert_swap_process_group,
)
from .runtime_tensors import expand_redundant_expert_slots as expand_redundant_expert_slots
from .runtime_types import (
    ExpertLayerState,
    _OptimizerParamBinding,
    _PipelineGradResult,
    _SwapStagingBuffer,
)
from .topology import Hierarchy


class ExpertSwapManager(
    ArtifactMixin,
    PipelineMixin,
    HotUpdateMixin,
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
        settings.validate_production_environment()
        if expert_swap_selector not in {"current_joint", "hiermoe_greedy_cover_p1"}:
            raise ValueError("Historical selectors were removed; use PlaceMoE hot_update.")
        if expert_swap_mode != "step" or expert_swap_max_pairs_per_layer != 0 or max_replica_rounds != 0:
            raise ValueError(
                "Historical online search was removed; use step mode with zero swap/replica search budgets."
            )
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
        cost_model_verify = bool(auto_calibration or settings._CALIBRATION_ONLY)
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
        if cost_model_verify and self.expert_swap_mode != "step":
            raise ValueError("PlaceMoE automatic calibration requires step mode.")
        if settings._HOT_UPDATE and not settings._HOT_UPDATE_WORK_ROOT:
            raise ValueError("PlaceMoE hot-update work root must not be empty.")
        self._initial_layout_path = settings._INITIAL_LAYOUT_PATH
        self._auto_calibration = bool(auto_calibration)
        self._auto_calibration_finalized = not self._auto_calibration
        self._auto_calibration_runtime_perf_model_path = str(runtime_perf_model_path)
        self._calibration_warmup_steps = int(
            settings._CALIBRATION_STEP
            if settings._CALIBRATION_ONLY
            else settings._PLACEMOE_RUNTIME_CONFIG.calibration.warmup_steps
        )
        self._cost_model_verify = cost_model_verify
        self._cost_model_validation_steps = int(
            settings._CALIBRATION_VALIDATION_STEPS
            if settings._CALIBRATION_ONLY
            else settings._PLACEMOE_RUNTIME_CONFIG.calibration.validation_steps
        )
        self._export_cost_model_samples = cost_model_verify
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
        self._initial_plans = (
            self._load_initial_artifact(self._initial_layout_path) if self._initial_layout_path else {}
        )

        self.activation_checkpointing_enabled = bool(activation_checkpointing_enabled)
        self.gradient_overlap_enabled = bool(self.redundant_slot_increment_per_device > 0 and ep_group is not None)
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
        self.latest_pair: str = "none"
        self._pending_state: dict[str, Any] | None = None
        self._placement_metrics: dict[str, float | int | str] = {}
        self._metrics_step = -1

        self._pipeline_lock = Lock()
        self._pipeline_grad_submit_lock = Lock()
        # NCCL host launches stay on the autograd thread; GPU work overlaps on its dedicated stream.
        self._pipeline_grad_executor = None
        self._pipeline_streams: dict[tuple[str, torch.device], Any] = {}
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
        self._pipeline_next_grad_index = 0
        self._pipeline_layer_order: tuple[str, ...] = ()
        self._pipeline_shutdown = False

    def layer_calibration_enabled(self) -> bool:
        return self._cost_model_verify and not self._cost_model_verify_complete

    def placement_metrics(self) -> dict[str, float | int | str]:
        return dict(self._placement_metrics)

    def _begin_metrics_step(self, step: int) -> None:
        if self._metrics_step == int(step):
            return
        self._metrics_step = int(step)
        self._placement_metrics = {
            "hiermoe/placement_replica_rounds_configured": "auto"
            if self.configured_max_replica_rounds is None
            else self.configured_max_replica_rounds,
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
            "hiermoe/pipeline_planner_backend": self._planner_collective_backend(self._pipeline_planner_group)
            if self._pipeline_planner_group is not None
            else "none",
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
        if self._cost_model_verify:
            layers = [self.layers[key] for key in sorted(self.layers)]
            return self._run_cost_model_verification(layers, int(step))
        self.latest_pair = "none"
        return self.latest_pair
