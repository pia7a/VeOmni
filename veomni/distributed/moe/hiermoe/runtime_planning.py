"""Initialization planning and its existing collective/score pipeline."""

from __future__ import annotations

import time
import zlib
from collections import defaultdict
from concurrent.futures import Future
from contextlib import nullcontext
from threading import Event
from typing import Any, Sequence

import torch

from ....utils.device import get_torch_device
from . import runtime_settings as settings
from .greedy_planner import GreedyCommunicationPlanner
from .planner import PlacementPlan
from .runtime_settings import _full_timing_range
from .runtime_tensors import _local_tensor_view, _placement_group_all_true_mask
from .runtime_types import (
    ExpertLayerState,
    _PendingPipelinePlan,
    _PipelinePlannerStageTiming,
    _PipelinePlannerWindows,
    _PipelinePlanResult,
    _PipelinePrepareSubstageTiming,
)


class PlanningMixin:
    """Initialization planning and its existing collective/score pipeline."""

    def _submit_pipeline_plan(
        self,
        layer: ExpertLayerState,
        selected_experts: torch.Tensor,
        step: int,
    ) -> None:
        executor = self._pipeline_plan_executor
        if executor is None or self._pipeline_shutdown:
            return
        if self.expert_swap_max_pairs_per_layer <= 0 and self.max_replica_rounds <= 0:
            return
        if not self._pipeline_is_final_microstep():
            return
        if self.expert_swap_interval <= 0 or int(step) % self.expert_swap_interval != 0:
            return
        layout = self._layer_layout(layer)
        if bool((layout < 0).any().item()):
            return
        with self._pipeline_lock:
            if layer.key in self._pipeline_plan_futures or layer.last_planned_step == int(step):
                return
            layer.last_planned_step = int(step)
            windows = _PipelinePlannerWindows()
            self._pipeline_planner_windows[layer.key] = windows
        device = selected_experts.device
        ready_event = self._pipeline_ready_event(device)
        submitted_at = time.perf_counter()
        route = selected_experts.detach()
        owners = layer.logical_to_physical.detach().cpu().clone()
        placement_version = int(layer.placement_version)
        future = executor.submit(
            self._pipeline_plan_worker,
            layer.key,
            route,
            layout,
            owners,
            placement_version,
            int(step),
            ready_event,
            submitted_at,
        )
        with self._pipeline_lock:
            self._pipeline_plan_futures[layer.key] = future

    @torch.no_grad()
    def _pipeline_plan_worker(
        self,
        layer_key: str,
        selected_experts: torch.Tensor,
        layout: torch.Tensor,
        owners: torch.Tensor,
        placement_version: int,
        source_step: int,
        ready_event: Any | None,
        submitted_at: float,
    ) -> _PipelinePlanResult:
        layer = self.layers[layer_key]
        device = selected_experts.device
        started = time.perf_counter()
        timing = _PipelinePlannerStageTiming()
        prepare_timing = _PipelinePrepareSubstageTiming()
        prepare_stage_index = 0

        def prepare_checkpoint(stage: str) -> None:
            nonlocal prepare_stage_index
            try:
                stage_index = settings._PIPELINE_PREPARE_SUBSTAGES.index(stage)
            except ValueError as exc:
                raise RuntimeError(f"Unknown pipeline Prepare stage for {layer_key}: {stage}.") from exc
            if stage_index < prepare_stage_index:
                expected_stage = settings._PIPELINE_PREPARE_SUBSTAGES[prepare_stage_index]
                raise RuntimeError(
                    f"Pipeline Prepare stage order diverged for {layer_key}: expected {expected_stage}, got {stage}."
                )
            # Some exact fallback paths do not materialize the compact
            # statistical substages. Treat their missing checkpoints as empty
            # stages while preserving the same six-window cut boundaries.
            prepare_stage_index = stage_index
            if prepare_timing is not None:
                ended_at = time.perf_counter()
                ended_thread_at = time.thread_time()
                prepare_timing.checkpoint(
                    stage,
                    ended_at,
                    ended_thread_at,
                    self._pipeline_stage_event(),
                )
            prepare_stage_index += 1
            paused = self._complete_pipeline_prepare_stage(layer_key, prepare_stage_index)
            if paused and prepare_timing is not None:
                resumed = self._pipeline_stage_event()
                prepare_timing.begin(resumed, time.perf_counter(), time.thread_time())

        def run() -> PlacementPlan:
            self._wait_pipeline_prepare_start(layer_key)
            timing.prepare_start = self._pipeline_stage_event()
            if prepare_timing is not None:
                prepare_timing.begin(timing.prepare_start, time.perf_counter(), time.thread_time())
            planner = self._planner_for_layer(
                layer,
                communication_scale=1.0,
                forward_compute_per_assignment=0.0,
                forward_compute_constant=0.0,
                process_group=self._pipeline_planner_group,
            )
            if not isinstance(planner, GreedyCommunicationPlanner):
                raise RuntimeError("The fixed pipeline requires GreedyCommunicationPlanner.")
            prepare_checkpoint("planner_setup")
            planner.reducer = lambda tensor: self._pipeline_planner_reduce_sum(layer_key, tensor, device, timing)
            plan = planner.plan_layers(
                [selected_experts],
                [layout],
                [owners],
                source_ranks=self.ep_rank,
                max_swaps=self.expert_swap_max_pairs_per_layer,
                max_replicas=self.max_replica_rounds,
                layer_seeds=[zlib.crc32(layer_key.encode("utf-8"))],
                step=source_step,
                communication_scales=[1.0],
                forward_compute_per_assignment=[0.0],
                forward_compute_constant=[0.0],
                skip_final_route_update=True,
                prepare_stage_callback=prepare_checkpoint,
            )[0]
            timing.score_end = self._pipeline_stage_event()
            with self._pipeline_lock:
                windows = self._pipeline_planner_windows.get(layer_key)
                if windows is not None:
                    windows.score_done_event = timing.score_end
                    windows.score_done.set()
            return plan

        plan = self._run_pipeline_stream_task("planner", device, ready_event, run)
        prepare_ms, collective_ms, score_ms = timing.durations_ms()
        prepare_substage_device_ms, prepare_substage_host_ms, prepare_substage_thread_cpu_ms = (
            ({}, {}, {}) if prepare_timing is None else prepare_timing.durations_ms()
        )
        if prepare_substage_device_ms:
            prepare_ms = sum(prepare_substage_device_ms.values())
        finished = time.perf_counter()
        active_ms = prepare_ms + collective_ms + score_ms
        if active_ms <= 0.0:
            active_ms = (finished - started) * 1000.0
        return _PipelinePlanResult(
            layer_key=layer_key,
            source_step=source_step,
            placement_version=placement_version,
            plan=plan,
            raw_ms=active_ms,
            latency_ms=(finished - submitted_at) * 1000.0,
            prepare_device_ms=prepare_ms,
            collective_device_ms=collective_ms,
            score_device_ms=score_ms,
            prepare_substage_device_ms=prepare_substage_device_ms,
            prepare_substage_host_ms=prepare_substage_host_ms,
            prepare_substage_thread_cpu_ms=prepare_substage_thread_cpu_ms,
        )

    def _pipeline_planner_reduce_sum(
        self,
        layer_key: str,
        tensor: torch.Tensor,
        device: torch.device,
        timing: _PipelinePlannerStageTiming | None = None,
    ) -> torch.Tensor:
        """Hand the reduction to the single ordered collective launcher."""

        if timing is not None:
            timing.prepare_end = self._pipeline_stage_event()
        with self._pipeline_lock:
            windows = self._pipeline_planner_windows[layer_key]
            windows.collective_tensor = tensor
            windows.collective_device = device
            windows.collective_timing = timing
            windows.collective_tensor_ready.set()
        windows.collective_result_ready.wait()
        if windows.collective_error is not None:
            raise windows.collective_error
        stream = self._pipeline_stream("planner", device)
        windows.score_gate.wait()
        with self._pipeline_lock:
            compute_done = self._pipeline_planner_compute_events.get(layer_key)
        if stream is not None and compute_done is not None:
            stream.wait_event(compute_done)
        if timing is not None:
            timing.score_start = self._pipeline_stage_event()
        return tensor

    def _launch_pipeline_planner_collective(
        self,
        layer_key: str,
        windows: _PipelinePlannerWindows,
    ) -> None:
        """Launch one layer's HCCL reduction from the globally ordered thread."""

        self._wait_pipeline_host_event(layer_key, windows.collective_tensor_ready)
        with self._pipeline_lock:
            if self._pipeline_planner_windows.get(layer_key) is not windows:
                return
            tensor = windows.collective_tensor
            device = windows.collective_device
            timing = windows.collective_timing
            dispatch_done = self._pipeline_planner_dispatch_events.get(layer_key)
        if tensor is None or device is None:
            windows.collective_done.set()
            windows.collective_result_ready.set()
            return
        if device.type == "cpu":
            stream = None
            stream_context = nullcontext()
        else:
            device_api = get_torch_device()
            device_api.set_device(device)
            stream = self._pipeline_stream("planner", device)
            stream_context = device_api.stream(stream)
        try:
            with stream_context:
                if stream is not None and dispatch_done is not None:
                    stream.wait_event(dispatch_done)
                if timing is not None:
                    timing.collective_start = self._pipeline_stage_event()
                self._planner_reduce_sum(tensor, self._pipeline_planner_group)
                if timing is not None:
                    timing.collective_end = self._pipeline_stage_event()
            with self._pipeline_lock:
                if self._pipeline_planner_windows.get(layer_key) is windows:
                    windows.collective_done_event = None if timing is None else timing.collective_end
        except BaseException as error:
            windows.collective_error = error
            raise
        finally:
            windows.collective_done.set()
            windows.collective_result_ready.set()

    @staticmethod
    def _wait_pipeline_stage(event: Event, future: Future[Any]) -> None:
        while not event.wait(timeout=settings._PIPELINE_HOST_EVENT_POLL_SECONDS):
            if future.done():
                future.result()

    def open_pipeline_planner_collective_window(self, layer_key: str) -> None:
        """Run the planner collective after combine backward and during expert GEMM."""

        if not self.fixed_pipeline_overlap:
            return
        layer = self.layers.get(layer_key)
        combine_done = None if layer is None else self._pipeline_ready_event(self._pipeline_device(layer))
        with self._pipeline_lock:
            windows = self._pipeline_planner_windows.get(layer_key)
            if windows is not None:
                self._pipeline_planner_dispatch_events[layer_key] = combine_done
        if windows is not None:
            windows.collective_gate.set()
            executor = self._pipeline_collective_executor
            if executor is None:
                raise RuntimeError("HierMoE pipeline collective executor is unavailable.")
            with self._pipeline_lock:
                if windows.collective_future is None:
                    windows.collective_future = executor.submit(
                        self._launch_pipeline_planner_collective,
                        layer_key,
                        windows,
                    )

    def close_pipeline_planner_collective_window(self, layer_key: str) -> None:
        """Enforce collective completion before dispatch-backward uses the EP communicator."""

        if not self.fixed_pipeline_overlap:
            return
        with self._pipeline_lock:
            windows = self._pipeline_planner_windows.get(layer_key)
            future = self._pipeline_plan_futures.get(layer_key)
        if windows is None or future is None:
            return
        deadline_event = self._pipeline_stage_event()
        wait_started = time.perf_counter()
        self._wait_pipeline_stage(windows.collective_done, future)
        host_wait_ms = (time.perf_counter() - wait_started) * 1000.0
        with self._pipeline_lock:
            if self._pipeline_planner_windows.get(layer_key) is windows:
                windows.collective_deadline_event = deadline_event
                done_event = windows.collective_done_event
            else:
                done_event = None
        layer = self.layers.get(layer_key)
        if layer is not None and done_event is not None and done_event.event is not None:
            device = self._pipeline_device(layer)
            device_api = get_torch_device()
            try:
                current_stream = device_api.current_stream(device)
            except TypeError:
                current_stream = device_api.current_stream()
            current_stream.wait_event(done_event.event)
        self._accumulate_metric("hiermoe/pipeline_planner_collective_host_gate_wait_ms", host_wait_ms)
        if host_wait_ms > 0.01:
            self._accumulate_metric("hiermoe/pipeline_planner_collective_host_gate_miss", 1)

    def open_pipeline_planner_score_window(self, layer_key: str) -> None:
        """Run candidate scoring during dispatch-backward Stage1 payload A2A."""

        if not self.fixed_pipeline_overlap:
            return
        with self._pipeline_lock:
            windows = self._pipeline_planner_windows.get(layer_key)
        if windows is None:
            return
        layer = self.layers.get(layer_key)
        score_ready = None if layer is None else self._pipeline_ready_event(self._pipeline_device(layer))
        with self._pipeline_lock:
            if self._pipeline_planner_windows.get(layer_key) is not windows:
                return
            self._pipeline_planner_compute_events[layer_key] = score_ready
        windows.score_gate.set()

    def close_pipeline_planner_score_window(self, layer_key: str) -> None:
        """Record the preferred score window without adding a layer barrier.

        Candidate scoring only consumes the previous step's route and its
        correctness deadline is the next-step plan collection.  Waiting here
        would turn every layer's short A2A window into a host barrier and
        prevent the planner queue from carrying unfinished C4 work diagonally
        across later layers.
        """

        if not self.fixed_pipeline_overlap:
            return
        with self._pipeline_lock:
            windows = self._pipeline_planner_windows.get(layer_key)
        if windows is None:
            return
        deadline_event = self._pipeline_stage_event()
        with self._pipeline_lock:
            if self._pipeline_planner_windows.get(layer_key) is windows:
                windows.score_deadline_event = deadline_event

    def _collect_pipeline_plans(self, step: int) -> str:
        with self._pipeline_lock:
            futures = [
                (key, self._pipeline_plan_futures[key])
                for key in self._pipeline_layer_order
                if key in self._pipeline_plan_futures
            ]
        committed: list[str] = []
        raw_ms = 0.0
        deadline_exposed_ms = 0.0
        prepare_exposed_ms = 0.0
        collective_exposed_ms = 0.0
        score_exposed_ms = 0.0
        score_window_overrun_ms = 0.0
        prepare_window_exposed_ms = [0.0] * len(settings._PIPELINE_PREPARE_CUT_POINTS)
        collective_window_misses = 0
        score_window_misses = 0
        deadline_misses = 0
        latency_ms = 0.0
        prepare_device_ms = 0.0
        collective_device_ms = 0.0
        score_device_ms = 0.0
        prepare_substage_device_ms = defaultdict(float)
        prepare_substage_host_ms = defaultdict(float)
        prepare_substage_thread_cpu_ms = defaultdict(float)
        accepted = 0
        for layer_key, future in futures:
            with self._pipeline_lock:
                windows = self._pipeline_planner_windows.get(layer_key)
            wait_started = time.perf_counter()
            result = future.result()
            layer_exposed_ms = (time.perf_counter() - wait_started) * 1000.0
            deadline_exposed_ms += layer_exposed_ms
            if layer_exposed_ms > 0.01:
                deadline_misses += 1
            if windows is not None:
                layer_prepare_exposed_ms, per_window = self._pipeline_prepare_exposure_ms(windows)
                prepare_exposed_ms += layer_prepare_exposed_ms
                for window_index, value in enumerate(per_window):
                    prepare_window_exposed_ms[window_index] += value
                layer_collective_exposed_ms = self._pipeline_stage_exposure_ms(
                    windows.collective_deadline_event,
                    windows.collective_done_event,
                )
                layer_score_window_overrun_ms = self._pipeline_stage_exposure_ms(
                    windows.score_deadline_event,
                    windows.score_done_event,
                )
                collective_exposed_ms += layer_collective_exposed_ms
                score_window_overrun_ms += layer_score_window_overrun_ms
                collective_window_misses += int(layer_collective_exposed_ms > 0.01)
                score_window_misses += int(layer_score_window_overrun_ms > 0.01)
            raw_ms += result.raw_ms
            latency_ms += result.latency_ms
            prepare_device_ms += result.prepare_device_ms
            collective_device_ms += result.collective_device_ms
            score_device_ms += result.score_device_ms
            for stage, value in result.prepare_substage_device_ms.items():
                prepare_substage_device_ms[stage] += value
            for stage, value in result.prepare_substage_host_ms.items():
                prepare_substage_host_ms[stage] += value
            for stage, value in result.prepare_substage_thread_cpu_ms.items():
                prepare_substage_thread_cpu_ms[stage] += value
            layer = self.layers[layer_key]
            layer.last_plan = result.plan
            self._record_plan_metrics(result.plan)
            if result.source_step != int(step) or result.placement_version != int(layer.placement_version):
                self._accumulate_metric("hiermoe/pipeline_planner_stale", 1)
            elif result.plan.actions:
                self._pipeline_pending_plans[layer_key] = _PendingPipelinePlan(
                    source_step=result.source_step,
                    placement_version=result.placement_version,
                    plan=result.plan,
                )
                committed.extend(f"{layer_key}:{action.format()}" for action in result.plan.actions)
                accepted += 1
            with self._pipeline_lock:
                self._pipeline_plan_futures.pop(layer_key, None)
                self._pipeline_planner_windows.pop(layer_key, None)
                self._pipeline_planner_dispatch_events.pop(layer_key, None)
                self._pipeline_planner_compute_events.pop(layer_key, None)
        self._accumulate_metric("hiermoe/pipeline_planner_jobs", len(futures))
        self._accumulate_metric("hiermoe/pipeline_planner_accepted", accepted)
        self._accumulate_metric("hiermoe/pipeline_planner_raw_ms", raw_ms)
        self._accumulate_metric("hiermoe/pipeline_planner_latency_ms", latency_ms)
        self._accumulate_metric("hiermoe/pipeline_planner_prepare_device_ms", prepare_device_ms)
        self._accumulate_metric("hiermoe/pipeline_planner_collective_device_ms", collective_device_ms)
        self._accumulate_metric("hiermoe/pipeline_planner_score_device_ms", score_device_ms)
        for stage in settings._PIPELINE_PREPARE_SUBSTAGES:
            self._accumulate_metric(
                f"hiermoe/pipeline_planner_prepare_{stage}_device_ms",
                prepare_substage_device_ms[stage],
            )
            self._accumulate_metric(
                f"hiermoe/pipeline_planner_prepare_{stage}_host_ms",
                prepare_substage_host_ms[stage],
            )
            self._accumulate_metric(
                f"hiermoe/pipeline_planner_prepare_{stage}_thread_cpu_ms",
                prepare_substage_thread_cpu_ms[stage],
            )
        self._accumulate_metric("hiermoe/pipeline_planner_prepare_exposed_ms", prepare_exposed_ms)
        self._accumulate_metric("hiermoe/pipeline_planner_collective_exposed_ms", collective_exposed_ms)
        self._accumulate_metric("hiermoe/pipeline_planner_score_window_overrun_ms", score_window_overrun_ms)
        self._accumulate_metric("hiermoe/pipeline_planner_score_exposed_ms", score_exposed_ms)
        self._accumulate_metric("hiermoe/pipeline_planner_collective_window_miss", collective_window_misses)
        self._accumulate_metric("hiermoe/pipeline_planner_score_window_miss", score_window_misses)
        for window_index, value in enumerate(prepare_window_exposed_ms):
            self._accumulate_metric(
                f"hiermoe/pipeline_planner_prepare_window_{window_index}_exposed_ms",
                value,
            )
        self._accumulate_metric("hiermoe/pipeline_planner_deadline_exposed_ms", deadline_exposed_ms)
        self._accumulate_metric("hiermoe/pipeline_planner_deadline_miss", deadline_misses)
        self._accumulate_metric(
            "hiermoe/pipeline_planner_exposed_ms",
            prepare_exposed_ms + collective_exposed_ms + score_exposed_ms + deadline_exposed_ms,
        )
        exposed_total = float(self._placement_metrics.get("hiermoe/pipeline_planner_exposed_ms", 0.0))
        if raw_ms > 0.0:
            self._placement_metrics["hiermoe/pipeline_planner_hidden_ratio"] = max(
                0.0,
                min(1.0, 1.0 - exposed_total / raw_ms),
            )
        self.latest_pair = ",".join(committed) if committed else "none"
        return self.latest_pair

    @torch.no_grad()
    def _plan_current_layer(self, layer: ExpertLayerState, step: int) -> list[str]:
        calibration = layer.planner_calibration
        selected = layer.latest_selected_experts
        greedy_cover = self.expert_swap_selector == "hiermoe_greedy_cover_p1"
        if (
            selected is None
            or (selected.numel() == 0 and not greedy_cover)
            or (calibration is None and not greedy_cover)
        ):
            return []
        planner = self._planner_for_layer(
            layer,
            communication_scale=1.0 if calibration is None else calibration.communication_scale,
            forward_compute_per_assignment=0.0 if calibration is None else calibration.forward_compute_per_assignment,
            forward_compute_constant=0.0 if calibration is None else calibration.forward_compute_constant,
        )
        with _full_timing_range("hiermoe_placement_planning"):
            plan = planner.plan(
                selected,
                self._layer_layout(layer),
                layer.logical_to_physical,
                source_ranks=self.ep_rank,
                max_swaps=self.expert_swap_max_pairs_per_layer,
                max_replicas=self.max_replica_rounds if layer.slot_layout_enabled else 0,
                step=step,
                layer_seed=zlib.crc32(layer.key.encode("utf-8")),
            )
            layer.last_plan = plan
            self._record_plan_metrics(plan)
        committed = self._execute_placement_plan(layer, plan, timing_prefix="hiermoe_placement")
        if self.expert_swap_mode == "layer" and plan.local_physical_routes is not None:
            layer.pending_physical_routes = plan.local_physical_routes
            layer.pending_route_data_ptr = selected.data_ptr()
        return committed

    @torch.no_grad()
    def _plan_historical_layers(self, layers: list[ExpertLayerState], step: int) -> list[str]:
        """Plan all initialized layers from the previous forward routes as one batch."""

        if not layers:
            return []
        consensus_device = _local_tensor_view(layers[0].primary_parameter).device
        globally_ready = _placement_group_all_true_mask(
            [layer.latest_selected_experts is not None for layer in layers],
            device=consensus_device,
            ep_size=self.ep_size,
            ep_group=self.ep_group,
        )
        ready = [layer for layer, is_ready in zip(layers, globally_ready, strict=True) if is_ready]
        if not ready:
            return []
        if any(bool((self._layer_layout(layer) < 0).any().item()) for layer in ready):
            # Empty-slot initialization has sequential marginal dependencies
            # within each layer. It is excluded from steady-state timing.
            committed: list[str] = []
            for layer in ready:
                committed.extend(self._plan_current_layer(layer, step))
            return committed
        structural_signature = {
            (
                layer.latest_hidden_size,
                layer.latest_bytes_per_element,
                layer.num_local_experts,
                layer.num_experts,
            )
            for layer in ready
        }
        if len(structural_signature) != 1:
            committed = []
            for layer in ready:
                committed.extend(self._plan_current_layer(layer, step))
            return committed

        calibrations = [layer.planner_calibration for layer in ready]
        communication_scales = [
            1.0 if calibration is None else calibration.communication_scale for calibration in calibrations
        ]
        compute_slopes = [
            0.0 if calibration is None else calibration.forward_compute_per_assignment for calibration in calibrations
        ]
        compute_constants = [
            0.0 if calibration is None else calibration.forward_compute_constant for calibration in calibrations
        ]
        planner = self._planner_for_layer(
            ready[0],
            communication_scale=communication_scales[0],
            forward_compute_per_assignment=compute_slopes[0],
            forward_compute_constant=compute_constants[0],
        )
        if not isinstance(planner, GreedyCommunicationPlanner):
            raise RuntimeError("Historical batched planning requires GreedyCommunicationPlanner.")

        started = time.perf_counter()
        with _full_timing_range("hiermoe_historical_route_batch_plan"):
            plans = planner.plan_layers(
                [layer.latest_selected_experts for layer in ready],
                [self._layer_layout(layer) for layer in ready],
                [layer.logical_to_physical for layer in ready],
                source_ranks=self.ep_rank,
                max_swaps=self.expert_swap_max_pairs_per_layer,
                max_replicas=self.max_replica_rounds,
                layer_seeds=[zlib.crc32(layer.key.encode("utf-8")) for layer in ready],
                step=step,
                communication_scales=communication_scales,
                forward_compute_per_assignment=compute_slopes,
                forward_compute_constant=compute_constants,
                skip_final_route_update=True,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._accumulate_metric("hiermoe/placement_step_batch_planning_ms", elapsed_ms)
        self._accumulate_metric("hiermoe/placement_step_batch_layers", len(ready))

        for layer, plan in zip(ready, plans, strict=True):
            layer.last_plan = plan
            self._record_plan_metrics(plan)
        committed: list[str] = []
        for layer, plan in zip(ready, plans, strict=True):
            committed.extend(
                self._execute_placement_plan(
                    layer,
                    plan,
                    timing_prefix="hiermoe_historical_placement",
                )
            )
        return committed

    @torch.no_grad()
    def _reset_redundant_slots_for_online_freeze(self, layers: Sequence[ExpertLayerState]) -> None:
        """Keep canonical owners and expose every redundant R2 slot for greedy filling."""

        for layer in layers:
            if layer.slot_to_logical is None:
                raise RuntimeError(f"Online freeze requires a slot layout for layer {layer.key}.")
            owners = layer.logical_to_physical.detach().to(device="cpu", dtype=torch.long)
            if int(torch.unique(owners).numel()) != layer.num_experts:
                raise RuntimeError(f"Online freeze found non-unique owner slots in layer {layer.key}.")
            layout = torch.full((layer.num_physical_slots,), -1, dtype=torch.long)
            logical = torch.arange(layer.num_experts, dtype=torch.long)
            layout.scatter_(0, owners, logical)
            empty_slots = int((layout < 0).sum().item())
            if empty_slots != self.replica_slot_capacity:
                raise RuntimeError(
                    f"Online freeze expected {self.replica_slot_capacity} redundant slots in layer {layer.key}, "
                    f"found {empty_slots}."
                )
            layer.slot_to_logical = layout
            layer.fixed_r2_layout = False
            layer.active_quota_policy = ()
            layer.pending_physical_routes = None
            layer.pending_route_data_ptr = 0
            layer.invalidate_cache()

    @torch.no_grad()
    def _run_online_freeze_step(self, step: int) -> str:
        calibration_step = self._online_freeze_calibration_step
        planning_step = calibration_step + 1
        if int(step) < calibration_step:
            self.latest_pair = "none"
            return self.latest_pair
        if int(step) > planning_step:
            self.latest_pair = "none"
            return self.latest_pair

        layers = [self.layers[layer_key] for layer_key in sorted(self.layers)]
        if int(step) == calibration_step:
            self.prepare_calibrations(int(step))
            if any(layer.planner_calibration is None for layer in layers):
                raise RuntimeError(
                    f"Online freeze calibration did not complete at step {step}; "
                    "the profiled layer events were not ready on every EP rank."
                )
            self.latest_pair = "none"
            return self.latest_pair

        if any(layer.planner_calibration is None for layer in layers):
            raise RuntimeError(
                f"Online freeze planning at step {step} has no calibration from step {calibration_step}."
            )
        r2_costs: dict[str, float] = {}
        for layer in layers:
            calibration = layer.planner_calibration
            selected = layer.latest_selected_experts
            if calibration is None or selected is None:
                raise RuntimeError(f"Online freeze cannot score the R2 baseline for layer {layer.key}.")
            planner = self._planner_for_layer(
                layer,
                communication_scale=calibration.communication_scale,
                forward_compute_per_assignment=calibration.forward_compute_per_assignment,
                forward_compute_constant=calibration.forward_compute_constant,
            )
            copy_slots, _copy_mask = layer.copy_slots_for_device(selected.device)
            r2_costs[layer.key] = planner.score_layout(
                selected,
                self._layer_layout(layer),
                source_ranks=self.ep_rank,
                owner_slots=layer.logical_to_physical,
                step=int(step),
                layer_seed=zlib.crc32(layer.key.encode("utf-8")),
                max_copies=int(copy_slots.shape[1]),
            ).total
        self._reset_redundant_slots_for_online_freeze(layers)
        with _full_timing_range("hiermoe_online_freeze_plan"):
            committed = self._plan_historical_layers(layers, int(step))
        expected_actions = len(layers) * self.replica_slot_capacity
        if len(committed) != expected_actions:
            raise RuntimeError(f"Online freeze committed {len(committed)} cover actions, expected {expected_actions}.")
        final_costs = {layer.key: layer.last_plan.final_cost.total for layer in layers if layer.last_plan is not None}
        if len(final_costs) != len(layers):
            raise RuntimeError("Online freeze did not retain a final predicted cost for every layer.")
        r2_total = sum(r2_costs.values())
        final_total = sum(final_costs.values())
        self._accumulate_metric("hiermoe/online_freeze_cover_count", len(committed))
        self._accumulate_metric("hiermoe/online_freeze_r2_predicted_cost_ms", r2_total)
        self._accumulate_metric("hiermoe/online_freeze_final_predicted_cost_ms", final_total)
        self._accumulate_metric("hiermoe/online_freeze_predicted_gain_ms", r2_total - final_total)
        self._accumulate_metric(
            "hiermoe/online_freeze_predicted_speedup",
            r2_total / final_total if final_total > 0.0 else 0.0,
        )
        self.latest_pair = ",".join(committed)
        return self.latest_pair
