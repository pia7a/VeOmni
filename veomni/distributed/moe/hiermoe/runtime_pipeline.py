"""Device streams, event windows, and worker lifecycle."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any

import torch
import torch.distributed as dist

from ....utils.accelerator_timing import AcceleratorEvent, record_accelerator_event
from ....utils.device import get_torch_device
from . import runtime_settings as settings
from .placemoe.runtime import terminate_planner_process
from .runtime_tensors import _local_tensor_view
from .runtime_types import ExpertLayerState, _PipelinePlannerWindows


class PipelineMixin:
    """Device streams, event windows, and worker lifecycle."""

    def _ensure_pipeline_plan_worker_capacity(self) -> None:
        """Guarantee that every gate-blocked layer owns a planner worker.

        Prepare spans forward and backward windows. A bounded executor smaller
        than the number of MoE layers can otherwise fill with forward-order
        tasks while the reverse-order task needed by backward remains queued.
        """

        if not self.fixed_pipeline_overlap or self._pipeline_shutdown:
            return
        required = max(1, settings._PIPELINE_PLAN_WORKERS, len(self.layers))
        with self._pipeline_lock:
            executor = self._pipeline_plan_executor
            if executor is None or self._pipeline_plan_worker_capacity >= required:
                return
            if self._pipeline_plan_futures:
                raise RuntimeError("Cannot resize the HierMoE planner executor while planner jobs are active.")
            self._pipeline_plan_executor = ThreadPoolExecutor(
                max_workers=required,
                thread_name_prefix="hiermoe-plan",
            )
            self._pipeline_plan_worker_capacity = required
        executor.shutdown(wait=True, cancel_futures=False)

    def placement_planning_enabled(self) -> bool:
        return (
            self._hot_update
            or self._cost_model_verify
            or (
                not self._initial_layout_path
                and self._ablation_replay_mode != "static"
                and (self.expert_swap_max_pairs_per_layer > 0 or self.redundant_slot_increment_per_device > 0)
            )
        )

    def _pipeline_device(self, layer: ExpertLayerState) -> torch.device:
        return _local_tensor_view(layer.primary_parameter).device

    def _pipeline_stream(self, kind: str, device: torch.device) -> Any | None:
        if device.type == "cpu":
            return None
        key = (str(kind), device)
        cached = self._pipeline_streams.get(key)
        if cached is not None:
            return cached
        device_api = get_torch_device()
        device_api.set_device(device)
        try:
            cached = device_api.Stream(device=device)
        except TypeError:
            cached = device_api.Stream()
        self._pipeline_streams[key] = cached
        return cached

    @staticmethod
    def _pipeline_ready_event(device: torch.device) -> Any | None:
        if device.type == "cpu":
            return None
        device_api = get_torch_device()
        try:
            current_stream = device_api.current_stream(device)
        except TypeError:
            current_stream = device_api.current_stream()
        event = device_api.Event()
        event.record(current_stream)
        return event

    def _pipeline_stage_event(self) -> AcceleratorEvent | None:
        return record_accelerator_event() if (settings._PIPELINE_STAGE_TIMING or self.fixed_pipeline_overlap) else None

    def _run_pipeline_stream_task(
        self,
        kind: str,
        device: torch.device,
        ready_event: Any | None,
        task: Any,
    ) -> Any:
        if device.type == "cpu":
            return task()
        device_api = get_torch_device()
        device_api.set_device(device)
        stream = self._pipeline_stream(kind, device)
        assert stream is not None
        with device_api.stream(stream):
            if ready_event is not None:
                ready_events = ready_event if isinstance(ready_event, tuple) else (ready_event,)
                for event in ready_events:
                    if event is not None:
                        stream.wait_event(event)
            result = task()
        stream.synchronize()
        return result

    def configure_pipeline_microstep(self, step: int, micro_step: int, num_micro_steps: int) -> None:
        """Advance placement and gradient-overlap state at a microbatch boundary."""

        if not self.fixed_pipeline_overlap and not self.gradient_overlap_enabled:
            return
        self._pipeline_step = int(step)
        self._pipeline_micro_step = int(micro_step)
        self._pipeline_num_micro_steps = max(1, int(num_micro_steps))
        if int(micro_step) != 0:
            return
        self._begin_metrics_step(step)
        with self._pipeline_lock:
            if self._pipeline_grad_futures:
                raise RuntimeError("HierMoE started a new step before redundant gradient synchronization completed.")
            self._pipeline_grad_ready.clear()
            self._pipeline_grad_ready_events.clear()
            self._pipeline_grad_dispatch_complete.clear()
            self._pipeline_grad_window_waited.clear()
            self._pipeline_grad_comm_blocked = False
            self._pipeline_grad_window_exposed_ms = 0.0
            self._pipeline_layer_order = tuple(self.layers)
            self._pipeline_next_migration_index = 0
            self._pipeline_next_grad_index = 0
        if not self.fixed_pipeline_overlap:
            return
        if self._ablation_replay_mode == "off":
            self._ensure_pipeline_plan_worker_capacity()
        if self._ablation_migration_mode == "hidden":
            self._launch_next_pipeline_migration()

    def _pipeline_is_final_microstep(self) -> bool:
        return self._pipeline_micro_step + 1 >= self._pipeline_num_micro_steps

    def _wait_pipeline_host_event(self, layer_key: str, event: Event) -> None:
        while not event.wait(timeout=settings._PIPELINE_HOST_EVENT_POLL_SECONDS):
            with self._pipeline_lock:
                future = self._pipeline_plan_futures.get(layer_key)
            if future is not None and future.done():
                future.result()
            if self._pipeline_shutdown:
                return

    def _wait_pipeline_prepare_start(self, layer_key: str) -> None:
        with self._pipeline_lock:
            windows = self._pipeline_planner_windows[layer_key]
        self._wait_pipeline_host_event(layer_key, windows.prepare_gates[0])

    def _complete_pipeline_prepare_stage(
        self,
        layer_key: str,
        completed_stages: int,
    ) -> bool:
        with self._pipeline_lock:
            windows = self._pipeline_planner_windows[layer_key]
            first_window = windows.prepare_next_window
            next_window = first_window
            while next_window < len(settings._PIPELINE_PREPARE_CUT_POINTS) and settings._PIPELINE_PREPARE_CUT_POINTS[
                next_window
            ] <= int(completed_stages):
                next_window += 1
            if next_window == first_window:
                return False
            done_event = record_accelerator_event()
            for window_index in range(first_window, next_window):
                windows.prepare_done_events[window_index] = done_event
            windows.prepare_next_window = next_window
            enqueued = windows.prepare_enqueued[first_window:next_window]
            next_gate = (
                windows.prepare_gates[next_window]
                if next_window < len(settings._PIPELINE_PREPARE_CUT_POINTS)
                else None
            )
        for event in enqueued:
            event.set()
        if next_gate is None:
            return False
        self._wait_pipeline_host_event(layer_key, next_gate)
        return True

    def open_pipeline_planner_prepare_window(self, layer_key: str, window_index: int) -> None:
        if not self.fixed_pipeline_overlap:
            return
        if not 0 <= int(window_index) < len(settings._PIPELINE_PREPARE_CUT_POINTS):
            raise ValueError(f"Invalid pipeline Prepare window index: {window_index}")
        with self._pipeline_lock:
            windows = self._pipeline_planner_windows.get(layer_key)
        if windows is not None:
            windows.prepare_gates[int(window_index)].set()

    def release_pipeline_planner_prepare(self, layer_key: str) -> None:
        """Release all Prepare gates for isolated planner tests and shutdown fallbacks."""

        if not self.fixed_pipeline_overlap:
            return
        with self._pipeline_lock:
            windows = self._pipeline_planner_windows.get(layer_key)
        if windows is not None:
            for gate in windows.prepare_gates:
                gate.set()

    def close_pipeline_planner_prepare_window(self, layer_key: str, window_index: int) -> None:
        if not self.fixed_pipeline_overlap:
            return
        index = int(window_index)
        if not 0 <= index < len(settings._PIPELINE_PREPARE_CUT_POINTS):
            raise ValueError(f"Invalid pipeline Prepare window index: {window_index}")
        a2a_end_event = record_accelerator_event()
        with self._pipeline_lock:
            windows = self._pipeline_planner_windows.get(layer_key)
        if windows is None:
            return
        host_wait_started = time.perf_counter()
        self._wait_pipeline_host_event(layer_key, windows.prepare_enqueued[index])
        host_wait_ms = (time.perf_counter() - host_wait_started) * 1000.0
        self._accumulate_metric("hiermoe/pipeline_planner_prepare_host_gate_wait_ms", host_wait_ms)
        self._accumulate_metric(
            f"hiermoe/pipeline_planner_prepare_window_{index}_host_gate_wait_ms",
            host_wait_ms,
        )
        with self._pipeline_lock:
            if self._pipeline_planner_windows.get(layer_key) is not windows:
                return
            windows.prepare_a2a_end_events[index] = a2a_end_event
            planner_done_event = windows.prepare_done_events[index]
        if planner_done_event is None or planner_done_event.event is None:
            return
        layer = self.layers.get(layer_key)
        if layer is None:
            return
        device = self._pipeline_device(layer)
        if device.type == "cpu":
            return
        device_api = get_torch_device()
        try:
            current_stream = device_api.current_stream(device)
        except TypeError:
            current_stream = device_api.current_stream()
        current_stream.wait_event(planner_done_event.event)

    @staticmethod
    def _pipeline_prepare_exposure_ms(windows: _PipelinePlannerWindows) -> tuple[float, tuple[float, ...]]:
        per_window = []
        for a2a_end, planner_done in zip(
            windows.prepare_a2a_end_events,
            windows.prepare_done_events,
            strict=True,
        ):
            if a2a_end is None or planner_done is None:
                per_window.append(0.0)
            else:
                per_window.append(max(0.0, a2a_end.elapsed_time(planner_done)))
        return sum(per_window), tuple(per_window)

    @staticmethod
    def _pipeline_stage_exposure_ms(
        deadline: AcceleratorEvent | None,
        done: AcceleratorEvent | None,
    ) -> float:
        if deadline is None or done is None:
            return 0.0
        return max(0.0, deadline.elapsed_time(done))

    def shutdown_pipeline(self) -> None:
        if (
            not self.fixed_pipeline_overlap and not self.gradient_overlap_enabled and not self._hot_update
        ) or self._pipeline_shutdown:
            return
        self._pipeline_shutdown = True
        hot_update_state = self._hot_update_controller.active_job
        if hot_update_state is not None and hot_update_state.process is not None:
            terminate_planner_process(hot_update_state.process)
            self._hot_update_event(
                "terminated_on_shutdown",
                update_mode=hot_update_state.update_mode,
                source_step=hot_update_state.source_step,
            )
        self._hot_update_controller.finish()
        with self._pipeline_lock:
            windows = tuple(self._pipeline_planner_windows.values())
        for window in windows:
            for gate in window.prepare_gates:
                gate.set()
            for enqueued in window.prepare_enqueued:
                enqueued.set()
            window.collective_gate.set()
            if window.collective_future is None:
                window.collective_done.set()
                window.collective_result_ready.set()
            window.score_gate.set()
        for future in tuple(self._pipeline_plan_futures.values()):
            future.result()
        for future in tuple(self._pipeline_migration_futures.values()):
            future.result()
        for future in tuple(self._pipeline_grad_futures.values()):
            future.result()
        for handle in self._pipeline_grad_hook_handles:
            handle.remove()
        for executor in (
            self._pipeline_plan_executor,
            self._pipeline_collective_executor,
            self._pipeline_migration_executor,
            self._pipeline_grad_executor,
        ):
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)

    def destroy_pipeline_process_groups(self) -> None:
        if not self._owns_pipeline_grad_group:
            return
        group = self._pipeline_grad_group
        self._pipeline_grad_group = None
        self._owns_pipeline_grad_group = False
        if (
            group is not None
            and dist.is_available()
            and dist.is_initialized()
            and group != dist.GroupMember.NON_GROUP_MEMBER
        ):
            dist.destroy_process_group(group)
