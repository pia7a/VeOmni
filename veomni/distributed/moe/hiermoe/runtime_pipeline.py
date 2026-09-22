"""Device streams, event windows, and worker lifecycle."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from ....utils.device import get_torch_device
from .placemoe.runtime import terminate_planner_process
from .runtime_tensors import _local_tensor_view
from .runtime_types import ExpertLayerState


class PipelineMixin:
    """Device streams, event windows, and worker lifecycle."""

    def placement_planning_enabled(self) -> bool:
        return self._hot_update or self._cost_model_verify

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
            self._pipeline_next_grad_index = 0
        if not self.fixed_pipeline_overlap:
            return

    def _pipeline_is_final_microstep(self) -> bool:
        return self._pipeline_micro_step + 1 >= self._pipeline_num_micro_steps

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
        for future in tuple(self._pipeline_grad_futures.values()):
            future.result()
        for handle in self._pipeline_grad_hook_handles:
            handle.remove()
        if self._pipeline_grad_executor is not None:
            self._pipeline_grad_executor.shutdown(wait=True, cancel_futures=False)

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
