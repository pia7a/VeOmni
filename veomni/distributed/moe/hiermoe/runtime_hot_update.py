"""Asynchronous planner jobs and atomic installation of layout/mapping artifacts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from importlib.util import find_spec
from typing import Any, Sequence

import torch
import torch.distributed as dist

from ....utils.device import synchronize
from . import runtime_settings as settings
from .placemoe.artifacts import build_placemoe_artifact, validate_placemoe_artifact
from .placemoe.runtime import (
    HotUpdateJob,
    PlaceMoECalibration,
    PlannerCommandSpec,
    UpdateKind,
    build_planner_command,
    launch_planner_process,
    planner_environment,
    terminate_planner_process,
)
from .placemoe.runtime.cpu_affinity import resolve_cpu_affinity
from .placemoe.types import LayerPlan, PlaceMoETopology
from .runtime_settings import logger
from .runtime_tensors import _cover_grouped_slot_entries_atomic, _ep_global_rank
from .runtime_types import ExpertLayerState, _CoverTensorEntry


class HotUpdateMixin:
    """Asynchronous planner jobs and atomic installation of layout/mapping artifacts."""

    @staticmethod
    def _bind_all_process_threads(cpu_ids: Sequence[int]) -> None:
        cpus = {int(cpu_id) for cpu_id in cpu_ids}
        if not cpus or not hasattr(os, "sched_setaffinity"):
            return
        task_root = "/proc/self/task"
        try:
            task_ids = tuple(int(name) for name in os.listdir(task_root) if name.isdigit())
        except OSError:
            task_ids = (0,)
        for task_id in task_ids:
            try:
                os.sched_setaffinity(task_id, cpus)
            except (OSError, ProcessLookupError):
                continue
        os.sched_setaffinity(0, cpus)

    def _configure_hot_update_training_affinity(self) -> None:
        if not self._hot_update:
            return
        node_rank = int(os.environ.get("GROUP_RANK", os.environ.get("NODE_RANK", "0")))
        if node_rank != 0:
            return
        plan = resolve_cpu_affinity(settings._HOT_UPDATE_RESOURCES)
        self._bind_all_process_threads(plan.training_cpu_ids)
        self._cpu_training_affinity = plan.training_cpu_ids
        self._cpu_planner_affinity = plan.planner_cpu_ids
        self._hot_update_resources = plan.planner_resources()
        self._hot_update_affinity_automatic = plan.automatic
        self._hot_update_planner_physical_cores = plan.planner_physical_cores
        logger.info_rank0(
            "PlaceMoE isolated hot-update CPU planner mode=%s training_cpus=%s planner_cpus=%s "
            "planner_physical_cores=%s workers=%s candidate_workers=%s worker_threads=%s.",
            "auto" if plan.automatic else "explicit",
            self._hot_update_resources.training_cpu_ids,
            self._hot_update_resources.planner_cpu_ids,
            plan.planner_physical_cores,
            plan.workers,
            plan.candidate_workers,
            plan.worker_threads,
        )

    def _hot_update_event(self, event: str, **values: Any) -> None:
        if self.ep_rank != 0:
            return
        root = os.path.abspath(settings._HOT_UPDATE_WORK_ROOT)
        os.makedirs(root, exist_ok=True)
        payload = {
            "event": str(event),
            "wall_time": time.time(),
            **values,
        }
        with open(os.path.join(root, "events.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _hot_update_builder_path(self) -> str:
        if settings._HOT_UPDATE_BUILDER:
            return os.path.abspath(settings._HOT_UPDATE_BUILDER)
        planner_spec = find_spec("placemoe.planner")
        if planner_spec is None or planner_spec.origin is None:
            raise RuntimeError("The installed PlaceMoE planner module could not be resolved.")
        return os.path.abspath(planner_spec.origin)

    def _write_hot_update_current_layout(
        self,
        layers: Sequence[ExpertLayerState],
        path: str,
        update_mode: str,
    ) -> None:
        plans: dict[str, LayerPlan] = {}
        for layer in layers:
            if layer.source_logical_to_physical is None:
                raise RuntimeError(f"PlaceMoE mapping update requires a source LUT for {layer.key}.")
            plans[layer.key] = LayerPlan(
                slot_to_logical=self._layer_layout(layer).tolist(),
                owner_slots=layer.logical_to_physical.detach().cpu().tolist(),
                source_logical_to_physical=layer.source_logical_to_physical.detach().cpu().tolist(),
            )
        payload = build_placemoe_artifact(
            plans,
            PlaceMoETopology(
                ep_size=self.ep_size,
                ranks_per_node=min(self.ep_size, self.hierarchy.local_world_size),
                num_experts=layers[0].num_experts,
                slots_per_rank=layers[0].num_local_experts,
            ),
            source={"algorithm": "placemoe-v1", "update_mode": update_mode},
        )
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    @torch.no_grad()
    def _capture_hot_update_routes(self, placement_step: int, training_step: int, job_dir: str) -> float:
        layers = [self.layers[layer_key] for layer_key in sorted(self.layers)]
        if not layers:
            raise RuntimeError("PlaceMoE hot update has no registered expert layers.")
        if any(
            layer.latest_selected_experts is None or int(layer.latest_route_step) != int(placement_step)
            for layer in layers
        ):
            missing = [
                layer.key
                for layer in layers
                if layer.latest_selected_experts is None or int(layer.latest_route_step) != int(placement_step)
            ]
            raise RuntimeError(
                f"PlaceMoE hot-update step {training_step} has stale or missing routes for {missing[:4]}."
            )
        started = time.perf_counter()
        capture_dir = os.path.join(job_dir, "routes", "step0000")
        if self.ep_rank == 0:
            os.makedirs(capture_dir, exist_ok=True)

        for layer_index, layer in enumerate(layers):
            selected = layer.latest_selected_experts
            assert selected is not None
            if selected.ndim != 2:
                raise RuntimeError(f"PlaceMoE hot updates require 2D routes for layer {layer.key}.")
            route = selected.detach().to(dtype=torch.int32).contiguous()
            shape = torch.tensor(tuple(int(value) for value in route.shape), dtype=torch.int64, device=route.device)
            if self.ep_size > 1:
                gathered_shapes = torch.empty((self.ep_size * 2,), dtype=torch.int64, device=route.device)
                dist.all_gather_into_tensor(gathered_shapes, shape, group=self.ep_group)
                shapes = gathered_shapes.view(self.ep_size, 2)
            else:
                shapes = shape.view(1, 2)
            max_numel = int((shapes[:, 0] * shapes[:, 1]).max().item())
            padded = torch.full((max_numel,), -1, dtype=torch.int32, device=route.device)
            padded[: route.numel()].copy_(route.reshape(-1))
            if self.ep_size > 1:
                gathered_routes = torch.empty(
                    (self.ep_size * max_numel,),
                    dtype=torch.int32,
                    device=route.device,
                )
                dist.all_gather_into_tensor(gathered_routes, padded, group=self.ep_group)
            else:
                gathered_routes = padded
            if self.ep_rank == 0:
                cpu_shapes = shapes.detach().cpu()
                cpu_routes = gathered_routes.detach().cpu().view(self.ep_size, max_numel)
                routes_by_rank = []
                for rank in range(self.ep_size):
                    rows, width = (int(value) for value in cpu_shapes[rank].tolist())
                    routes_by_rank.append(cpu_routes[rank, : rows * width].view(rows, width).clone())
                torch.save(
                    {
                        "format": "hiermoe-local-route-bundle-v1",
                        "ep_size": self.ep_size,
                        "source_training_step": int(training_step),
                        "layer_key": layer.key,
                        "routes_by_rank": routes_by_rank,
                    },
                    os.path.join(capture_dir, f"layer{layer_index:02d}_call0_all_ranks.pt"),
                )
            del gathered_routes, padded, shapes
        if self.ep_group is not None and self.ep_size > 1:
            dist.barrier(group=self.ep_group)
        return (time.perf_counter() - started) * 1000.0

    def _launch_hot_update(
        self,
        *,
        placement_step: int,
        training_step: int,
        update_mode: str,
    ) -> None:
        try:
            update_kind = UpdateKind(update_mode)
        except ValueError as error:
            raise ValueError(f"Unsupported PlaceMoE update mode {update_mode!r}.") from error
        layers = [self.layers[layer_key] for layer_key in sorted(self.layers)]
        job_dir = os.path.join(
            os.path.abspath(settings._HOT_UPDATE_WORK_ROOT),
            f"source_step_{int(training_step):06d}_{update_mode}",
        )
        snapshot_ms = self._capture_hot_update_routes(placement_step, training_step, job_dir)
        layout_path = os.path.join(job_dir, "layout.json")
        input_layout_path = os.path.join(job_dir, "current_layout.json")
        report_path = os.path.join(job_dir, "report.json")
        planner_log_path = os.path.join(job_dir, "planner.log")
        submitted_at = time.perf_counter()
        process: subprocess.Popen[bytes] | None = None
        if self.ep_rank == 0:
            os.makedirs(job_dir, exist_ok=True)
            builder = self._hot_update_builder_path()
            if not os.path.isfile(builder):
                raise RuntimeError(f"PlaceMoE planner does not exist: {builder}.")
            primary_slots = layers[0].num_experts // self.ep_size
            redundant_slots = layers[0].num_local_experts - primary_slots
            self._write_hot_update_current_layout(layers, input_layout_path, update_mode)
            resources = self._hot_update_resources
            calibration = getattr(
                self,
                "_hot_update_calibration",
                PlaceMoECalibration(
                    inter_ms_per_byte=settings._HOT_UPDATE_INTER_MS_PER_BYTE,
                    intra_ms_per_byte=settings._HOT_UPDATE_INTRA_MS_PER_BYTE,
                    route_ms_per_assignment=settings._HOT_UPDATE_ROUTE_MS_PER_ASSIGNMENT,
                    communication_multiplier=settings._HOT_UPDATE_COMMUNICATION_MULTIPLIER,
                    compute_ms_per_assignment=settings._HOT_UPDATE_COMPUTE_MS_PER_ASSIGNMENT,
                    compute_multiplier=settings._HOT_UPDATE_COMPUTE_MULTIPLIER,
                ),
            )
            command = build_planner_command(
                PlannerCommandSpec(
                    python=sys.executable,
                    planner_path=builder,
                    route_root=os.path.join(job_dir, "routes"),
                    kind=update_kind,
                    layer_keys=tuple(layer.key for layer in layers),
                    ep_size=self.ep_size,
                    ranks_per_node=min(self.ep_size, self.hierarchy.local_world_size),
                    hierarchy_group_sizes=tuple(int(size) for size in self.hierarchy.group_sizes),
                    num_experts=layers[0].num_experts,
                    slots_per_rank=layers[0].num_local_experts,
                    primary_slots_per_rank=primary_slots,
                    redundant_slots_per_rank=redundant_slots,
                    hidden_size=layers[0].latest_hidden_size,
                    bytes_per_element=layers[0].latest_bytes_per_element,
                    output_layout=layout_path,
                    output_report=report_path,
                    input_layout=input_layout_path,
                ),
                calibration,
                resources,
            )
            environment = planner_environment(resources)
            os.makedirs(job_dir, exist_ok=True)
            with open(planner_log_path, "wb") as planner_log:
                process = launch_planner_process(command, stdout=planner_log, environment=environment)
        self._hot_update_controller.start(
            HotUpdateJob(
                kind=update_kind,
                source_step=int(training_step),
                placement_versions=tuple(int(layer.placement_version) for layer in layers),
                submitted_at=submitted_at,
                snapshot_ms=snapshot_ms,
                job_dir=job_dir,
                layout_path=layout_path,
                report_path=report_path,
                planner_log_path=planner_log_path,
                process=process,
            )
        )
        self._hot_update_last_source_step = int(training_step)
        self._hot_update_last_snapshot_ms = snapshot_ms
        self._hot_update_event(
            "submitted",
            update_mode=update_mode,
            source_step=int(training_step),
            snapshot_ms=snapshot_ms,
            job_dir=job_dir,
            planner_pid=None if process is None else process.pid,
        )

    def _hot_update_status(self, state: HotUpdateJob, device: torch.device) -> int:
        status = 0
        if self.ep_rank == 0:
            assert state.process is not None
            return_code = state.process.poll()
            status = 0 if return_code is None else (1 if return_code == 0 else 2)
        if self.ep_size > 1:
            status_tensor = torch.tensor([status], dtype=torch.int32, device=device)
            dist.broadcast(
                status_tensor,
                src=_ep_global_rank(self.ep_group, 0),
                group=self.ep_group,
            )
            status = int(status_tensor.item())
        return status

    def _finish_hot_update_job(self, state: HotUpdateJob) -> None:
        """Reap the complete planner process group before releasing the job."""

        process = getattr(state, "process", None)
        if self.ep_rank == 0 and process is not None:
            terminate_planner_process(process)
        self._hot_update_controller.finish()

    def _broadcast_hot_update_payload(
        self,
        state: HotUpdateJob,
        device: torch.device,
    ) -> dict[str, Any]:
        payload: dict[str, Any] | None = None
        if self.ep_rank == 0:
            with open(state.layout_path, encoding="utf-8") as handle:
                payload = json.load(handle)
        if self.ep_size > 1:
            objects: list[Any] = [payload]
            dist.broadcast_object_list(
                objects,
                src=_ep_global_rank(self.ep_group, 0),
                group=self.ep_group,
                device=device,
            )
            payload = objects[0]
        if not isinstance(payload, dict):
            raise RuntimeError("PlaceMoE hot update broadcast an invalid layout payload.")
        try:
            validate_placemoe_artifact(payload)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("PlaceMoE hot update produced an invalid PlaceMoE artifact.") from error
        return payload

    def _prepare_hot_update_layer(
        self,
        layer: ExpertLayerState,
        raw_layer: dict[str, Any],
        update_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate and materialize one layer update without mutating runtime state."""

        target, owners = self._validate_placement_layout(
            layer,
            raw_layer.get("slot_to_logical", ()),
            raw_layer.get("owner_slots"),
        )
        if owners is None:
            raise RuntimeError(f"PlaceMoE layout for {layer.key} has no owner mapping.")
        source_lut = (
            torch.as_tensor(
                raw_layer.get("source_logical_to_physical", ()),
                dtype=torch.long,
            )
            .detach()
            .cpu()
        )
        expected_shape = (self.ep_size, layer.num_experts)
        if tuple(source_lut.shape) != expected_shape:
            raise RuntimeError(
                f"PlaceMoE mapping for {layer.key} has shape {tuple(source_lut.shape)}, expected {expected_shape}."
            )
        if bool(((source_lut < 0) | (source_lut >= layer.num_physical_slots)).any().item()):
            raise RuntimeError(f"PlaceMoE mapping for {layer.key} references an invalid slot.")

        current = self._layer_layout(layer)
        if update_mode == "mapping":
            if not torch.equal(target, current) or not torch.equal(
                owners,
                layer.logical_to_physical.detach().cpu(),
            ):
                raise RuntimeError(f"PlaceMoE mapping update attempted to change layout L for {layer.key}.")
            lookup_layout = current
        elif update_mode == "full":
            desired_experts = {int(value) for value in target.tolist() if int(value) >= 0}
            available_experts = {int(value) for value in current.tolist() if int(value) >= 0}
            missing = sorted(desired_experts - available_experts)
            if missing:
                raise RuntimeError(f"PlaceMoE layout cannot find current state for experts {missing} in {layer.key}.")
            lookup_layout = target
        else:
            raise RuntimeError(f"Unknown PlaceMoE update mode: {update_mode}.")

        logical = torch.arange(layer.num_experts, dtype=torch.long).view(1, -1)
        if not torch.equal(
            lookup_layout.index_select(0, source_lut.reshape(-1)).view_as(source_lut),
            logical.expand_as(source_lut),
        ):
            raise RuntimeError(f"PlaceMoE mapping for {layer.key} references the wrong expert.")
        return target, owners, source_lut

    @torch.no_grad()
    def _install_hot_update_layout(
        self,
        layer: ExpertLayerState,
        raw_layer: dict[str, Any],
    ) -> int:
        target, owners, source_lut = self._prepare_hot_update_layer(layer, raw_layer, "full")

        current = self._layer_layout(layer)
        changed_slots = [slot for slot in range(layer.num_physical_slots) if current[slot] != target[slot]]
        if changed_slots:
            state_tensors = self._slot_op_state_tensors(layer)
            grouped_entries: dict[tuple[int, int], list[_CoverTensorEntry]] = defaultdict(list)
            zero_entry_groups: list[tuple[int, list[_CoverTensorEntry]]] = []
            for dst_slot in changed_slots:
                desired = int(target[dst_slot].item())
                dst_rank = dst_slot // layer.num_local_experts
                if desired < 0:
                    zero_entry_groups.append(
                        (
                            dst_rank,
                            self._slot_op_cover_entries_from_tensors(
                                state_tensors,
                                num_local_experts=layer.num_local_experts,
                                src_slot=dst_slot,
                                dst_slot=dst_slot,
                            ),
                        )
                    )
                    continue
                candidates = [int(value) for value in torch.nonzero(current == desired, as_tuple=False).flatten()]
                if not candidates:
                    raise RuntimeError(
                        f"PlaceMoE layout cannot find current state for expert {desired} in {layer.key}."
                    )
                same_rank = [slot for slot in candidates if slot // layer.num_local_experts == dst_rank]
                owner_slot = int(layer.logical_to_physical[desired].item())
                src_slot = min(same_rank or ([owner_slot] if owner_slot in candidates else candidates))
                src_rank = src_slot // layer.num_local_experts
                grouped_entries[(src_rank, dst_rank)].extend(
                    self._slot_op_cover_entries_from_tensors(
                        state_tensors,
                        num_local_experts=layer.num_local_experts,
                        src_slot=src_slot,
                        dst_slot=dst_slot,
                    )
                )
            _cover_grouped_slot_entries_atomic(
                grouped_entries,
                self.ep_rank,
                self.ep_size,
                self.ep_group,
                zero_entry_groups=zero_entry_groups,
                debug_validate=self.debug_validate,
            )
            synchronize()

        layout_changed = not torch.equal(current, target)
        lut_changed = layer.source_logical_to_physical is None or not torch.equal(
            layer.source_logical_to_physical,
            source_lut,
        )
        layer.slot_to_logical = target
        layer.fixed_r2_layout = False
        layer.active_quota_policy = ()
        layer.pending_physical_routes = None
        layer.pending_route_data_ptr = 0
        layer.latest_physical_routes = None
        layer.latest_forward_traffic_endpoint_statistics = None
        self._refresh_layer_mapping_from_slots(layer, owners)
        layer.source_logical_to_physical = source_lut.clone()
        layer._device_source_mapping_cache.clear()
        layer.placement_version += int(layout_changed or lut_changed)
        return len(changed_slots)

    @torch.no_grad()
    def _install_hot_update_mapping(
        self,
        layer: ExpertLayerState,
        raw_layer: dict[str, Any],
    ) -> int:
        _target, _owners, source_lut = self._prepare_hot_update_layer(layer, raw_layer, "mapping")
        changed = layer.source_logical_to_physical is None or not torch.equal(
            layer.source_logical_to_physical,
            source_lut,
        )
        layer.pending_physical_routes = None
        layer.pending_route_data_ptr = 0
        layer.latest_physical_routes = None
        layer.latest_forward_traffic_endpoint_statistics = None
        layer.source_logical_to_physical = source_lut.clone()
        layer._device_source_mapping_cache.clear()
        layer.placement_version += int(changed)
        return int(changed)

    @torch.no_grad()
    def _apply_hot_update(self, state: HotUpdateJob, training_step: int) -> str:
        layers = [self.layers[layer_key] for layer_key in sorted(self.layers)]
        versions = tuple(int(layer.placement_version) for layer in layers)
        if versions != state.placement_versions:
            raise RuntimeError(
                f"PlaceMoE layout from step {state.source_step} is stale: "
                f"source versions={state.placement_versions}, current versions={versions}."
            )
        device = self._pipeline_device(layers[0])
        payload = self._broadcast_hot_update_payload(state, device)
        raw_layers = payload.get("layers")
        if not isinstance(raw_layers, dict) or set(raw_layers) != set(self.layers):
            raise RuntimeError("PlaceMoE layout layer keys do not match the registered model.")
        for layer in layers:
            raw_layer = raw_layers[layer.key]
            if not isinstance(raw_layer, dict):
                raise RuntimeError(f"PlaceMoE layout for {layer.key} is not a mapping.")
            self._prepare_hot_update_layer(layer, raw_layer, state.update_mode)

        migration_started = time.perf_counter()
        moved_slots = 0
        for layer in layers:
            raw_layer = raw_layers[layer.key]
            if state.update_mode == "full":
                moved_slots += self._install_hot_update_layout(layer, raw_layer)
            else:
                self._install_hot_update_mapping(layer, raw_layer)
        if self.ep_group is not None and self.ep_size > 1:
            dist.barrier(group=self.ep_group)
        migration_ms = (time.perf_counter() - migration_started) * 1000.0
        planner_ms = (time.perf_counter() - state.submitted_at) * 1000.0
        if self.ep_rank == 0 and os.path.isfile(state.report_path):
            with open(state.report_path, encoding="utf-8") as handle:
                report = json.load(handle)
            planner_ms = float(report.get("aggregate", {}).get("planner_wall_ms", planner_ms))
        self._hot_update_updates += 1
        if state.update_mode == "full":
            self._hot_update_layout_updates += 1
        else:
            self._hot_update_mapping_updates += 1
        self._hot_update_last_apply_step = int(training_step)
        self._hot_update_last_staleness_steps = int(training_step) - int(state.source_step)
        self._hot_update_last_planner_ms = planner_ms
        self._hot_update_last_migration_ms = migration_ms
        self._hot_update_last_moved_slots = moved_slots
        self._hot_update_event(
            "applied",
            update_mode=state.update_mode,
            source_step=state.source_step,
            apply_step=int(training_step),
            staleness_steps=int(training_step) - int(state.source_step),
            snapshot_ms=state.snapshot_ms,
            planner_ms=planner_ms,
            migration_ms=migration_ms,
            moved_slots=moved_slots,
        )
        return f"placemoe_{state.update_mode}_update:{state.source_step}->{int(training_step)}:{moved_slots}"

    @torch.no_grad()
    def _run_hot_update_step(self, placement_step: int) -> str:
        training_step = int(placement_step) + 1
        self._hot_update_controller.observe_step(training_step)
        state = self._hot_update_controller.active_job
        if state is not None:
            layers = [self.layers[layer_key] for layer_key in sorted(self.layers)]
            status = self._hot_update_status(state, self._pipeline_device(layers[0]))
            if status == 0:
                self.latest_pair = f"placemoe_{state.update_mode}_update_running"
                return self.latest_pair
            if status == 2:
                self._hot_update_event(
                    "failed",
                    update_mode=state.update_mode,
                    source_step=state.source_step,
                    planner_log_path=state.planner_log_path,
                )
                message = f"PlaceMoE planner failed for source step {state.source_step}; see {state.planner_log_path}."
                self._finish_hot_update_job(state)
                if self._hot_update_controller.failure_policy == "continue":
                    logger.error("%s Keeping the current layout and mapping.", message)
                    self.latest_pair = f"placemoe_hot_update_failed:{state.source_step}"
                    return self.latest_pair
                raise RuntimeError(message)
            try:
                self.latest_pair = self._apply_hot_update(state, training_step)
            finally:
                self._finish_hot_update_job(state)
            return self.latest_pair

        update_kind = self._hot_update_controller.next_update()
        if update_kind is None:
            self.latest_pair = "none"
            return self.latest_pair
        update_mode = update_kind.value
        self._launch_hot_update(
            placement_step=placement_step,
            training_step=training_step,
            update_mode=update_mode,
        )
        self.latest_pair = f"placemoe_{update_mode}_update_submitted:{training_step}"
        return self.latest_pair
