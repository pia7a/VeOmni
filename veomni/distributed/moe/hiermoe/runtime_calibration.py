"""Runtime timing observations and communication/compute calibration."""

from __future__ import annotations

import json
import math
import os
import time
import zlib
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist

from ....utils.accelerator_timing import AcceleratorEvent, record_accelerator_event
from ....utils.device import get_device_type, synchronize
from . import runtime_settings as settings
from .greedy_planner import GreedyCommunicationPlanner
from .placemoe.runtime import PlaceMoECalibration
from .runtime_settings import _env_flag, logger
from .runtime_tensors import _local_tensor_view, _placement_group_boolean_consensus
from .runtime_types import ExpertLayerState, _CostModelTiming, _PendingLayerTiming, _PlannerCalibration


class CalibrationMixin:
    """Runtime timing observations and communication/compute calibration."""

    @staticmethod
    def _timing_event() -> AcceleratorEvent:
        event = record_accelerator_event()
        if event is not None:
            return event
        return AcceleratorEvent(device_type=get_device_type(), event=None, wall_time=time.perf_counter())

    def placement_timing_event(self) -> AcceleratorEvent:
        return self._timing_event()

    def record_layer_timing(
        self,
        *,
        layer_key: str,
        step: int,
        selected_experts: torch.Tensor,
        tokens_per_local_expert: torch.Tensor,
        dispatch_start: AcceleratorEvent,
        dispatch_end: AcceleratorEvent,
        compute_start: AcceleratorEvent,
        compute_end: AcceleratorEvent,
        combine_start: AcceleratorEvent,
        combine_end: AcceleratorEvent,
        selected_dim: int | None = None,
        communication_events: dict[str, tuple[AcceleratorEvent, AcceleratorEvent]] | None = None,
    ) -> None:
        del selected_dim
        layer = self.layers.get(layer_key)
        if layer is None or not self.placement_planning_enabled() or not self.layer_calibration_enabled():
            return
        if layer.slot_to_logical is None:
            layout = torch.full((layer.num_experts,), -1, dtype=torch.long)
            logical = torch.arange(layer.num_experts, dtype=torch.long)
            layout.scatter_(0, layer.logical_to_physical.to(torch.long), logical)
        else:
            layout = layer.slot_to_logical.detach().cpu().clone()
        timing = _PendingLayerTiming(
            step=int(step),
            selected_experts=selected_experts.detach(),
            slot_to_logical=layout,
            local_assignment_count=tokens_per_local_expert.detach().sum().to(dtype=torch.float32),
            dispatch_start=dispatch_start,
            dispatch_end=dispatch_end,
            compute_start=compute_start,
            compute_end=compute_end,
            combine_start=combine_start,
            combine_end=combine_end,
        )
        layer.pending_timing = timing
        capture_cost_model_sample = (
            self._cost_model_verify
            and int(self._online_freeze_calibration_step)
            <= int(step)
            <= int(self._online_freeze_calibration_step) + int(self._cost_model_validation_steps)
        ) or (self._online_freeze_cost_mode != "off" and int(step) == int(self._online_freeze_calibration_step))
        if capture_cost_model_sample:
            physical_routes = layer.latest_physical_routes
            if physical_routes is None or physical_routes.shape != selected_experts.shape:
                raise RuntimeError(
                    f"Cost-model verification did not capture the Forward physical routes for {layer_key}."
                )
            layer.cost_model_timings.append(
                _CostModelTiming(
                    step=int(step),
                    physical_routes=physical_routes.detach(),
                    local_expert_token_counts=tokens_per_local_expert.detach(),
                    local_assignment_count=tokens_per_local_expert.detach().sum().to(dtype=torch.float32),
                    communication_events=communication_events,
                    dispatch_start=dispatch_start,
                    dispatch_end=dispatch_end,
                    compute_start=compute_start,
                    compute_end=compute_end,
                    combine_start=combine_start,
                    combine_end=combine_end,
                )
            )

    def record_local_expert_token_counts(self, layer_key: str, tokens_per_local_expert: torch.Tensor) -> None:
        layer = self.layers.get(layer_key)
        if layer is None or not layer.slot_layout_enabled:
            return
        counts = tokens_per_local_expert.detach()
        if counts.ndim != 1 or int(counts.numel()) != int(layer.num_local_experts):
            return
        if layer.accumulated_tokens_per_local_expert is None:
            layer.accumulated_tokens_per_local_expert = counts.clone()
        else:
            if tuple(layer.accumulated_tokens_per_local_expert.shape) != tuple(counts.shape):
                layer.accumulated_tokens_per_local_expert = counts.clone()
            else:
                layer.accumulated_tokens_per_local_expert = layer.accumulated_tokens_per_local_expert.to(
                    device=counts.device, dtype=counts.dtype
                )
                layer.accumulated_tokens_per_local_expert.add_(counts)

    @staticmethod
    def _events_ready(timing: _PendingLayerTiming) -> bool:
        for accelerator_event in (
            timing.dispatch_end,
            timing.compute_end,
            timing.combine_end,
        ):
            event = accelerator_event.event
            query = getattr(event, "query", None)
            if callable(query):
                try:
                    if not query():
                        return False
                except Exception:
                    return False
        return True

    @staticmethod
    def _fit_nonnegative_compute_model(samples: list[tuple[float, float]]) -> tuple[float, float]:
        """Fit y = slope * x + intercept with both parameters constrained non-negative."""

        finite = [(x, y) for x, y in samples if math.isfinite(x) and math.isfinite(y) and x >= 0.0 and y >= 0.0]
        if not finite:
            return 0.0, 0.0
        x = torch.tensor([row[0] for row in finite], dtype=torch.float64)
        y = torch.tensor([row[1] for row in finite], dtype=torch.float64)
        candidates: list[tuple[float, float]] = []
        design = torch.stack((x, torch.ones_like(x)), dim=1)
        solution = torch.linalg.lstsq(design, y).solution
        unconstrained = (float(solution[0].item()), float(solution[1].item()))
        if unconstrained[0] >= 0.0 and unconstrained[1] >= 0.0:
            candidates.append(unconstrained)
        x_square_sum = float(torch.dot(x, x).item())
        if x_square_sum > 0.0:
            candidates.append((max(0.0, float(torch.dot(x, y).item()) / x_square_sum), 0.0))
        else:
            candidates.append((0.0, 0.0))
        candidates.append((0.0, max(0.0, float(y.mean().item()))))
        return min(
            candidates,
            key=lambda row: float(torch.square(row[0] * x + row[1] - y).sum().item()),
        )

    @staticmethod
    def _fit_positive_through_origin(samples: Sequence[tuple[float, float]]) -> float:
        """Fit y = slope * x through the origin with a non-negative slope."""

        finite = [(x, y) for x, y in samples if math.isfinite(x) and math.isfinite(y) and x > 0.0 and y >= 0.0]
        if not finite:
            return 0.0
        x = torch.tensor([row[0] for row in finite], dtype=torch.float64)
        y = torch.tensor([row[1] for row in finite], dtype=torch.float64)
        denominator = float(torch.dot(x, x).item())
        if denominator <= 0.0:
            return 0.0
        return max(0.0, float(torch.dot(x, y).item()) / denominator)

    @staticmethod
    def _fit_nonnegative_linear_model(
        feature_rows: Sequence[Sequence[float]],
        targets: Sequence[float],
    ) -> tuple[tuple[float, ...], float]:
        """Fit a small non-negative affine model by enumerating active sets."""

        if len(feature_rows) != len(targets) or not feature_rows:
            raise ValueError("Linear cost-model fitting requires non-empty paired samples.")
        feature_count = len(feature_rows[0])
        if feature_count <= 0 or any(len(row) != feature_count for row in feature_rows):
            raise ValueError("Linear cost-model feature rows must have one consistent non-zero width.")
        finite_rows = [
            (tuple(float(value) for value in row), float(target))
            for row, target in zip(feature_rows, targets, strict=True)
            if all(math.isfinite(float(value)) and float(value) >= 0.0 for value in row)
            and math.isfinite(float(target))
            and float(target) >= 0.0
        ]
        if not finite_rows:
            return tuple(0.0 for _ in range(feature_count)), 0.0
        x = torch.tensor([row for row, _target in finite_rows], dtype=torch.float64)
        y = torch.tensor([target for _row, target in finite_rows], dtype=torch.float64)
        best: tuple[float, tuple[float, ...], float] | None = None
        # With at most three physical features, exact active-set enumeration is
        # deterministic and avoids introducing a SciPy dependency.
        for mask in range(1 << feature_count):
            active = [index for index in range(feature_count) if mask & (1 << index)]
            for use_intercept in (False, True):
                columns = [x[:, index] for index in active]
                if use_intercept:
                    columns.append(torch.ones_like(y))
                if columns:
                    design = torch.stack(columns, dim=1)
                    solution = torch.linalg.lstsq(design, y).solution
                    if bool((solution < 0.0).any().item()):
                        continue
                    fitted = design @ solution
                else:
                    solution = torch.empty((0,), dtype=torch.float64)
                    fitted = torch.zeros_like(y)
                coefficients = [0.0 for _ in range(feature_count)]
                for position, feature_index in enumerate(active):
                    coefficients[feature_index] = float(solution[position].item())
                intercept = float(solution[-1].item()) if use_intercept else 0.0
                error = float(torch.square(fitted - y).sum().item())
                candidate = (error, tuple(coefficients), intercept)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        assert best is not None
        return best[1], best[2]

    @staticmethod
    def _predict_linear_model(
        feature_rows: Sequence[Sequence[float]],
        coefficients: Sequence[float],
        intercept: float,
    ) -> list[float]:
        return [
            float(intercept)
            + sum(float(coefficient) * float(value) for coefficient, value in zip(coefficients, row, strict=True))
            for row in feature_rows
        ]

    @staticmethod
    def _cost_model_diagnostics(
        actual_values: Sequence[float],
        predicted_values: Sequence[float],
    ) -> dict[str, float]:
        """Return deterministic regression diagnostics for one modeled phase."""

        actual = torch.tensor(tuple(actual_values), dtype=torch.float64)
        predicted = torch.tensor(tuple(predicted_values), dtype=torch.float64)
        if actual.numel() == 0 or actual.shape != predicted.shape:
            raise ValueError("Cost-model diagnostics require non-empty paired samples.")
        residual = predicted - actual
        squared_error = torch.square(residual)
        centered = actual - actual.mean()
        total_variance = float(torch.square(centered).sum().item())
        r_squared = 1.0 - float(squared_error.sum().item()) / total_variance if total_variance > 0.0 else float("nan")
        relative = residual.abs() / actual.abs().clamp_min(1.0e-6)
        return {
            "r_squared": r_squared,
            "mape_percent": float(relative.mean().item()) * 100.0,
            "rmse_ms": float(torch.sqrt(squared_error.mean()).item()),
            "max_abs_error_ms": float(residual.abs().max().item()),
            "actual_min_ms": float(actual.min().item()),
            "actual_max_ms": float(actual.max().item()),
            "actual_mean_ms": float(actual.mean().item()),
            "predicted_min_ms": float(predicted.min().item()),
            "predicted_max_ms": float(predicted.max().item()),
            "predicted_mean_ms": float(predicted.mean().item()),
        }

    @torch.no_grad()
    def _cost_model_step_observations(
        self,
        layers: Sequence[ExpertLayerState],
        *,
        step: int,
    ) -> dict[str, Any]:
        """Aggregate exact Forward-route features and measured NPU times."""

        ordered_layers = sorted(layers, key=lambda value: value.key)
        if not ordered_layers:
            raise RuntimeError("Cost-model verification requires registered expert layers.")
        common_device = _local_tensor_view(ordered_layers[0].primary_parameter).device
        local_sample_counts = torch.tensor(
            [sum(int(timing.step) == int(step) for timing in layer.cost_model_timings) for layer in ordered_layers],
            dtype=torch.int64,
            device=common_device,
        )
        if self.ep_group is not None and self.ep_size > 1:
            gathered_counts = torch.empty(
                (self.ep_size * len(ordered_layers),),
                dtype=local_sample_counts.dtype,
                device=common_device,
            )
            dist.all_gather_into_tensor(gathered_counts, local_sample_counts, group=self.ep_group)
            gathered_counts = gathered_counts.view(self.ep_size, len(ordered_layers))
        else:
            gathered_counts = local_sample_counts.view(1, -1)
        if bool((gathered_counts != gathered_counts[0]).any().item()):
            raise RuntimeError(
                f"Cost-model verification sample counts differ across EP ranks at step {step}: "
                f"{gathered_counts.detach().cpu().tolist()}."
            )
        if bool((local_sample_counts <= 0).any().item()):
            raise RuntimeError(
                f"Cost-model verification has missing layer samples at step {step}: "
                f"{local_sample_counts.detach().cpu().tolist()}."
            )

        synchronize()
        local_packed_rows: list[torch.Tensor] = []
        local_assignment_packed_rows: list[torch.Tensor] = []
        local_timing_rows: list[tuple[float, ...]] = []
        local_expert_token_rows: list[torch.Tensor] = []
        row_layer_indices: list[int] = []
        row_call_indices: list[int] = []
        layer_row_ranges: list[tuple[GreedyCommunicationPlanner, int, int]] = []
        row_start = 0
        if int(self.hierarchy.selected_dim) == 3:
            communication_event_names = (
                "stage1_a2a",
                "stage2_a2a",
                "stage3_a2a",
                "combine_stage3_a2a",
                "combine_stage2_a2a",
                "combine_stage1_a2a",
            )
        elif int(self.hierarchy.selected_dim) == 2:
            communication_event_names = ("stage1_a2a", "stage2_a2a", "combine_stage2_a2a", "combine_stage1_a2a")
        else:
            communication_event_names = ("stage2_a2a", "combine_stage2_a2a")
        for layer_index, layer in enumerate(ordered_layers):
            timings = [timing for timing in layer.cost_model_timings if int(timing.step) == int(step)]
            if not all(
                timing.dispatch_start.elapsed_time(timing.dispatch_end) >= 0.0
                and timing.compute_start.elapsed_time(timing.compute_end) >= 0.0
                and timing.combine_start.elapsed_time(timing.combine_end) >= 0.0
                for timing in timings
            ):
                raise RuntimeError(f"Cost-model verification found an invalid timing event in {layer.key}.")
            planner = self._planner_for_layer(
                layer,
                communication_scale=1.0,
                forward_compute_per_assignment=0.0,
                forward_compute_constant=0.0,
            )
            if not isinstance(planner, GreedyCommunicationPlanner):
                # Cost verification needs the exact hierarchical traffic
                # feature extractor, independently of the runtime selector.
                planner = self._cpu_exact_planner_for_layer(layer)
            routes = [self._routes_for_cost_model_planner(layer, timing.physical_routes) for timing in timings]
            if all(route.shape == routes[0].shape for route in routes):
                stacked_routes = torch.stack(routes, dim=0)
                packed = planner._local_packed_counts(stacked_routes)
                assignment_packed = planner._local_packed_assignment_counts(stacked_routes)
            else:
                packed = torch.cat([planner._local_packed_counts(route) for route in routes], dim=0)
                assignment_packed = torch.cat(
                    [planner._local_packed_assignment_counts(route) for route in routes],
                    dim=0,
                )
            local_packed_rows.append(packed)
            local_assignment_packed_rows.append(assignment_packed)
            for call_index, timing in enumerate(timings):
                stage_times = [-1.0 for _ in communication_event_names]
                if timing.communication_events is not None and all(
                    name in timing.communication_events for name in communication_event_names
                ):
                    stage_times = [
                        timing.communication_events[name][0].elapsed_time(timing.communication_events[name][1])
                        for name in communication_event_names
                    ]
                local_timing_rows.append(
                    (
                        timing.dispatch_start.elapsed_time(timing.dispatch_end)
                        + timing.combine_start.elapsed_time(timing.combine_end),
                        timing.compute_start.elapsed_time(timing.compute_end),
                        float(timing.local_assignment_count.item()),
                        *stage_times,
                    )
                )
                local_expert_token_rows.append(timing.local_expert_token_counts.to(dtype=torch.float32))
                row_layer_indices.append(layer_index)
                row_call_indices.append(call_index)
            row_end = row_start + len(timings)
            layer_row_ranges.append((planner, row_start, row_end))
            row_start = row_end

        local_packed = torch.cat(local_packed_rows, dim=0)
        local_assignment_packed = torch.cat(local_assignment_packed_rows, dim=0)
        if self.ep_group is not None and self.ep_size > 1:
            source_packed = torch.empty(
                (self.ep_size * row_start, local_packed.shape[1]),
                dtype=local_packed.dtype,
                device=common_device,
            )
            dist.all_gather_into_tensor(source_packed, local_packed.contiguous(), group=self.ep_group)
            source_packed = source_packed.view(self.ep_size, row_start, local_packed.shape[1])
            source_assignment_packed = torch.empty_like(source_packed)
            dist.all_gather_into_tensor(
                source_assignment_packed,
                local_assignment_packed.contiguous(),
                group=self.ep_group,
            )
            source_assignment_packed = source_assignment_packed.view(
                self.ep_size,
                row_start,
                local_assignment_packed.shape[1],
            )
        else:
            source_packed = local_packed.unsqueeze(0)
            source_assignment_packed = local_assignment_packed.unsqueeze(0)
        global_packed = source_packed.sum(dim=0)

        communication_units = torch.empty(
            (row_start,),
            dtype=torch.float32,
            device=common_device,
        )
        receive_only_communication_units = torch.empty_like(communication_units)
        level_count = len(layer_row_ranges[0][0]._count_widths())
        source_send_maxima = torch.empty(
            (row_start, level_count),
            dtype=torch.float32,
            device=common_device,
        )
        destination_receive_maxima = torch.empty_like(source_send_maxima)
        traffic_features: dict[str, torch.Tensor] = {}
        for planner, start, end in layer_row_ranges:
            receive_only_communication_units[start:end] = planner._communication_cost_details(
                global_packed[start:end]
            )[1]
            (
                _communication,
                communication_units[start:end],
                source_send_maxima[start:end],
                destination_receive_maxima[start:end],
                _selected_dim,
            ) = planner._source_aware_communication_cost_details(source_packed[:, start:end])
            layer_features = planner._hierarchical_traffic_features(
                source_packed[:, start:end],
                source_assignment_packed[:, start:end],
            )
            for name, values in layer_features.items():
                target = traffic_features.get(name)
                if target is None:
                    target = torch.empty((row_start,), dtype=torch.float32, device=common_device)
                    traffic_features[name] = target
                target[start:end] = values

        local_timings = torch.tensor(local_timing_rows, dtype=torch.float32, device=common_device)
        local_expert_tokens = torch.stack(local_expert_token_rows).to(device=common_device, dtype=torch.float32)
        if self.ep_group is not None and self.ep_size > 1:
            gathered_timings = torch.empty(
                (self.ep_size * row_start, local_timings.shape[1]),
                dtype=local_timings.dtype,
                device=common_device,
            )
            dist.all_gather_into_tensor(gathered_timings, local_timings, group=self.ep_group)
            gathered_timings = gathered_timings.view(self.ep_size, row_start, local_timings.shape[1])
            gathered_expert_tokens = torch.empty(
                (self.ep_size * row_start, local_expert_tokens.shape[1]),
                dtype=local_expert_tokens.dtype,
                device=common_device,
            )
            dist.all_gather_into_tensor(gathered_expert_tokens, local_expert_tokens.contiguous(), group=self.ep_group)
            gathered_expert_tokens = gathered_expert_tokens.view(self.ep_size, row_start, local_expert_tokens.shape[1])
        else:
            gathered_timings = local_timings.view(1, row_start, local_timings.shape[1])
            gathered_expert_tokens = local_expert_tokens.view(1, row_start, local_expert_tokens.shape[1])

        actual_communication = gathered_timings[:, :, 0].max(dim=0).values
        actual_compute = gathered_timings[:, :, 1].max(dim=0).values
        peak_assignments = gathered_timings[:, :, 2].max(dim=0).values
        stage_timings = gathered_timings[:, :, 3:]
        raw_a2a_available = bool((stage_timings >= 0.0).all().item())
        actual_stage_a2a = (
            stage_timings.max(dim=0).values if raw_a2a_available else torch.empty((0, 0), device=common_device)
        )
        actual_raw_a2a = actual_stage_a2a.sum(dim=1) if raw_a2a_available else torch.empty((0,), device=common_device)
        source_destination_assignments = source_assignment_packed[:, :, : self.ep_size].sum(dim=0)
        destination_assignments = gathered_timings[:, :, 2].transpose(0, 1)
        destination_assignment_deltas = (source_destination_assignments - destination_assignments).abs()
        source_assignment_totals = source_destination_assignments.sum(dim=1)
        destination_assignment_totals = destination_assignments.sum(dim=1)
        return {
            "sample_count": row_start,
            "compute_fit_sample_count": int(self.ep_size * row_start),
            "communication_units": communication_units.detach().cpu().tolist(),
            "receive_only_communication_units": receive_only_communication_units.detach().cpu().tolist(),
            "communication_level_names": [
                "rank",
                *[
                    f"group_{int(size)}"
                    for size in self.hierarchy.group_sizes[: max(0, int(self.hierarchy.selected_dim) - 1)]
                ],
            ],
            "source_send_maxima": source_send_maxima.detach().cpu().tolist(),
            "destination_receive_maxima": destination_receive_maxima.detach().cpu().tolist(),
            "traffic_features": {name: values.detach().cpu().tolist() for name, values in traffic_features.items()},
            "peak_assignments": peak_assignments.detach().cpu().tolist(),
            "actual_communication_ms": actual_communication.detach().cpu().tolist(),
            "actual_compute_ms": actual_compute.detach().cpu().tolist(),
            "actual_raw_a2a_ms": actual_raw_a2a.detach().cpu().tolist(),
            "actual_stage_a2a_ms": actual_stage_a2a.detach().cpu().tolist(),
            "actual_stage_a2a_names": list(communication_event_names),
            "sample_alignment": {
                "ep_size": int(self.ep_size),
                "row_count_per_rank": int(row_start),
                "layer_keys": [layer.key for layer in ordered_layers],
                "row_layer_indices": row_layer_indices,
                "row_call_indices": row_call_indices,
                "source_assignment_totals": source_assignment_totals.detach().cpu().tolist(),
                "destination_assignment_totals": destination_assignment_totals.detach().cpu().tolist(),
                "destination_rank_mismatch_counts": (destination_assignment_deltas > 0.5)
                .sum(dim=1)
                .detach()
                .cpu()
                .tolist(),
                "destination_rank_max_abs_deltas": destination_assignment_deltas.max(dim=1)
                .values.detach()
                .cpu()
                .tolist(),
            },
            "paired_expert_token_counts": gathered_expert_tokens.reshape(-1, gathered_expert_tokens.shape[-1])
            .detach()
            .cpu()
            .tolist(),
            "paired_assignments": gathered_timings[:, :, 2].reshape(-1).detach().cpu().tolist(),
            "paired_compute_ms": gathered_timings[:, :, 1].reshape(-1).detach().cpu().tolist(),
        }

    def _record_cost_model_report(self, phase: str, report: dict[str, Any]) -> None:
        prefix = f"hiermoe/cost_model_{phase}"
        self._accumulate_metric(f"{prefix}_samples", int(report["sample_count"]))
        self._accumulate_metric(f"{prefix}_communication_r2", float(report["communication"]["r_squared"]))
        self._accumulate_metric(
            f"{prefix}_communication_mape_percent",
            float(report["communication"]["mape_percent"]),
        )
        self._accumulate_metric(
            f"{prefix}_receive_only_communication_r2",
            float(report["receive_only_communication"]["r_squared"]),
        )
        self._accumulate_metric(
            f"{prefix}_receive_only_communication_mape_percent",
            float(report["receive_only_communication"]["mape_percent"]),
        )
        self._accumulate_metric(f"{prefix}_compute_r2", float(report["compute"]["r_squared"]))
        self._accumulate_metric(f"{prefix}_compute_mape_percent", float(report["compute"]["mape_percent"]))
        self._accumulate_metric(f"{prefix}_joint_r2", float(report["joint"]["r_squared"]))
        self._accumulate_metric(f"{prefix}_joint_mape_percent", float(report["joint"]["mape_percent"]))
        network_joint_models = report.get("traffic_feature_models", {}).get("network_joint", {})
        if network_joint_models:
            best_network_joint = min(
                network_joint_models.values(),
                key=lambda row: (
                    float(row["mape_percent"]),
                    -float(row["r_squared"]),
                ),
            )
            self._accumulate_metric(
                f"{prefix}_network_joint_r2",
                float(best_network_joint["r_squared"]),
            )
            self._accumulate_metric(
                f"{prefix}_network_joint_mape_percent",
                float(best_network_joint["mape_percent"]),
            )
        self._accumulate_metric(
            f"{prefix}_communication_units_min",
            float(report["communication_units"]["min"]),
        )
        self._accumulate_metric(
            f"{prefix}_communication_units_max",
            float(report["communication_units"]["max"]),
        )
        self._accumulate_metric(
            f"{prefix}_peak_assignments_min",
            float(report["peak_assignments"]["min"]),
        )
        self._accumulate_metric(
            f"{prefix}_peak_assignments_max",
            float(report["peak_assignments"]["max"]),
        )
        logger.info_rank0("HierMoE cost model %s report: %s", phase, json.dumps(report, sort_keys=True))

    @torch.no_grad()
    def _run_cost_model_verification(self, layers: Sequence[ExpertLayerState], step: int) -> str:
        calibration_step = int(self._online_freeze_calibration_step)
        validation_end_step = calibration_step + int(self._cost_model_validation_steps)
        if int(step) < calibration_step or int(step) > validation_end_step:
            self.latest_pair = "none"
            return self.latest_pair

        started = time.perf_counter()
        observations = self._cost_model_step_observations(layers, step=int(step))
        communication_units = list(observations["communication_units"])
        receive_only_communication_units = list(
            observations.get("receive_only_communication_units", communication_units)
        )
        peak_assignments = list(observations["peak_assignments"])
        actual_communication = list(observations["actual_communication_ms"])
        actual_compute = list(observations["actual_compute_ms"])
        actual_raw_a2a = list(observations.get("actual_raw_a2a_ms", ()))
        traffic_features = {
            str(name): list(values) for name, values in dict(observations.get("traffic_features", {})).items()
        }
        base_feature_models = {
            "legacy_source_aware": ("legacy_source_aware",),
            "stage_unique_endpoint": ("stage_unique_endpoint_link_units",),
            "stage_payload_endpoint": ("stage_payload_endpoint_link_units",),
            "stage_payload_endpoint_edge": (
                "stage_payload_endpoint_link_units",
                "stage_payload_edge_link_units",
            ),
            "stage_payload_inter_intra": (
                "stage1_payload_endpoint_bytes",
                "stage2_payload_endpoint_bytes",
            ),
            "stage_payload_levels": (
                "stage1_payload_endpoint_bytes",
                "stage2_payload_endpoint_bytes",
                "stage3_payload_endpoint_bytes",
            ),
            "stage_payload_lane_shared_node": (
                "stage_payload_endpoint_link_units",
                "stage_shared_node_endpoint_link_units",
            ),
            "stage_remote_payload_endpoint": ("stage_remote_payload_endpoint_link_units",),
            "stage_remote_payload_endpoint_self": (
                "stage_remote_payload_endpoint_link_units",
                "stage_self_payload_link_units",
            ),
            "stage_remote_payload_endpoint_edge": (
                "stage_remote_payload_endpoint_link_units",
                "stage_remote_payload_edge_link_units",
            ),
        }
        feature_values = {"legacy_source_aware": communication_units, **traffic_features}
        base_feature_models = {
            name: features
            for name, features in base_feature_models.items()
            if all(feature in feature_values for feature in features)
        }
        model_rows = {
            name: [
                [float(feature_values[feature][row]) for feature in features]
                for row in range(len(actual_communication))
            ]
            for name, features in base_feature_models.items()
        }
        joint_model_rows = {
            name: [[*row, float(peak_assignments[index])] for index, row in enumerate(rows)]
            for name, rows in model_rows.items()
        }

        if int(step) == calibration_step:
            communication_slope, communication_constant = self._fit_nonnegative_compute_model(
                list(zip(communication_units, actual_communication, strict=True))
            )
            compute_slope, compute_constant = self._fit_nonnegative_compute_model(
                list(
                    zip(
                        observations["paired_assignments"],
                        observations["paired_compute_ms"],
                        strict=True,
                    )
                )
            )
            self._cost_model_verify_coefficients = (
                communication_slope,
                communication_constant,
                compute_slope,
                compute_constant,
            )
            self._cost_model_verify_receive_only_coefficients = self._fit_nonnegative_compute_model(
                list(zip(receive_only_communication_units, actual_communication, strict=True))
            )
            actual_joint_targets = [
                communication + compute
                for communication, compute in zip(actual_communication, actual_compute, strict=True)
            ]
            feature_coefficients: dict[str, dict[str, tuple[tuple[float, ...], float]]] = {
                "communication": {
                    name: self._fit_nonnegative_linear_model(rows, actual_communication)
                    for name, rows in model_rows.items()
                },
                "joint": {
                    name: self._fit_nonnegative_linear_model(rows, actual_joint_targets)
                    for name, rows in joint_model_rows.items()
                },
            }
            if actual_raw_a2a:
                feature_coefficients["raw_a2a"] = {
                    name: self._fit_nonnegative_linear_model(rows, actual_raw_a2a) for name, rows in model_rows.items()
                }
                # The placement objective is network A2A plus expert compute,
                # not the wider dispatch/combine region. Compose the two
                # independently calibrated models so held-out validation
                # measures exactly the objective consumed by the layout
                # planner. Keep the wider ``joint`` target above as a
                # diagnostic for local remap/pack and rank-arrival overheads.
                feature_coefficients["network_joint"] = {
                    name: (
                        (*raw_coefficients, compute_slope),
                        raw_intercept + compute_constant,
                    )
                    for name, (raw_coefficients, raw_intercept) in feature_coefficients["raw_a2a"].items()
                }
            self._cost_model_verify_feature_coefficients = feature_coefficients
            phase = "calibration"
        else:
            if self._cost_model_verify_coefficients is None:
                raise RuntimeError("Cost-model validation has no coefficients from the calibration step.")
            if self._cost_model_verify_receive_only_coefficients is None:
                raise RuntimeError("Cost-model validation has no receive-only coefficients from the calibration step.")
            if self._cost_model_verify_feature_coefficients is None:
                raise RuntimeError("Cost-model validation has no traffic-feature coefficients.")
            communication_slope, communication_constant, compute_slope, compute_constant = (
                self._cost_model_verify_coefficients
            )
            phase = "validation"

        assert self._cost_model_verify_receive_only_coefficients is not None
        receive_only_slope, receive_only_constant = self._cost_model_verify_receive_only_coefficients
        predicted_communication = [
            communication_slope * value + communication_constant for value in communication_units
        ]
        receive_only_predicted_communication = [
            receive_only_slope * value + receive_only_constant for value in receive_only_communication_units
        ]
        predicted_compute = [compute_slope * value + compute_constant for value in peak_assignments]
        actual_joint = [
            communication + compute
            for communication, compute in zip(actual_communication, actual_compute, strict=True)
        ]
        predicted_joint = [
            communication + compute
            for communication, compute in zip(predicted_communication, predicted_compute, strict=True)
        ]
        assert self._cost_model_verify_feature_coefficients is not None
        feature_model_report: dict[str, dict[str, Any]] = {}
        target_rows: dict[str, tuple[dict[str, list[list[float]]], list[float]]] = {
            "communication": (model_rows, actual_communication),
            "joint": (
                joint_model_rows,
                [
                    communication + compute
                    for communication, compute in zip(actual_communication, actual_compute, strict=True)
                ],
            ),
        }
        if actual_raw_a2a and "raw_a2a" in self._cost_model_verify_feature_coefficients:
            target_rows["raw_a2a"] = (model_rows, actual_raw_a2a)
        if actual_raw_a2a and "network_joint" in self._cost_model_verify_feature_coefficients:
            target_rows["network_joint"] = (
                joint_model_rows,
                [raw_a2a + compute for raw_a2a, compute in zip(actual_raw_a2a, actual_compute, strict=True)],
            )
        for target_name, (rows_by_model, targets) in target_rows.items():
            target_report: dict[str, Any] = {}
            for model_name, rows in rows_by_model.items():
                coefficients, intercept = self._cost_model_verify_feature_coefficients[target_name][model_name]
                predicted = self._predict_linear_model(rows, coefficients, intercept)
                target_report[model_name] = {
                    "feature_names": [
                        *base_feature_models[model_name],
                        *(["peak_assignments"] if target_name in {"joint", "network_joint"} else []),
                    ],
                    "coefficients": list(coefficients),
                    "intercept_ms": float(intercept),
                    **self._cost_model_diagnostics(targets, predicted),
                }
            feature_model_report[target_name] = target_report
        report: dict[str, Any] = {
            "step": int(step),
            "sample_count": int(observations["sample_count"]),
            "compute_fit_sample_count": int(observations["compute_fit_sample_count"]),
            "coefficients": {
                "communication_ms_per_model_unit": communication_slope,
                "communication_constant_ms": communication_constant,
                "compute_ms_per_assignment": compute_slope,
                "compute_constant_ms": compute_constant,
                "receive_only_communication_ms_per_model_unit": receive_only_slope,
                "receive_only_communication_constant_ms": receive_only_constant,
            },
            "communication_units": {
                "min": min(communication_units),
                "max": max(communication_units),
                "mean": sum(communication_units) / len(communication_units),
            },
            "peak_assignments": {
                "min": min(peak_assignments),
                "max": max(peak_assignments),
                "mean": sum(peak_assignments) / len(peak_assignments),
            },
            "communication": self._cost_model_diagnostics(actual_communication, predicted_communication),
            "receive_only_communication": self._cost_model_diagnostics(
                actual_communication,
                receive_only_predicted_communication,
            ),
            "compute": self._cost_model_diagnostics(actual_compute, predicted_compute),
            "joint": self._cost_model_diagnostics(actual_joint, predicted_joint),
            "traffic_feature_models": feature_model_report,
            "traffic_feature_ranges": {
                name: {
                    "min": min(float(value) for value in values),
                    "max": max(float(value) for value in values),
                    "mean": sum(float(value) for value in values) / len(values),
                }
                for name, values in traffic_features.items()
            },
            "sample_alignment": observations.get("sample_alignment", {"available": False}),
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
        if self._export_cost_model_samples:
            paired_assignments = [float(value) for value in observations["paired_assignments"]]
            paired_compute = [float(value) for value in observations["paired_compute_ms"]]
            actual_stage_rows = [
                [float(value) for value in row] for row in observations.get("actual_stage_a2a_ms", ())
            ]
            offline_samples = {
                "paired_expert_token_counts": observations["paired_expert_token_counts"],
                "paired_assignments": observations["paired_assignments"],
                "paired_compute_ms": observations["paired_compute_ms"],
                "peak_assignments": observations["peak_assignments"],
                "actual_communication_ms": observations["actual_communication_ms"],
                **{
                    name: [float(value) for value in values]
                    for name, values in traffic_features.items()
                    if name.startswith("stage") and name.endswith("_payload_endpoint_bytes")
                },
            }
            if int(self.hierarchy.selected_dim) == 3:
                if any(len(row) != 6 for row in actual_stage_rows):
                    raise RuntimeError("Three-stage cost samples require six dispatch/combine A2A timings.")
                offline_samples.update(
                    {
                        "actual_stage1_a2a_ms": [row[0] + row[5] for row in actual_stage_rows],
                        "actual_stage2_a2a_ms": [row[1] + row[4] for row in actual_stage_rows],
                        "actual_stage3_a2a_ms": [row[2] + row[3] for row in actual_stage_rows],
                    }
                )
            elif int(self.hierarchy.selected_dim) == 2:
                if any(len(row) != 4 for row in actual_stage_rows):
                    raise RuntimeError("Two-stage cost samples require four dispatch/combine A2A timings.")
                offline_samples.update(
                    {
                        "actual_stage1_a2a_ms": [row[0] + row[3] for row in actual_stage_rows],
                        "actual_stage2_a2a_ms": [row[1] + row[2] for row in actual_stage_rows],
                    }
                )
            else:
                if any(len(row) != 2 for row in actual_stage_rows):
                    raise RuntimeError("One-stage cost samples require two dispatch/combine A2A timings.")
                offline_samples.update(
                    {
                        "actual_stage1_a2a_ms": [0.0 for _row in actual_stage_rows],
                        "actual_stage2_a2a_ms": [row[0] + row[1] for row in actual_stage_rows],
                    }
                )
            report["offline_scorer_samples"] = offline_samples
            sample_model = (
                "stage_payload_levels" if int(self.hierarchy.selected_dim) == 3 else "stage_payload_inter_intra"
            )
            report["sample_data"] = {
                "feature_values": {
                    **{name: [float(value) for value in values] for name, values in traffic_features.items()},
                    "peak_assignments": [float(value) for value in peak_assignments],
                },
                "compute": {
                    "assignments": paired_assignments,
                    "measured_ms": paired_compute,
                    "predicted_ms": [compute_slope * value + compute_constant for value in paired_assignments],
                },
                "communication_region": {
                    "feature_model": sample_model,
                    "measured_ms": [float(value) for value in actual_communication],
                    "predicted_ms": self._predict_linear_model(
                        model_rows[sample_model],
                        *self._cost_model_verify_feature_coefficients["communication"][sample_model],
                    ),
                },
                "joint_moe_region": {
                    "feature_model": sample_model,
                    "measured_ms": [float(value) for value in actual_joint],
                    "predicted_ms": self._predict_linear_model(
                        joint_model_rows[sample_model],
                        *self._cost_model_verify_feature_coefficients["joint"][sample_model],
                    ),
                },
            }
            if actual_raw_a2a and "network_joint" in self._cost_model_verify_feature_coefficients:
                report["sample_data"]["network_joint"] = {
                    "feature_model": sample_model,
                    "measured_ms": [
                        float(raw_a2a + compute)
                        for raw_a2a, compute in zip(
                            actual_raw_a2a,
                            actual_compute,
                            strict=True,
                        )
                    ],
                    "predicted_ms": self._predict_linear_model(
                        joint_model_rows[sample_model],
                        *self._cost_model_verify_feature_coefficients["network_joint"][sample_model],
                    ),
                }
        if actual_raw_a2a:
            report["raw_a2a_ms"] = {
                "min": min(actual_raw_a2a),
                "max": max(actual_raw_a2a),
                "mean": sum(actual_raw_a2a) / len(actual_raw_a2a),
                "stage_names": list(observations.get("actual_stage_a2a_names", ())),
            }
        level_names = list(observations.get("communication_level_names", ()))
        source_send_rows = list(observations.get("source_send_maxima", ()))
        destination_receive_rows = list(observations.get("destination_receive_maxima", ()))
        if level_names and source_send_rows and destination_receive_rows:
            report["communication_bottlenecks"] = {
                level: {
                    "source_send_min": min(float(row[level_index]) for row in source_send_rows),
                    "source_send_max": max(float(row[level_index]) for row in source_send_rows),
                    "source_send_mean": sum(float(row[level_index]) for row in source_send_rows)
                    / len(source_send_rows),
                    "destination_receive_min": min(float(row[level_index]) for row in destination_receive_rows),
                    "destination_receive_max": max(float(row[level_index]) for row in destination_receive_rows),
                    "destination_receive_mean": sum(float(row[level_index]) for row in destination_receive_rows)
                    / len(destination_receive_rows),
                    "source_dominant_samples": sum(
                        float(source_row[level_index]) > float(destination_row[level_index])
                        for source_row, destination_row in zip(
                            source_send_rows,
                            destination_receive_rows,
                            strict=True,
                        )
                    ),
                }
                for level_index, level in enumerate(level_names)
            }
        self._cost_model_reports[int(step)] = report
        self._record_cost_model_report(phase, report)
        for layer in layers:
            layer.cost_model_timings = [timing for timing in layer.cost_model_timings if int(timing.step) != int(step)]
        if int(step) == validation_end_step:
            self._cost_model_verify_complete = True
        self.latest_pair = "none"
        return self.latest_pair

    def finalize_auto_calibration(
        self,
        *,
        trainer_step: int,
        local_timing_rows: Sequence[dict[str, Any]],
    ) -> None:
        """Build and install a report-only planner artifact inside the training job."""

        if not self._auto_calibration or self._auto_calibration_finalized:
            return
        schedule_end = int(self._online_freeze_calibration_step) + int(self._cost_model_validation_steps) + 1
        if int(trainer_step) < schedule_end:
            return

        gathered_rows: list[Any] = [None for _ in range(self.ep_size)]
        if self.ep_group is not None and self.ep_size > 1:
            dist.all_gather_object(gathered_rows, list(local_timing_rows), group=self.ep_group)
        else:
            gathered_rows[0] = list(local_timing_rows)

        artifact: dict[str, Any] | None = None
        runtime_perf_model_sha256 = ""
        local_error = ""
        try:
            import hashlib

            from .placemoe.calibration import (
                ModelCalibrationSchedule,
                build_planner_calibration_artifact,
                sha256_path,
                summarize_phase_timing_rows,
            )

            timing_rows = [row for rank_rows in gathered_rows for row in rank_rows]
            expected_timing_steps = range(
                int(self._online_freeze_calibration_step) + 1,
                schedule_end + 1,
            )
            phase_summary = summarize_phase_timing_rows(
                timing_rows,
                expected_ranks=range(self.ep_size),
                expected_steps=expected_timing_steps,
            )
            report_steps = range(
                int(self._online_freeze_calibration_step),
                int(self._online_freeze_calibration_step) + int(self._cost_model_validation_steps) + 1,
            )
            missing_reports = [step for step in report_steps if step not in self._cost_model_reports]
            if missing_reports:
                raise RuntimeError(f"missing in-training cost-model reports for steps {missing_reports}")
            training_log_lines = []
            for index, step in enumerate(report_steps):
                phase = "calibration" if index == 0 else "validation"
                training_log_lines.append(
                    f"HierMoE cost model {phase} report: " + json.dumps(self._cost_model_reports[step], sort_keys=True)
                )
            training_log_text = "\n".join(training_log_lines) + "\n"

            runtime_perf_model_path = Path(self._auto_calibration_runtime_perf_model_path).expanduser().resolve()
            if not runtime_perf_model_path.is_file():
                raise RuntimeError(
                    "PlaceMoE in-training calibration requires a readable runtime performance model, "
                    f"got {runtime_perf_model_path}."
                )
            runtime_perf_model = json.loads(runtime_perf_model_path.read_text(encoding="utf-8"))
            if not isinstance(runtime_perf_model, dict):
                raise RuntimeError("runtime performance model must contain a JSON object")
            runtime_perf_model_sha256 = sha256_path(runtime_perf_model_path)
            calibration_config = settings._PLACEMOE_RUNTIME_CONFIG.calibration
            model_id = str(calibration_config.expected_scope.get("model_id") or "runtime-model")
            training_config = {
                "model": {"model_path": model_id},
                "train": {
                    "accelerator": {"ep_size": self.ep_size},
                    "hiermoe": {"hierarchy_group_sizes": list(self.hierarchy.group_sizes)},
                },
            }
            schedule = ModelCalibrationSchedule(
                warmup_steps=int(calibration_config.warmup_steps),
                validation_steps=int(calibration_config.validation_steps),
            )
            artifact = build_planner_calibration_artifact(
                training_config=training_config,
                runtime_perf_model=runtime_perf_model,
                runtime_perf_model_sha256=runtime_perf_model_sha256,
                training_log_text=training_log_text,
                training_log_sha256=hashlib.sha256(training_log_text.encode("utf-8")).hexdigest(),
                phase_timing_summaries=[phase_summary],
                ranks_per_node=int(self.hierarchy.local_world_size),
                schedule=schedule,
                model_id=model_id,
            )
            artifact["provenance"]["generation_mode"] = "in_training"
        except Exception as error:  # keep ranks in lockstep before surfacing a structural failure
            local_error = str(error)

        build_states: list[Any] = [None for _ in range(self.ep_size)]
        build_state = {
            "ep_rank": int(self.ep_rank),
            "error": local_error,
            "runtime_perf_model_sha256": runtime_perf_model_sha256,
            "artifact": artifact if int(self.ep_rank) == 0 else None,
        }
        if self.ep_group is not None and self.ep_size > 1:
            dist.all_gather_object(build_states, build_state, group=self.ep_group)
        else:
            build_states[0] = build_state

        build_errors = [
            f"ep_rank={state['ep_rank']}: {state['error']}"
            for state in build_states
            if isinstance(state, dict) and state.get("error")
        ]
        if build_errors:
            raise RuntimeError("PlaceMoE in-training calibration failed: " + "; ".join(build_errors))
        runtime_hashes = {
            str(state["runtime_perf_model_sha256"])
            for state in build_states
            if isinstance(state, dict) and state.get("runtime_perf_model_sha256")
        }
        if len(runtime_hashes) != 1:
            raise RuntimeError(
                "PlaceMoE in-training calibration requires the same runtime performance model on every EP rank; "
                f"observed SHA-256 values {sorted(runtime_hashes)}."
            )
        authoritative_artifacts = [
            state["artifact"]
            for state in build_states
            if isinstance(state, dict) and int(state.get("ep_rank", -1)) == 0 and state.get("artifact") is not None
        ]
        if len(authoritative_artifacts) != 1:
            raise RuntimeError("PlaceMoE in-training calibration did not receive one authoritative artifact.")
        artifact_payload = json.dumps(authoritative_artifacts[0], indent=2, sort_keys=True, allow_nan=False) + "\n"
        artifact = json.loads(artifact_payload)

        calibration_config = settings._PLACEMOE_RUNTIME_CONFIG.calibration
        output = Path(calibration_config.output)
        local_world_size = int(self.hierarchy.local_world_size)
        is_node_leader = self.ep_rank % local_world_size == 0
        temporary = output.with_name(f".{output.name}.rank{self.ep_rank}.tmp")
        previous_contents: bytes | None = None
        previous_existed = False
        local_error = ""
        if is_node_leader:
            try:
                output.parent.mkdir(parents=True, exist_ok=True)
                previous_existed = output.exists()
                if previous_existed:
                    previous_contents = output.read_bytes()
                temporary.write_text(artifact_payload, encoding="utf-8")
            except Exception as error:
                local_error = str(error)

        device = self._pipeline_device(next(iter(self.layers.values())))
        failure = torch.tensor([bool(local_error)], dtype=torch.int32, device=device)
        if self.ep_group is not None and self.ep_size > 1:
            dist.all_reduce(failure, op=dist.ReduceOp.MAX, group=self.ep_group)
        if int(failure.item()) != 0:
            if is_node_leader:
                temporary.unlink(missing_ok=True)
            if local_error:
                logger.error("PlaceMoE in-training calibration staging failed: %s", local_error)
            raise RuntimeError("PlaceMoE in-training calibration staging failed; existing artifacts are unchanged.")

        committed = False
        local_error = ""
        if is_node_leader:
            try:
                os.replace(temporary, output)
                committed = True
            except Exception as error:
                local_error = str(error)

        failure.fill_(bool(local_error))
        if self.ep_group is not None and self.ep_size > 1:
            dist.all_reduce(failure, op=dist.ReduceOp.MAX, group=self.ep_group)
        if int(failure.item()) != 0:
            rollback_error = ""
            if is_node_leader:
                try:
                    temporary.unlink(missing_ok=True)
                    if committed:
                        if previous_existed:
                            rollback = output.with_name(f".{output.name}.rank{self.ep_rank}.rollback")
                            assert previous_contents is not None
                            rollback.write_bytes(previous_contents)
                            os.replace(rollback, output)
                        else:
                            output.unlink(missing_ok=True)
                except Exception as error:
                    rollback_error = str(error)
            failure.fill_(bool(rollback_error))
            if self.ep_group is not None and self.ep_size > 1:
                dist.all_reduce(failure, op=dist.ReduceOp.MAX, group=self.ep_group)
            if local_error:
                logger.error("PlaceMoE in-training calibration commit failed: %s", local_error)
            if rollback_error:
                logger.error("PlaceMoE in-training calibration rollback failed: %s", rollback_error)
            suffix = " Some ranks could not restore their previous artifact." if int(failure.item()) != 0 else ""
            raise RuntimeError("PlaceMoE in-training calibration commit failed; rolled back node outputs." + suffix)

        coefficients = artifact["coefficients"]
        self._hot_update_calibration = PlaceMoECalibration(
            inter_ms_per_byte=float(coefficients["inter_ms_per_byte"]),
            intra_ms_per_byte=float(coefficients["intra_ms_per_byte"]),
            route_ms_per_assignment=float(coefficients["route_ms_per_assignment"]),
            communication_multiplier=float(coefficients["communication_multiplier"]),
            compute_ms_per_assignment=float(coefficients["compute_ms_per_assignment"]),
            compute_multiplier=float(coefficients["compute_multiplier"]),
        )
        self._auto_calibration_finalized = True
        self._cost_model_verify = False
        os.environ["VEOMNI_PLACEMOE_AUTO_CALIBRATION"] = "0"
        from .all_to_all import configure_hiermoe_internal_timing

        configure_hiermoe_internal_timing(_env_flag("VEOMNI_HIERMOE_INTERNAL_TIMING"))
        validation = artifact["held_out_validation"]
        self._auto_calibration_compute_mape = float(validation["compute"]["mape_percent"])
        self._auto_calibration_communication_mape = float(validation["communication"]["mape_percent"])
        self._auto_calibration_joint_mape = float(validation["joint"]["mape_percent"])
        logger.info_rank0(
            "PlaceMoE in-training calibration installed coefficients and wrote %s. "
            "MAPE compute=%.3f%% communication=%.3f%% joint=%.3f%%.",
            settings._PLACEMOE_RUNTIME_CONFIG.calibration.output,
            validation["compute"]["mape_percent"],
            validation["communication"]["mape_percent"],
            validation["joint"]["mape_percent"],
        )

    @torch.no_grad()
    def _prepare_online_freeze_calibrations(
        self,
        layers: Sequence[ExpertLayerState],
        *,
        step: int,
        started: float,
    ) -> None:
        """Validate offline traffic coefficients and fit online GEMM cost."""

        ordered_layers = sorted(layers, key=lambda value: value.key)
        records = [
            (layer, layer.pending_timing)
            for layer in ordered_layers
            if layer.pending_timing is not None
            and layer.pending_timing.step == int(step)
            and self._events_ready(layer.pending_timing)
        ]
        if len(records) != len(ordered_layers):
            self._accumulate_metric(
                "hiermoe/placement_calibration_ms",
                (time.perf_counter() - started) * 1000.0,
            )
            return

        has_full_samples = all(
            any(int(timing.step) == int(step) for timing in layer.cost_model_timings) for layer in ordered_layers
        )
        communication_samples = 0
        compute_samples: list[tuple[float, float]]
        communication_diagnostics: dict[str, float] | None = None
        compute_diagnostics: dict[str, float] | None = None
        joint_diagnostics: dict[str, float] | None = None
        traffic_scale = 1.0
        traffic_constant = self._online_freeze_traffic_intercept_ms
        traffic_predictors: list[float] = []
        if has_full_samples:
            observations = self._cost_model_step_observations(ordered_layers, step=int(step))
            traffic_features = dict(observations["traffic_features"])
            stage1 = [float(value) for value in traffic_features["stage1_payload_endpoint_bytes"]]
            stage2 = [float(value) for value in traffic_features["stage2_payload_endpoint_bytes"]]
            peak_assignments = [float(value) for value in observations["peak_assignments"]]
            actual_communication = [float(value) for value in observations["actual_communication_ms"]]
            route_coefficient = (
                self._online_freeze_route_ms_per_assignment if self._online_freeze_cost_mode == "joint" else 0.0
            )
            traffic_predictors = [
                self._online_freeze_inter_ms_per_byte * inter_bytes
                + self._online_freeze_intra_ms_per_byte * intra_bytes
                + route_coefficient * assignments
                for inter_bytes, intra_bytes, assignments in zip(
                    stage1,
                    stage2,
                    peak_assignments,
                    strict=True,
                )
            ]
            traffic_scale = self._fit_positive_through_origin(
                [
                    (predictor, max(0.0, actual - traffic_constant))
                    for predictor, actual in zip(
                        traffic_predictors,
                        actual_communication,
                        strict=True,
                    )
                ]
            )
            if traffic_scale <= 0.0:
                traffic_scale = self._fit_positive_through_origin(
                    list(zip(traffic_predictors, actual_communication, strict=True))
                )
                traffic_constant = 0.0
            predicted_communication = [
                traffic_scale * predictor + traffic_constant for predictor in traffic_predictors
            ]
            communication_diagnostics = self._cost_model_diagnostics(
                actual_communication,
                predicted_communication,
            )
            communication_samples = len(actual_communication)
            compute_samples = [
                (float(assignments), float(compute_ms))
                for assignments, compute_ms in zip(
                    observations["paired_assignments"],
                    observations["paired_compute_ms"],
                    strict=True,
                )
                if float(assignments) > 0.0
            ]
        else:
            # Unit tests and legacy callers may provide one pending sample per
            # layer without the full microbatch route capture.
            compute_samples = []
            for _layer, timing in records:
                assert timing is not None
                local_values = torch.stack(
                    (
                        timing.local_assignment_count.to(dtype=torch.float32),
                        torch.tensor(
                            timing.compute_start.elapsed_time(timing.compute_end),
                            dtype=torch.float32,
                            device=timing.local_assignment_count.device,
                        ),
                    )
                )
                if self.ep_group is not None and self.ep_size > 1:
                    gathered_flat = torch.empty(
                        (self.ep_size * int(local_values.numel()),),
                        dtype=local_values.dtype,
                        device=local_values.device,
                    )
                    dist.all_gather_into_tensor(gathered_flat, local_values, group=self.ep_group)
                    gathered = gathered_flat.view(self.ep_size, int(local_values.numel()))
                else:
                    gathered = local_values.view(1, -1)
                compute_samples.extend(
                    (float(row[0].item()), float(row[1].item())) for row in gathered if float(row[0].item()) > 0.0
                )

        compute_slope, compute_constant = self._fit_nonnegative_compute_model(compute_samples)
        if compute_slope <= 0.0:
            compute_slope = self._fit_positive_through_origin(compute_samples)
            compute_constant = 0.0
        compute_diagnostics = self._cost_model_diagnostics(
            [target for _assignments, target in compute_samples],
            [compute_slope * assignments + compute_constant for assignments, _target in compute_samples],
        )

        if has_full_samples:
            actual_communication = [float(value) for value in observations["actual_communication_ms"]]
            actual_compute = [float(value) for value in observations["actual_compute_ms"]]
            predicted_communication = [
                traffic_scale * predictor + traffic_constant for predictor in traffic_predictors
            ]
            predicted_compute = [
                compute_slope * float(assignments) + compute_constant
                for assignments in observations["peak_assignments"]
            ]
            joint_diagnostics = self._cost_model_diagnostics(
                [
                    communication + compute
                    for communication, compute in zip(actual_communication, actual_compute, strict=True)
                ],
                [
                    communication + compute
                    for communication, compute in zip(predicted_communication, predicted_compute, strict=True)
                ],
            )

        if self._online_freeze_cost_mode == "joint":
            planner_compute_slope = compute_slope
            planner_compute_constant = compute_constant
        else:
            planner_compute_slope = 0.0
            planner_compute_constant = 0.0

        for layer, timing in records:
            assert timing is not None
            layer.planner_calibration = _PlannerCalibration(
                source_step=timing.step,
                communication_scale=traffic_scale,
                forward_compute_per_assignment=planner_compute_slope,
                forward_compute_constant=planner_compute_constant,
            )
            layer.pending_timing = None
            layer.cost_model_timings = [sample for sample in layer.cost_model_timings if int(sample.step) != int(step)]

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._accumulate_metric("hiermoe/placement_calibration_ms", elapsed_ms)
        self._accumulate_metric("hiermoe/placement_calibrated_layers", len(records))
        self._accumulate_metric("hiermoe/placement_calibration_communication_samples", communication_samples)
        self._accumulate_metric("hiermoe/placement_calibration_compute_samples", len(compute_samples))
        self._accumulate_metric("hiermoe/placement_traffic_online_scale", traffic_scale)
        self._accumulate_metric("hiermoe/placement_traffic_online_constant_ms", traffic_constant)
        if traffic_predictors:
            self._accumulate_metric("hiermoe/placement_traffic_predictor_min_ms", min(traffic_predictors))
            self._accumulate_metric("hiermoe/placement_traffic_predictor_max_ms", max(traffic_predictors))
        self._accumulate_metric("hiermoe/placement_forward_compute_ms_per_assignment", compute_slope)
        self._accumulate_metric("hiermoe/placement_forward_compute_constant_ms", compute_constant)
        self._accumulate_metric(
            "hiermoe/placement_full_compute_ms_per_assignment",
            self._online_freeze_compute_ratio * compute_slope,
        )
        if communication_diagnostics is not None:
            self._accumulate_metric(
                "hiermoe/placement_calibration_communication_r2",
                communication_diagnostics["r_squared"],
            )
            self._accumulate_metric(
                "hiermoe/placement_calibration_communication_mape_percent",
                communication_diagnostics["mape_percent"],
            )
        if compute_diagnostics is not None:
            self._accumulate_metric(
                "hiermoe/placement_calibration_compute_r2",
                compute_diagnostics["r_squared"],
            )
            self._accumulate_metric(
                "hiermoe/placement_calibration_compute_mape_percent",
                compute_diagnostics["mape_percent"],
            )
        if joint_diagnostics is not None:
            self._accumulate_metric(
                "hiermoe/placement_calibration_joint_r2",
                joint_diagnostics["r_squared"],
            )
            self._accumulate_metric(
                "hiermoe/placement_calibration_joint_mape_percent",
                joint_diagnostics["mape_percent"],
            )
        if communication_diagnostics is not None and communication_diagnostics["mape_percent"] > 5.0:
            logger.warning_rank0(
                "Online-freeze per-layer E2E traffic timings are too noisy for action-level validation: "
                "MAPE=%.3f%% exceeds 5%%; "
                f"R2={communication_diagnostics['r_squared']:.6f}, "
                f"RMSE={communication_diagnostics['rmse_ms']:.3f} ms, "
                f"max_abs={communication_diagnostics['max_abs_error_ms']:.3f} ms, "
                f"online_scale={traffic_scale:.9g}, intercept={traffic_constant:.3f} ms, "
                f"predictor_range="
                f"[{min(traffic_predictors, default=0.0):.3f}, "
                f"{max(traffic_predictors, default=0.0):.3f}] ms. "
                "Keeping the offline multi-layout feature ratios and validating the frozen winner by E2E.",
                communication_diagnostics["mape_percent"],
            )

    @torch.no_grad()
    def prepare_calibrations(self, step: int) -> None:
        started = time.perf_counter()
        if self.expert_swap_selector == "hiermoe_greedy_cover_p1":
            uncalibrated = [layer for layer in self.layers.values() if layer.planner_calibration is None]
            if not self.layers:
                return
            consensus_device = _local_tensor_view(next(iter(self.layers.values())).primary_parameter).device
            all_need_calibration, need_state_agrees = _placement_group_boolean_consensus(
                bool(uncalibrated),
                device=consensus_device,
                ep_size=self.ep_size,
                ep_group=self.ep_group,
            )
            if not need_state_agrees:
                raise RuntimeError("HierMoE planner calibration state differs across the EP group.")
            if not all_need_calibration:
                return
            local_ready = not any(
                layer.pending_timing is None
                or layer.pending_timing.step > int(step)
                or not self._events_ready(layer.pending_timing)
                for layer in uncalibrated
            )
            all_ready, _ready_state_agrees = _placement_group_boolean_consensus(
                local_ready,
                device=consensus_device,
                ep_size=self.ep_size,
                ep_group=self.ep_group,
            )
            if not all_ready:
                self._accumulate_metric(
                    "hiermoe/placement_calibration_ms",
                    (time.perf_counter() - started) * 1000.0,
                )
                return
            if self._online_freeze_cost_mode != "off":
                self._prepare_online_freeze_calibrations(
                    uncalibrated,
                    step=int(step),
                    started=started,
                )
                return
        elif not self.layer_calibration_enabled():
            return
        updated = 0
        greedy_records: list[tuple[ExpertLayerState, _PendingLayerTiming, float, float, float]] = []
        for layer_key in sorted(self.layers):
            layer = self.layers[layer_key]
            timing = layer.pending_timing
            if timing is None or timing.step > int(step) or not self._events_ready(timing):
                continue
            selected = timing.selected_experts
            planner = self._planner_for_layer(
                layer,
                communication_scale=1.0,
                forward_compute_per_assignment=1.0,
                forward_compute_constant=0.0,
            )
            copy_slots, _copy_mask = layer.copy_slots_for_device(selected.device)
            reference = planner.score_layout(
                selected,
                timing.slot_to_logical,
                source_ranks=self.ep_rank,
                owner_slots=layer.logical_to_physical,
                step=timing.step,
                layer_seed=zlib.crc32(layer.key.encode("utf-8")),
                max_copies=int(copy_slots.shape[1]),
            )
            values = torch.tensor(
                [
                    timing.dispatch_start.elapsed_time(timing.dispatch_end)
                    + timing.combine_start.elapsed_time(timing.combine_end),
                    timing.compute_start.elapsed_time(timing.compute_end),
                    timing.local_assignment_count,
                ],
                dtype=torch.float32,
                device=selected.device,
            )
            if self.ep_group is not None and self.ep_size > 1:
                dist.all_reduce(values, op=dist.ReduceOp.MAX, group=self.ep_group)
            communication_units = reference.communication_model_units
            peak_assignments = float(values[2].item())
            forward_communication_ms = float(values[0].item())
            forward_compute_ms = float(values[1].item())
            if communication_units <= 0.0 or peak_assignments <= 0.0:
                continue
            if self.expert_swap_selector == "hiermoe_greedy_cover_p1":
                # The greedy planner explicitly accounts for four communication
                # phases. Normalize the measured forward dispatch+combine pair
                # to the residual scale of one modeled phase.
                communication_scale = forward_communication_ms / (2.0 * communication_units)
            else:
                # CurrentRoutePlanner.communication_model_units already includes
                # all four communication phases.
                communication_scale = (2.0 * forward_communication_ms) / communication_units
            if not math.isfinite(communication_scale):
                continue
            if self.expert_swap_selector == "hiermoe_greedy_cover_p1":
                greedy_records.append((layer, timing, communication_scale, peak_assignments, forward_compute_ms))
                continue
            compute_scale = forward_compute_ms / peak_assignments
            if not math.isfinite(compute_scale):
                continue
            layer.planner_calibration = _PlannerCalibration(
                source_step=timing.step,
                communication_scale=communication_scale,
                forward_compute_per_assignment=compute_scale,
            )
            layer.pending_timing = None
            updated += 1
        if greedy_records:
            compute_scale, compute_constant = self._fit_nonnegative_compute_model(
                [
                    (peak_assignments, forward_compute_ms)
                    for _, _, _, peak_assignments, forward_compute_ms in greedy_records
                ]
            )
            for layer, timing, communication_scale, _peak_assignments, _forward_compute_ms in greedy_records:
                layer.planner_calibration = _PlannerCalibration(
                    source_step=timing.step,
                    communication_scale=communication_scale,
                    forward_compute_per_assignment=compute_scale,
                    forward_compute_constant=compute_constant,
                )
                layer.pending_timing = None
                updated += 1
            self._accumulate_metric("hiermoe/placement_compute_ms_per_assignment", compute_scale)
            self._accumulate_metric("hiermoe/placement_compute_constant_ms", compute_constant)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._accumulate_metric("hiermoe/placement_calibration_ms", elapsed_ms)
        self._accumulate_metric("hiermoe/placement_calibrated_layers", updated)

    def _cpu_exact_planner_for_layer(self, layer: ExpertLayerState) -> GreedyCommunicationPlanner:
        """Build the same exact scorer as the fixed-pipeline NPU backend on CPU."""

        return GreedyCommunicationPlanner(
            hierarchy=self.hierarchy,
            perf_model=self.perf_model,
            hidden_size=layer.latest_hidden_size,
            bytes_per_element=layer.latest_bytes_per_element,
            slots_per_rank=layer.num_local_experts,
            communication_scale=1.0,
            forward_compute_per_assignment=0.0,
            forward_compute_constant=0.0,
            smooth_max_gamma=self.smooth_max_gamma,
            reducer=None,
            candidate_chunk_size=settings._SWAP_COST_CHUNK_CANDIDATES,
            process_group=None,
            max_copies=self.greedy_max_copies_per_expert,
            assume_unique_routes=True,
            layer_parallel_streams=settings._GREEDY_LAYER_PARALLEL_STREAMS,
            adaptive_topk=False,
            adaptive_topk_initial=settings._GREEDY_ADAPTIVE_TOPK_INITIAL,
            adaptive_topk_strict_certificate=False,
            exact_primitive_topk=0,
            post_shortlist_compact_pair=False,
            exact_primitive_max_only=False,
        )
