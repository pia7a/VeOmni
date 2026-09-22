"""Communication cost statistics shared by calibration and route scoring."""

from __future__ import annotations

import torch

from .perf_model import HierMoEPerfModel
from .topology import Hierarchy
from .traffic import TrafficAccounting


GREEDY_COMMUNICATION_PHASE_MULTIPLIER = 4.0


class CalibrationCostModel(TrafficAccounting):
    """Count and score routes without constructing a candidate-search planner."""

    def __init__(
        self,
        *,
        hierarchy: Hierarchy,
        perf_model: HierMoEPerfModel,
        hidden_size: int,
        bytes_per_element: int,
        slots_per_rank: int,
        smooth_max_gamma: float,
    ) -> None:
        if smooth_max_gamma <= 0:
            raise ValueError("smooth_max_gamma must be positive.")
        self.hierarchy = hierarchy
        self.perf_model = perf_model
        self.hidden_size = int(hidden_size)
        self.bytes_per_element = int(bytes_per_element)
        self.slots_per_rank = int(slots_per_rank)
        self.smooth_max_gamma = float(smooth_max_gamma)
        self.communication_scale = 1.0

    def _communication_cost_details(
        self,
        packed_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        widths = self._count_widths()
        rows = packed_counts.split(widths, dim=1)
        rank_counts = rows[0]
        rank_max = rank_counts.max(dim=1).values
        dimensions = [
            self.perf_model.a2a.alpha + float(self.ep_size * self.payload_bytes) * rank_max * self.perf_model.a2a.beta
        ]
        max_dim = max(1, int(self.hierarchy.selected_dim))
        for dim in range(2, max_dim + 1):
            total = torch.zeros_like(rank_max)
            previous_size = 1
            for level_index, raw_size in enumerate(self.hierarchy.group_sizes[: dim - 1]):
                size = int(raw_size)
                link = self.perf_model.inter[min(level_index, len(self.perf_model.inter) - 1)]
                group_max = rows[level_index + 1].max(dim=1).values
                scale = float((size / previous_size) * self.payload_bytes)
                total = total + link.alpha + scale * group_max * link.beta
                previous_size = size
            intra_scale = float((self.ep_size / previous_size) * self.payload_bytes)
            total = total + self.perf_model.intra.alpha + intra_scale * rank_max * self.perf_model.intra.beta
            dimensions.append(total)
        per_dim = torch.stack(dimensions, dim=1)
        communication_units = torch.logsumexp(per_dim * self.smooth_max_gamma, dim=1) / self.smooth_max_gamma
        communication = GREEDY_COMMUNICATION_PHASE_MULTIPLIER * self.communication_scale * communication_units
        return communication, communication_units, rank_counts.argmax(dim=1), per_dim.argmax(dim=1) + 1

    def _source_aware_communication_cost_details(
        self,
        source_packed_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score duplicate-free traffic using both source and destination bottlenecks.

        ``source_packed_counts`` preserves the source-rank dimension that is
        normally removed by the placement-statistics reduction. At every
        hierarchy level, the effective payload is the larger of the maximum
        outgoing source-group payload and maximum incoming destination-group
        payload.
        """

        if source_packed_counts.ndim != 3:
            raise ValueError(
                "Source-aware communication counts must have shape "
                f"[source_rank, batch, packed_width], got {tuple(source_packed_counts.shape)}."
            )
        if int(source_packed_counts.shape[0]) != self.ep_size:
            raise ValueError(f"Expected {self.ep_size} source ranks, got {int(source_packed_counts.shape[0])}.")
        widths = self._count_widths()
        if int(source_packed_counts.shape[2]) != sum(widths):
            raise ValueError(f"Expected packed width {sum(widths)}, got {int(source_packed_counts.shape[2])}.")

        source_rows = source_packed_counts.split(widths, dim=2)
        receive_rows = source_packed_counts.sum(dim=0).split(widths, dim=1)
        receive_maxima = [row.max(dim=1).values for row in receive_rows]

        rank_send_max = source_rows[0].sum(dim=2).max(dim=0).values
        send_maxima = [rank_send_max]
        for level_index, raw_size in enumerate(
            self.hierarchy.group_sizes[: max(0, int(self.hierarchy.selected_dim) - 1)]
        ):
            size = int(raw_size)
            num_source_groups = self.ep_size // size
            source_group_payload = (
                source_rows[level_index + 1]
                .reshape(num_source_groups, size, source_packed_counts.shape[1], widths[level_index + 1])
                .sum(dim=1)
                .sum(dim=2)
            )
            send_maxima.append(source_group_payload.max(dim=0).values)

        bottleneck_rows = [
            torch.maximum(send_max, receive_max)
            for send_max, receive_max in zip(send_maxima, receive_maxima, strict=True)
        ]
        rank_bottleneck = bottleneck_rows[0]
        dimensions = [
            self.perf_model.a2a.alpha
            + float(self.ep_size * self.payload_bytes) * rank_bottleneck * self.perf_model.a2a.beta
        ]
        max_dim = max(1, int(self.hierarchy.selected_dim))
        for dim in range(2, max_dim + 1):
            total = torch.zeros_like(rank_bottleneck)
            previous_size = 1
            for level_index, raw_size in enumerate(self.hierarchy.group_sizes[: dim - 1]):
                size = int(raw_size)
                link = self.perf_model.inter[min(level_index, len(self.perf_model.inter) - 1)]
                scale = float((size / previous_size) * self.payload_bytes)
                total = total + link.alpha + scale * bottleneck_rows[level_index + 1] * link.beta
                previous_size = size
            intra_scale = float((self.ep_size / previous_size) * self.payload_bytes)
            total = total + self.perf_model.intra.alpha + intra_scale * rank_bottleneck * self.perf_model.intra.beta
            dimensions.append(total)
        per_dim = torch.stack(dimensions, dim=1)
        communication_units = torch.logsumexp(per_dim * self.smooth_max_gamma, dim=1) / self.smooth_max_gamma
        communication = GREEDY_COMMUNICATION_PHASE_MULTIPLIER * self.communication_scale * communication_units
        return (
            communication,
            communication_units,
            torch.stack(send_maxima, dim=1),
            torch.stack(receive_maxima, dim=1),
            per_dim.argmax(dim=1) + 1,
        )
