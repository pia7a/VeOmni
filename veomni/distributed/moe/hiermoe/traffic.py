"""Exact hierarchical token counts and calibrated endpoint costs.

Shared by route replay and runtime calibration. Communication deduplicates tokens
per destination group; compute counts every token/expert assignment (paper II-C).
These methods preserve the original counting and floating-point operation order.
"""

from __future__ import annotations

import torch

from .perf_model import HierMoEPerfModel


class TrafficAccounting:
    """Counting primitives; the caller supplies topology and cost coefficients."""

    @property
    def ep_size(self) -> int:
        return int(self.hierarchy.ep_size)

    @property
    def payload_bytes(self) -> int:
        return self.hidden_size * self.bytes_per_element

    def _count_widths(self) -> tuple[int, ...]:
        widths = [self.ep_size]
        for size in self.hierarchy.group_sizes[: max(0, int(self.hierarchy.selected_dim) - 1)]:
            size = int(size)
            if size <= 0 or self.ep_size % size != 0:
                raise ValueError(f"Invalid hierarchy group size {size} for EP size {self.ep_size}.")
            widths.append(self.ep_size // size)
        return tuple(widths)

    def _local_packed_counts(self, physical_slots: torch.Tensor) -> torch.Tensor:
        physical = physical_slots
        if physical.ndim == 2:
            physical = physical.unsqueeze(0)
        ranks = torch.div(physical, self.slots_per_rank, rounding_mode="floor")
        batch, num_tokens, top_k = ranks.shape
        rows: list[torch.Tensor] = []

        rank_hits = torch.zeros((batch * num_tokens, self.ep_size), dtype=torch.bool, device=ranks.device)
        rank_hits.scatter_(1, ranks.reshape(batch * num_tokens, top_k), True)
        rows.append(rank_hits.view(batch, num_tokens, self.ep_size).sum(dim=1).to(torch.float32))
        for size in self.hierarchy.group_sizes[: max(0, int(self.hierarchy.selected_dim) - 1)]:
            size = int(size)
            groups = torch.div(ranks, size, rounding_mode="floor")
            num_groups = self.ep_size // size
            group_hits = torch.zeros((batch * num_tokens, num_groups), dtype=torch.bool, device=ranks.device)
            group_hits.scatter_(1, groups.reshape(batch * num_tokens, top_k), True)
            rows.append(group_hits.view(batch, num_tokens, num_groups).sum(dim=1).to(torch.float32))
        return torch.cat(rows, dim=1)

    def _local_packed_assignment_counts(self, physical_slots: torch.Tensor) -> torch.Tensor:
        """Count non-deduplicated assignments for every hierarchy destination."""

        physical = physical_slots
        if physical.ndim == 2:
            physical = physical.unsqueeze(0)
        ranks = torch.div(physical, self.slots_per_rank, rounding_mode="floor").to(torch.long)
        batch = int(ranks.shape[0])
        flat_ranks = ranks.reshape(batch, -1)
        ones = torch.ones_like(flat_ranks, dtype=torch.float32)
        rows: list[torch.Tensor] = []

        rank_counts = torch.zeros((batch, self.ep_size), dtype=torch.float32, device=ranks.device)
        rank_counts.scatter_add_(1, flat_ranks, ones)
        rows.append(rank_counts)
        for raw_size in self.hierarchy.group_sizes[: max(0, int(self.hierarchy.selected_dim) - 1)]:
            size = int(raw_size)
            groups = torch.div(flat_ranks, size, rounding_mode="floor")
            num_groups = self.ep_size // size
            group_counts = torch.zeros((batch, num_groups), dtype=torch.float32, device=ranks.device)
            group_counts.scatter_add_(1, groups, ones)
            rows.append(group_counts)
        return torch.cat(rows, dim=1)

    @staticmethod
    def _stage_traffic_features(
        unique_matrix: torch.Tensor,
        assignment_matrix: torch.Tensor,
        *,
        hidden_bytes: int,
        metadata_bytes: int,
    ) -> dict[str, torch.Tensor]:
        """Return exact endpoint/edge features for one A2A stage.

        Both matrices have shape ``[batch, group, source, destination]``.
        Dispatch sends unique hidden rows plus assignment metadata, while
        combine reverses only the unique hidden rows.
        """

        if unique_matrix.shape != assignment_matrix.shape or unique_matrix.ndim != 4:
            raise ValueError("Stage traffic matrices must have the same [batch, group, source, destination] shape.")
        unique = unique_matrix.to(torch.float32)
        assignments = assignment_matrix.to(torch.float32)
        unique_send = unique.sum(dim=3)
        unique_receive = unique.sum(dim=2)
        assignment_send = assignments.sum(dim=3)
        assignment_receive = assignments.sum(dim=2)

        dispatch_send_bytes = float(hidden_bytes) * unique_send + float(metadata_bytes) * assignment_send
        dispatch_receive_bytes = float(hidden_bytes) * unique_receive + float(metadata_bytes) * assignment_receive
        dispatch_endpoint_bytes = torch.maximum(
            dispatch_send_bytes.amax(dim=(1, 2)),
            dispatch_receive_bytes.amax(dim=(1, 2)),
        )
        unique_endpoint = torch.maximum(
            unique_send.amax(dim=(1, 2)),
            unique_receive.amax(dim=(1, 2)),
        )
        combine_endpoint_bytes = float(hidden_bytes) * unique_endpoint

        dispatch_edge_bytes = (float(hidden_bytes) * unique + float(metadata_bytes) * assignments).amax(dim=(1, 2, 3))
        combine_edge_bytes = float(hidden_bytes) * unique.amax(dim=(1, 2, 3))
        active_send_peers = (unique > 0).sum(dim=3, dtype=torch.float32).amax(dim=(1, 2))
        active_receive_peers = (unique > 0).sum(dim=2, dtype=torch.float32).amax(dim=(1, 2))
        if int(unique.shape[2]) != int(unique.shape[3]):
            raise ValueError("A2A stage traffic matrices must have equal source and destination widths.")
        diagonal = torch.eye(
            int(unique.shape[2]),
            dtype=torch.bool,
            device=unique.device,
        ).view(1, 1, int(unique.shape[2]), int(unique.shape[3]))
        remote_unique = unique.masked_fill(diagonal, 0.0)
        remote_assignments = assignments.masked_fill(diagonal, 0.0)
        remote_unique_send = remote_unique.sum(dim=3)
        remote_unique_receive = remote_unique.sum(dim=2)
        remote_assignment_send = remote_assignments.sum(dim=3)
        remote_assignment_receive = remote_assignments.sum(dim=2)
        remote_dispatch_endpoint_bytes = torch.maximum(
            (float(hidden_bytes) * remote_unique_send + float(metadata_bytes) * remote_assignment_send).amax(
                dim=(1, 2)
            ),
            (float(hidden_bytes) * remote_unique_receive + float(metadata_bytes) * remote_assignment_receive).amax(
                dim=(1, 2)
            ),
        )
        remote_unique_endpoint = torch.maximum(
            remote_unique_send.amax(dim=(1, 2)),
            remote_unique_receive.amax(dim=(1, 2)),
        )
        remote_dispatch_edge_bytes = (
            float(hidden_bytes) * remote_unique + float(metadata_bytes) * remote_assignments
        ).amax(dim=(1, 2, 3))
        remote_combine_edge_bytes = float(hidden_bytes) * remote_unique.amax(dim=(1, 2, 3))
        diagonal_unique = unique.diagonal(dim1=2, dim2=3)
        diagonal_assignments = assignments.diagonal(dim1=2, dim2=3)
        self_endpoint_bytes = (
            2.0 * float(hidden_bytes) * diagonal_unique + float(metadata_bytes) * diagonal_assignments
        ).amax(dim=(1, 2))
        return {
            "unique_endpoint_tokens": unique_endpoint,
            "full_endpoint_bytes": dispatch_endpoint_bytes + combine_endpoint_bytes,
            "full_edge_bytes": dispatch_edge_bytes + combine_edge_bytes,
            "remote_full_endpoint_bytes": (
                remote_dispatch_endpoint_bytes + float(hidden_bytes) * remote_unique_endpoint
            ),
            "remote_full_edge_bytes": remote_dispatch_edge_bytes + remote_combine_edge_bytes,
            "self_endpoint_bytes": self_endpoint_bytes,
            "full_total_bytes": (
                2.0 * float(hidden_bytes) * unique.sum(dim=(1, 2, 3))
                + float(metadata_bytes) * assignments.sum(dim=(1, 2, 3))
            ),
            "max_active_peers": torch.maximum(active_send_peers, active_receive_peers),
        }

    def _hierarchical3d_traffic_features(
        self,
        source_unique_counts: torch.Tensor,
        source_assignment_counts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        intra_size = int(self.hierarchy.group_sizes[0])
        mid_size = int(self.hierarchy.group_sizes[1])
        if intra_size <= 1 or mid_size <= intra_size or mid_size % intra_size or self.ep_size % mid_size:
            raise ValueError(f"Invalid three-stage hierarchy {(intra_size, mid_size, self.ep_size)}.")
        num_mid_groups = self.ep_size // mid_size
        groups_per_mid = mid_size // intra_size
        widths = self._count_widths()
        expected_widths = (self.ep_size, self.ep_size // intra_size, num_mid_groups)
        if widths[:3] != expected_widths:
            raise ValueError(f"Unexpected three-stage packed widths {widths}.")
        unique_rank, unique_intra, unique_mid = source_unique_counts.split(widths, dim=2)[:3]
        assignment_rank, assignment_intra, assignment_mid = source_assignment_counts.split(widths, dim=2)[:3]
        batch = int(source_unique_counts.shape[1])

        def matrices(
            rank_counts: torch.Tensor,
            intra_counts: torch.Tensor,
            mid_counts: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            stage1 = (
                mid_counts.reshape(num_mid_groups, mid_size, batch, num_mid_groups).permute(2, 1, 0, 3).contiguous()
            )
            stage2 = (
                intra_counts.reshape(
                    num_mid_groups,
                    groups_per_mid,
                    intra_size,
                    batch,
                    num_mid_groups,
                    groups_per_mid,
                )
                .sum(dim=0)
                .permute(2, 3, 1, 0, 4)
                .reshape(batch, num_mid_groups * intra_size, groups_per_mid, groups_per_mid)
                .contiguous()
            )
            stage3 = (
                rank_counts.reshape(
                    num_mid_groups,
                    groups_per_mid,
                    intra_size,
                    batch,
                    num_mid_groups,
                    groups_per_mid,
                    intra_size,
                )
                .sum(dim=(0, 1))
                .permute(1, 2, 3, 0, 4)
                .reshape(batch, num_mid_groups * groups_per_mid, intra_size, intra_size)
                .contiguous()
            )
            return stage1, stage2, stage3

        unique_stages = matrices(unique_rank, unique_intra, unique_mid)
        assignment_stages = matrices(assignment_rank, assignment_intra, assignment_mid)
        hidden_bytes = int(self.payload_bytes)
        stages = tuple(
            self._stage_traffic_features(
                unique,
                assignments,
                hidden_bytes=hidden_bytes,
                metadata_bytes=(3 * 4 if index == 0 else 2 * 4),
            )
            for index, (unique, assignments) in enumerate(zip(unique_stages, assignment_stages, strict=True))
        )
        stage1_node = self._stage_traffic_features(
            unique_stages[0].sum(dim=1, keepdim=True),
            assignment_stages[0].sum(dim=1, keepdim=True),
            hidden_bytes=hidden_bytes,
            metadata_bytes=3 * 4,
        )
        inter_links = self.perf_model.inter
        links = (
            inter_links[0],
            inter_links[1] if len(inter_links) > 1 else inter_links[0],
            self.perf_model.intra,
        )

        def weighted(feature: str) -> torch.Tensor:
            return sum(float(link.beta) * stage[feature] for link, stage in zip(links, stages, strict=True))

        result = {
            "stage_unique_endpoint_link_units": sum(
                2.0 * float(link.beta) * float(hidden_bytes) * stage["unique_endpoint_tokens"]
                for link, stage in zip(links, stages, strict=True)
            ),
            "stage_payload_endpoint_link_units": weighted("full_endpoint_bytes"),
            "stage_payload_edge_link_units": weighted("full_edge_bytes"),
            "stage_shared_node_endpoint_link_units": (
                float(links[0].beta) * stage1_node["full_endpoint_bytes"]
                + sum(
                    float(link.beta) * stage["full_endpoint_bytes"]
                    for link, stage in zip(links[1:], stages[1:], strict=True)
                )
            ),
            "stage_remote_payload_endpoint_link_units": weighted("remote_full_endpoint_bytes"),
            "stage_remote_payload_edge_link_units": weighted("remote_full_edge_bytes"),
            "stage_self_payload_link_units": weighted("self_endpoint_bytes"),
        }
        for index, stage in enumerate(stages, start=1):
            result.update(
                {
                    f"stage{index}_payload_endpoint_bytes": stage["full_endpoint_bytes"],
                    f"stage{index}_remote_payload_endpoint_bytes": stage["remote_full_endpoint_bytes"],
                    f"stage{index}_payload_edge_bytes": stage["full_edge_bytes"],
                    f"stage{index}_payload_total_bytes": stage["full_total_bytes"],
                    f"stage{index}_max_active_peers": stage["max_active_peers"],
                }
            )
        return result

    def _hierarchical_traffic_features(
        self,
        source_unique_counts: torch.Tensor,
        source_assignment_counts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Derive traffic features matching the runtime two-stage hierarchy.

        Stage 1 is split into one cross-node A2A group per local-rank lane.
        Stage 2 runs inside each destination node from relay lanes to target
        local ranks. This preserves peer/edge structure that endpoint-only
        hierarchy counts discard.
        """

        if source_unique_counts.shape != source_assignment_counts.shape:
            raise ValueError("Unique and assignment source counts must have identical shapes.")
        if source_unique_counts.ndim != 3 or int(source_unique_counts.shape[0]) != self.ep_size:
            raise ValueError(
                "Hierarchical source counts must have shape "
                f"[{self.ep_size}, batch, packed_width], got {tuple(source_unique_counts.shape)}."
            )
        if int(self.hierarchy.selected_dim) == 1:
            widths = self._count_widths()
            if widths != (self.ep_size,):
                raise ValueError(f"Unexpected one-stage packed widths {widths}.")
            unique_rank = source_unique_counts.permute(1, 0, 2).unsqueeze(1).contiguous()
            assignment_rank = source_assignment_counts.permute(1, 0, 2).unsqueeze(1).contiguous()
            stage = self._stage_traffic_features(
                unique_rank,
                assignment_rank,
                hidden_bytes=int(self.payload_bytes),
                metadata_bytes=3 * 4,
            )
            zero = torch.zeros_like(stage["full_endpoint_bytes"])
            beta = float(self.perf_model.intra.beta)
            return {
                "stage_unique_endpoint_link_units": (
                    2.0 * beta * float(self.payload_bytes) * stage["unique_endpoint_tokens"]
                ),
                "stage_payload_endpoint_link_units": beta * stage["full_endpoint_bytes"],
                "stage_payload_edge_link_units": beta * stage["full_edge_bytes"],
                "stage_shared_node_endpoint_link_units": beta * stage["full_endpoint_bytes"],
                "stage_remote_payload_endpoint_link_units": beta * stage["remote_full_endpoint_bytes"],
                "stage_remote_payload_edge_link_units": beta * stage["remote_full_edge_bytes"],
                "stage_self_payload_link_units": beta * stage["self_endpoint_bytes"],
                "stage1_payload_endpoint_bytes": zero,
                "stage2_payload_endpoint_bytes": stage["full_endpoint_bytes"],
                "stage1_remote_payload_endpoint_bytes": zero,
                "stage2_remote_payload_endpoint_bytes": stage["remote_full_endpoint_bytes"],
                "stage1_payload_edge_bytes": zero,
                "stage2_payload_edge_bytes": stage["full_edge_bytes"],
                "stage1_payload_total_bytes": zero,
                "stage2_payload_total_bytes": stage["full_total_bytes"],
                "stage1_max_active_peers": zero,
                "stage2_max_active_peers": stage["max_active_peers"],
            }
        if int(self.hierarchy.selected_dim) == 3:
            return self._hierarchical3d_traffic_features(source_unique_counts, source_assignment_counts)
        if int(self.hierarchy.selected_dim) != 2 or len(self.hierarchy.group_sizes) < 2:
            raise ValueError("Traffic-matrix diagnostics currently require a two-stage hierarchy.")
        intra_size = int(self.hierarchy.group_sizes[0])
        if intra_size <= 1 or self.ep_size % intra_size != 0:
            raise ValueError(f"Invalid intra-node size {intra_size} for EP size {self.ep_size}.")
        num_nodes = self.ep_size // intra_size
        widths = self._count_widths()
        if widths[:2] != (self.ep_size, num_nodes):
            raise ValueError(f"Unexpected two-stage packed widths {widths}.")

        unique_rank, unique_node = source_unique_counts.split(widths, dim=2)[:2]
        assignment_rank, assignment_node = source_assignment_counts.split(widths, dim=2)[:2]
        batch = int(source_unique_counts.shape[1])

        # [source_node, lane, batch, destination_node]
        stage1_unique = unique_node.reshape(num_nodes, intra_size, batch, num_nodes)
        stage1_assignments = assignment_node.reshape(num_nodes, intra_size, batch, num_nodes)
        # One independent cross-node group per local-rank lane.
        stage1_unique = stage1_unique.permute(2, 1, 0, 3).contiguous()
        stage1_assignments = stage1_assignments.permute(2, 1, 0, 3).contiguous()

        # Aggregate source nodes sharing the same relay lane at each
        # destination node: [batch, destination_node, relay_lane, local_rank].
        stage2_unique = (
            unique_rank.reshape(num_nodes, intra_size, batch, num_nodes, intra_size)
            .sum(dim=0)
            .permute(1, 2, 0, 3)
            .contiguous()
        )
        stage2_assignments = (
            assignment_rank.reshape(num_nodes, intra_size, batch, num_nodes, intra_size)
            .sum(dim=0)
            .permute(1, 2, 0, 3)
            .contiguous()
        )

        hidden_bytes = int(self.payload_bytes)
        # _pack_meta_weights converts every column to float32 for bf16 routes.
        stage1 = self._stage_traffic_features(
            stage1_unique,
            stage1_assignments,
            hidden_bytes=hidden_bytes,
            metadata_bytes=3 * 4,
        )
        stage2 = self._stage_traffic_features(
            stage2_unique,
            stage2_assignments,
            hidden_bytes=hidden_bytes,
            metadata_bytes=2 * 4,
        )

        # A shared node uplink can be bottlenecked by aggregate traffic across
        # all local-rank lanes even though HCCL creates one group per lane.
        stage1_node_unique = stage1_unique.sum(dim=1, keepdim=True)
        stage1_node_assignments = stage1_assignments.sum(dim=1, keepdim=True)
        stage1_node = self._stage_traffic_features(
            stage1_node_unique,
            stage1_node_assignments,
            hidden_bytes=hidden_bytes,
            metadata_bytes=3 * 4,
        )

        inter_link = self.perf_model.inter[0]
        intra_link = self.perf_model.intra
        unique_link_units = (
            2.0 * float(inter_link.beta) * float(hidden_bytes) * stage1["unique_endpoint_tokens"]
            + 2.0 * float(intra_link.beta) * float(hidden_bytes) * stage2["unique_endpoint_tokens"]
        )
        payload_endpoint_link_units = (
            float(inter_link.beta) * stage1["full_endpoint_bytes"]
            + float(intra_link.beta) * stage2["full_endpoint_bytes"]
        )
        payload_edge_link_units = (
            float(inter_link.beta) * stage1["full_edge_bytes"] + float(intra_link.beta) * stage2["full_edge_bytes"]
        )
        shared_node_link_units = (
            float(inter_link.beta) * stage1_node["full_endpoint_bytes"]
            + float(intra_link.beta) * stage2["full_endpoint_bytes"]
        )
        remote_payload_endpoint_link_units = (
            float(inter_link.beta) * stage1["remote_full_endpoint_bytes"]
            + float(intra_link.beta) * stage2["remote_full_endpoint_bytes"]
        )
        remote_payload_edge_link_units = (
            float(inter_link.beta) * stage1["remote_full_edge_bytes"]
            + float(intra_link.beta) * stage2["remote_full_edge_bytes"]
        )
        self_payload_link_units = (
            float(inter_link.beta) * stage1["self_endpoint_bytes"]
            + float(intra_link.beta) * stage2["self_endpoint_bytes"]
        )
        return {
            "stage_unique_endpoint_link_units": unique_link_units,
            "stage_payload_endpoint_link_units": payload_endpoint_link_units,
            "stage_payload_edge_link_units": payload_edge_link_units,
            "stage_shared_node_endpoint_link_units": shared_node_link_units,
            "stage_remote_payload_endpoint_link_units": remote_payload_endpoint_link_units,
            "stage_remote_payload_edge_link_units": remote_payload_edge_link_units,
            "stage_self_payload_link_units": self_payload_link_units,
            "stage1_payload_endpoint_bytes": stage1["full_endpoint_bytes"],
            "stage2_payload_endpoint_bytes": stage2["full_endpoint_bytes"],
            "stage1_remote_payload_endpoint_bytes": stage1["remote_full_endpoint_bytes"],
            "stage2_remote_payload_endpoint_bytes": stage2["remote_full_endpoint_bytes"],
            "stage1_payload_edge_bytes": stage1["full_edge_bytes"],
            "stage2_payload_edge_bytes": stage2["full_edge_bytes"],
            "stage1_payload_total_bytes": stage1["full_total_bytes"],
            "stage2_payload_total_bytes": stage2["full_total_bytes"],
            "stage1_max_active_peers": stage1["max_active_peers"],
            "stage2_max_active_peers": stage2["max_active_peers"],
        }

    def _local_traffic_endpoint_statistics(
        self,
        unique_counts: torch.Tensor,
        assignment_counts: torch.Tensor,
        *,
        source_rank: int,
    ) -> torch.Tensor:
        """Build compact source-aware sufficient statistics for both A2A stages.

        The full traffic matrix is unnecessary for endpoint-bottleneck
        scoring. For each stage and payload kind, only the per-source send and
        per-destination receive totals are retained. SUM reduction across EP
        ranks reconstructs the exact endpoint totals while preserving source
        lane/node identity.
        """

        if unique_counts.ndim != 2 or assignment_counts.ndim != 2:
            raise ValueError("Traffic counts must be two-dimensional.")
        if int(unique_counts.shape[0]) != int(assignment_counts.shape[0]):
            raise ValueError("Unique and assignment traffic counts must have the same batch size.")
        if int(self.hierarchy.selected_dim) == 1 and len(self.hierarchy.group_sizes) == 1:
            if int(unique_counts.shape[1]) != self.ep_size or int(assignment_counts.shape[1]) != self.ep_size:
                raise ValueError("Single-stage traffic counts must contain one column per EP rank.")
            if source_rank < 0 or source_rank >= self.ep_size:
                raise ValueError(f"Invalid source rank {source_rank} for EP size {self.ep_size}.")
            batch = int(unique_counts.shape[0])

            def endpoint_rows(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                send = values.new_zeros((batch, self.ep_size))
                send[:, int(source_rank)] = values.sum(dim=1)
                return send, values

            unique_send, unique_receive = endpoint_rows(unique_counts)
            assignment_send, assignment_receive = endpoint_rows(assignment_counts)
            empty = unique_counts.new_zeros((batch, self.ep_size))
            return torch.cat(
                (
                    empty,
                    empty,
                    empty,
                    empty,
                    unique_send,
                    unique_receive,
                    assignment_send,
                    assignment_receive,
                ),
                dim=1,
            )
        if int(self.hierarchy.selected_dim) != 2 or len(self.hierarchy.group_sizes) < 2:
            raise ValueError("Traffic endpoint scoring currently requires a two-stage hierarchy.")
        intra_size = int(self.hierarchy.group_sizes[0])
        if intra_size <= 1 or self.ep_size % intra_size != 0:
            raise ValueError(f"Invalid intra-node size {intra_size} for EP size {self.ep_size}.")
        if source_rank < 0 or source_rank >= self.ep_size:
            raise ValueError(f"Invalid source rank {source_rank} for EP size {self.ep_size}.")
        num_nodes = self.ep_size // intra_size
        widths = self._count_widths()
        if widths[:2] != (self.ep_size, num_nodes):
            raise ValueError(f"Unexpected two-stage packed widths {widths}.")

        unique_rank, unique_node = unique_counts.split(widths, dim=1)[:2]
        if int(assignment_counts.shape[1]) == self.ep_size:
            assignment_rank = assignment_counts
            assignment_node = assignment_rank.view(-1, num_nodes, intra_size).sum(dim=2)
        elif int(assignment_counts.shape[1]) == sum(widths):
            assignment_rank, assignment_node = assignment_counts.split(widths, dim=1)[:2]
        else:
            raise ValueError(
                f"Expected assignment width {self.ep_size} or {sum(widths)}, got {int(assignment_counts.shape[1])}."
            )
        batch = int(unique_counts.shape[0])
        lane = int(source_rank) % intra_size

        def stage1_rows(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            send = values.new_zeros((batch, self.ep_size))
            send[:, int(source_rank)] = values.sum(dim=1)
            receive = values.new_zeros((batch, self.ep_size))
            indices = lane * num_nodes + torch.arange(num_nodes, device=values.device)
            receive.scatter_(1, indices.view(1, -1).expand(batch, -1), values)
            return send, receive

        def stage2_rows(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            by_node = values.view(batch, num_nodes, intra_size).sum(dim=2)
            send = values.new_zeros((batch, self.ep_size))
            indices = torch.arange(num_nodes, device=values.device) * intra_size + lane
            send.scatter_(1, indices.view(1, -1).expand(batch, -1), by_node)
            return send, values

        unique_stage1_send, unique_stage1_receive = stage1_rows(unique_node)
        assignment_stage1_send, assignment_stage1_receive = stage1_rows(assignment_node)
        unique_stage2_send, unique_stage2_receive = stage2_rows(unique_rank)
        assignment_stage2_send, assignment_stage2_receive = stage2_rows(assignment_rank)
        return torch.cat(
            (
                unique_stage1_send,
                unique_stage1_receive,
                assignment_stage1_send,
                assignment_stage1_receive,
                unique_stage2_send,
                unique_stage2_receive,
                assignment_stage2_send,
                assignment_stage2_receive,
            ),
            dim=1,
        )

    def _traffic_endpoint_cost_details(
        self,
        endpoint_statistics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score globally reduced endpoint statistics with the hybrid model."""

        if endpoint_statistics.ndim != 2 or int(endpoint_statistics.shape[1]) != 8 * self.ep_size:
            raise ValueError(
                f"Expected traffic endpoint width {8 * self.ep_size}, got {tuple(endpoint_statistics.shape)}."
            )
        (
            unique_stage1_send,
            unique_stage1_receive,
            assignment_stage1_send,
            assignment_stage1_receive,
            unique_stage2_send,
            unique_stage2_receive,
            assignment_stage2_send,
            assignment_stage2_receive,
        ) = endpoint_statistics.split(self.ep_size, dim=1)
        hidden_bytes = float(self.payload_bytes)

        def full_endpoint_bytes(
            unique_send: torch.Tensor,
            unique_receive: torch.Tensor,
            assignment_send: torch.Tensor,
            assignment_receive: torch.Tensor,
            metadata_bytes: int,
        ) -> torch.Tensor:
            dispatch = torch.maximum(
                (hidden_bytes * unique_send + float(metadata_bytes) * assignment_send).amax(dim=1),
                (hidden_bytes * unique_receive + float(metadata_bytes) * assignment_receive).amax(dim=1),
            )
            combine = hidden_bytes * torch.maximum(
                unique_send.amax(dim=1),
                unique_receive.amax(dim=1),
            )
            return dispatch + combine

        stage1_bytes = full_endpoint_bytes(
            unique_stage1_send,
            unique_stage1_receive,
            assignment_stage1_send,
            assignment_stage1_receive,
            3 * 4,
        )
        stage2_bytes = full_endpoint_bytes(
            unique_stage2_send,
            unique_stage2_receive,
            assignment_stage2_send,
            assignment_stage2_receive,
            2 * 4,
        )
        peak_assignments, peak_compute_rank = assignment_stage2_receive.max(dim=1)
        network = self.traffic_communication_phase_multiplier * (
            self.traffic_inter_ms_per_byte * stage1_bytes + self.traffic_intra_ms_per_byte * stage2_bytes
        )
        local_route = (
            self.traffic_communication_phase_multiplier * self.traffic_route_ms_per_assignment * peak_assignments
        )
        communication = self.communication_scale * (network + local_route)
        compute = self.traffic_compute_phase_multiplier * (
            self.forward_compute_per_assignment * peak_assignments + self.forward_compute_constant
        )
        units = (
            self.traffic_inter_ms_per_byte * stage1_bytes
            + self.traffic_intra_ms_per_byte * stage2_bytes
            + self.traffic_route_ms_per_assignment * peak_assignments
        )
        # PlacementCost only uses this rank diagnostically. Preserve the
        # actual destination-rank bottleneck rather than the byte magnitude.
        peak_rank = (hidden_bytes * unique_stage2_receive + float(2 * 4) * assignment_stage2_receive).argmax(dim=1)
        selected_dim = torch.full_like(peak_rank, int(self.hierarchy.selected_dim))
        return communication, compute, units, peak_rank, peak_compute_rank, selected_dim


class RouteCostModel(TrafficAccounting):
    """Read-only cost model for replay; owns no candidate search or collectives."""

    def __init__(
        self,
        *,
        hierarchy,
        hidden_size,
        bytes_per_element,
        slots_per_rank,
        forward_compute_per_assignment,
        traffic_inter_ms_per_byte,
        traffic_intra_ms_per_byte,
        traffic_route_ms_per_assignment,
        traffic_communication_phase_multiplier,
        traffic_compute_phase_multiplier,
    ):
        self.hierarchy = hierarchy
        self.perf_model = HierMoEPerfModel.default()
        self.hidden_size = int(hidden_size)
        self.bytes_per_element = int(bytes_per_element)
        self.slots_per_rank = int(slots_per_rank)
        self.forward_compute_per_assignment = float(forward_compute_per_assignment)
        self.forward_compute_constant = 0.0
        self.communication_scale = 1.0
        self.traffic_inter_ms_per_byte = float(traffic_inter_ms_per_byte)
        self.traffic_intra_ms_per_byte = float(traffic_intra_ms_per_byte)
        self.traffic_route_ms_per_assignment = float(traffic_route_ms_per_assignment)
        self.traffic_communication_phase_multiplier = float(traffic_communication_phase_multiplier)
        self.traffic_compute_phase_multiplier = float(traffic_compute_phase_multiplier)
        if (
            min(
                self.forward_compute_per_assignment,
                self.traffic_inter_ms_per_byte,
                self.traffic_intra_ms_per_byte,
                self.traffic_route_ms_per_assignment,
                self.traffic_communication_phase_multiplier,
                self.traffic_compute_phase_multiplier,
            )
            < 0
        ):
            raise ValueError("Cost-model coefficients and phase multipliers must be non-negative.")
