"""Expert parameter expansion, optimizer-state access, and tensor transfer primitives."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

import torch
import torch.distributed as dist
from torch import nn


try:
    from torch.distributed._tensor import DTensor
except ImportError:  # pragma: no cover - older torch fallback
    DTensor = ()  # type: ignore[assignment]

from placemoe.model_adapter import resolve_moe_model_adapter

from . import runtime_settings as settings
from .runtime_types import _CoverTensorEntry, _OptimizerParamBinding, _SwapBucketItem, _SwapTensorEntry


def _is_dtensor(tensor: torch.Tensor) -> bool:
    return bool(DTensor) and isinstance(tensor, DTensor)


def _local_tensor_view(tensor: torch.Tensor) -> torch.Tensor:
    if not _is_dtensor(tensor):
        return tensor
    local = tensor.to_local()
    if tuple(local.shape) != tuple(tensor.shape):
        raise NotImplementedError(
            "HierMoE expert swap requires complete local expert tensors. "
            f"Got DTensor global shape {tuple(tensor.shape)} and local shape {tuple(local.shape)}."
        )
    return local


def _copy_tensor_attrs(src: torch.Tensor, dst: torch.Tensor, attrs: tuple[str, ...]) -> None:
    for attr in attrs:
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))


def _expanded_local_parameter(param: torch.nn.Parameter, target_slots: int) -> torch.nn.Parameter:
    local = _local_tensor_view(param)
    if int(local.shape[0]) == int(target_slots):
        return param
    if int(local.shape[0]) > int(target_slots):
        raise ValueError(f"Cannot shrink HierMoE expert parameter from {tuple(local.shape)} to {target_slots} slots.")

    expanded = torch.empty(
        (int(target_slots), *tuple(local.shape[1:])),
        dtype=local.dtype,
        device=local.device,
    )
    if local.device.type != "meta":
        expanded[: local.shape[0]].copy_(local.detach())
        expanded[local.shape[0] :].zero_()
    new_param = torch.nn.Parameter(expanded, requires_grad=param.requires_grad)
    _copy_tensor_attrs(param, new_param, ("spec_info",))
    return new_param


def expand_redundant_expert_slots(model: nn.Module, *, ep_size: int, redundant_slot_increment_per_device: int) -> int:
    """Reserve empty local expert slots after EP slicing and before FSDP wrapping."""

    increment = max(0, int(redundant_slot_increment_per_device))
    if increment == 0 or int(ep_size) <= 1:
        return 0

    expanded_layers = 0
    for _key, module in model.named_modules():
        adapter = resolve_moe_model_adapter(module)
        if adapter is None:
            continue
        num_experts = adapter.num_experts(module)
        if num_experts % int(ep_size) != 0:
            raise ValueError(
                f"HierMoE redundant slots require num_experts={num_experts} divisible by ep_size={ep_size}."
            )
        base_slots = num_experts // int(ep_size)
        target_slots = base_slots + increment
        expert_parameters = adapter.expert_parameters(module)
        local_parameters = tuple(_local_tensor_view(item.parameter) for item in expert_parameters)
        if all(int(parameter.shape[0]) == target_slots for parameter in local_parameters):
            continue
        invalid = [
            f"{item.name}={tuple(parameter.shape)}"
            for item, parameter in zip(expert_parameters, local_parameters, strict=True)
            if int(parameter.shape[0]) != base_slots
        ]
        if invalid:
            raise ValueError(
                "HierMoE redundant slot expansion must run immediately after EP slicing. "
                f"Expected {base_slots} local experts, got {', '.join(invalid)}."
            )
        for item in expert_parameters:
            adapter.replace_expert_parameter(
                module, item.name, _expanded_local_parameter(item.parameter, target_slots)
            )
        expanded_layers += 1
    return expanded_layers


def _ep_global_rank(ep_group: dist.ProcessGroup | None, ep_rank: int) -> int:
    if ep_group is None:
        return int(ep_rank)
    try:
        return int(dist.get_global_rank(ep_group, int(ep_rank)))
    except (AttributeError, RuntimeError, ValueError):
        return int(ep_rank)


def _create_expert_swap_process_group(
    ep_group: dist.ProcessGroup | None,
    ep_size: int,
    group_desc: str = "hiermoe_expert_swap",
) -> dist.ProcessGroup | None:
    if ep_group is None or ep_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return None
    global_ranks = [_ep_global_rank(ep_group, rank) for rank in range(ep_size)]
    try:
        swap_group = dist.new_group(
            ranks=global_ranks,
            backend=dist.get_backend(ep_group),
            use_local_synchronization=True,
            group_desc=group_desc,
        )
    except TypeError as error:
        raise RuntimeError(
            "HierMoE asynchronous expert swap requires PyTorch new_group support for "
            "use_local_synchronization and group_desc."
        ) from error
    if swap_group is None or swap_group == dist.GroupMember.NON_GROUP_MEMBER:
        raise RuntimeError("HierMoE failed to create the dedicated expert-swap process group.")
    # Eager initialization avoids the first P2P batch requiring every group rank.
    dist.barrier(group=swap_group)
    return swap_group


def _swap_local_slot(tensor: torch.Tensor, lhs_slot: int, rhs_slot: int) -> None:
    local_tensor = _local_tensor_view(tensor)
    tmp = local_tensor.detach()[lhs_slot].clone()
    local_tensor.detach()[lhs_slot].copy_(local_tensor.detach()[rhs_slot])
    local_tensor.detach()[rhs_slot].copy_(tmp)


def _copy_local_slot(tensor: torch.Tensor, src_slot: int, dst_slot: int) -> None:
    local_tensor = _local_tensor_view(tensor)
    local_tensor.detach()[dst_slot].copy_(local_tensor.detach()[src_slot])


def _zero_local_slot(tensor: torch.Tensor, slot: int) -> None:
    local_tensor = _local_tensor_view(tensor)
    local_tensor.detach()[slot].zero_()


def _chunk_swap_bucket(bucket: list[_SwapBucketItem]) -> list[list[_SwapBucketItem]]:
    chunks: list[list[_SwapBucketItem]] = []
    current: list[_SwapBucketItem] = []
    current_nbytes = 0
    for item in bucket:
        item_nbytes = item[4]
        if current and current_nbytes + item_nbytes > settings._MAX_SWAP_BUCKET_BYTES:
            chunks.append(current)
            current = []
            current_nbytes = 0
        current.append(item)
        current_nbytes += item_nbytes
    if current:
        chunks.append(current)
    return chunks


def _pack_swap_chunk(chunk: list[_SwapBucketItem]) -> torch.Tensor:
    send_parts = [entry[2] for entry in chunk]
    return torch.cat(send_parts, dim=0) if len(send_parts) > 1 else send_parts[0]


def _swap_chunk_nbytes(chunk: list[_SwapBucketItem]) -> int:
    return sum(item[4] for item in chunk)


def _unpack_swap_chunk(recv_buffer: torch.Tensor, chunk: list[_SwapBucketItem]) -> None:
    offset = 0
    for local_tensor, local_slot, _send_view, numel, _nbytes in chunk:
        recv_view = recv_buffer[offset : offset + numel].view_as(local_tensor.detach()[local_slot])
        local_tensor.detach()[local_slot].copy_(recv_view)
        offset += numel


@torch.no_grad()
def _exchange_or_swap_grouped_slot_entries_collective(
    grouped_entries: dict[tuple[int, int], list[_SwapTensorEntry]],
    ep_rank: int,
    ep_size: int,
    ep_group: dist.ProcessGroup | None,
) -> None:
    remote_buckets: dict[tuple[torch.device, torch.dtype], dict[int, list[_SwapBucketItem]]] = defaultdict(
        lambda: defaultdict(list)
    )
    active_bucket_keys: set[tuple[torch.device, torch.dtype]] = set()

    for (lhs_rank, rhs_rank), entry_list in sorted(grouped_entries.items()):
        if lhs_rank == rhs_rank:
            if ep_rank == lhs_rank:
                for entry in entry_list:
                    _swap_local_slot(entry.tensor, entry.lhs_slot, entry.rhs_slot)
            continue

        for entry in entry_list:
            local_tensor = _local_tensor_view(entry.tensor)
            active_bucket_keys.add((local_tensor.device, local_tensor.dtype))

        if ep_group is None or ep_size <= 1 or ep_rank not in (lhs_rank, rhs_rank):
            continue

        peer_rank = rhs_rank if ep_rank == lhs_rank else lhs_rank
        for entry in entry_list:
            local_slot = entry.lhs_slot if ep_rank == lhs_rank else entry.rhs_slot
            local_tensor = _local_tensor_view(entry.tensor)
            slot_view = local_tensor.detach()[local_slot]
            send_view = slot_view.contiguous().view(-1)
            numel = int(send_view.numel())
            nbytes = numel * int(send_view.element_size())
            remote_buckets[(send_view.device, send_view.dtype)][peer_rank].append(
                (local_tensor, int(local_slot), send_view, numel, nbytes)
            )

    if ep_group is None or ep_size <= 1:
        return

    def _bucket_sort_key(key: tuple[torch.device, torch.dtype]) -> tuple[str, int, str]:
        device, dtype = key
        return (device.type, -1 if device.index is None else int(device.index), str(dtype))

    for device, dtype in sorted(active_bucket_keys, key=_bucket_sort_key):
        peer_buckets = remote_buckets.get((device, dtype), {})
        send_split_tensor = torch.zeros((ep_size,), dtype=torch.long, device=device)
        for peer_rank, bucket in peer_buckets.items():
            send_split_tensor[int(peer_rank)] = sum(item[3] for item in bucket)

        recv_split_tensor = torch.empty_like(send_split_tensor)
        dist.all_to_all_single(recv_split_tensor, send_split_tensor, group=ep_group)
        input_splits = [int(value) for value in send_split_tensor.detach().cpu().tolist()]
        output_splits = [int(value) for value in recv_split_tensor.detach().cpu().tolist()]

        send_buffer = torch.empty((sum(input_splits),), dtype=dtype, device=device)
        offset = 0
        for peer_rank in range(ep_size):
            bucket = peer_buckets.get(peer_rank, ())
            if not bucket:
                continue
            packed = _pack_swap_chunk(bucket)
            send_buffer[offset : offset + packed.numel()].copy_(packed)
            offset += packed.numel()

        recv_buffer = torch.empty((sum(output_splits),), dtype=dtype, device=device)
        dist.all_to_all_single(
            recv_buffer,
            send_buffer,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=ep_group,
        )

        offset = 0
        for peer_rank, split_size in enumerate(output_splits):
            if split_size <= 0:
                continue
            bucket = peer_buckets.get(peer_rank)
            if not bucket:
                raise RuntimeError(
                    f"Expert Swap collective exchange received {split_size} values from rank {peer_rank} "
                    "without a matching local swap plan."
                )
            expected = sum(item[3] for item in bucket)
            if split_size != expected:
                raise RuntimeError(
                    f"Expert Swap collective exchange split mismatch from rank {peer_rank}: "
                    f"expected {expected}, got {split_size}."
                )
            _unpack_swap_chunk(recv_buffer[offset : offset + split_size], bucket)
            offset += split_size


@torch.no_grad()
def _cover_slot_entries(
    entries: Iterable[_CoverTensorEntry],
    src_rank: int,
    dst_rank: int,
    ep_rank: int,
    ep_group: dist.ProcessGroup | None,
) -> None:
    entry_list = list(entries)
    if not entry_list:
        return
    if src_rank == dst_rank:
        if ep_rank == src_rank:
            for entry in entry_list:
                _copy_local_slot(entry.tensor, entry.src_slot, entry.dst_slot)
        return
    if ep_group is None or ep_rank not in (src_rank, dst_rank):
        return

    peer_global_rank = _ep_global_rank(ep_group, dst_rank if ep_rank == src_rank else src_rank)
    buckets: dict[tuple[torch.device, torch.dtype], list[tuple[torch.Tensor, int, torch.Tensor, int]]] = defaultdict(
        list
    )
    for entry in entry_list:
        local_tensor = _local_tensor_view(entry.tensor)
        if ep_rank == src_rank:
            view = local_tensor.detach()[entry.src_slot].contiguous().view(-1)
            buckets[(view.device, view.dtype)].append((local_tensor, -1, view, int(view.numel())))
        else:
            view = local_tensor.detach()[entry.dst_slot].view(-1)
            buckets[(view.device, view.dtype)].append((local_tensor, int(entry.dst_slot), view, int(view.numel())))

    for bucket in buckets.values():
        if ep_rank == src_rank:
            send_buffer = torch.cat([item[2] for item in bucket], dim=0) if len(bucket) > 1 else bucket[0][2]
            dist.send(send_buffer, dst=peer_global_rank)
        else:
            total_numel = sum(item[3] for item in bucket)
            recv_buffer = torch.empty((total_numel,), dtype=bucket[0][2].dtype, device=bucket[0][2].device)
            dist.recv(recv_buffer, src=peer_global_rank)
            offset = 0
            for _local_tensor, _dst_slot, view, numel in bucket:
                view.copy_(recv_buffer[offset : offset + numel].view_as(view))
                offset += numel


@torch.no_grad()
def _placement_group_succeeded(
    local_success: bool,
    *,
    device: torch.device,
    ep_size: int,
    ep_group: dist.ProcessGroup | None,
) -> bool:
    if ep_size <= 1 or ep_group is None:
        return bool(local_success)
    backend = str(dist.get_backend(ep_group)).lower().rsplit(".", maxsplit=1)[-1]
    status_device = torch.device("cpu") if backend == "gloo" else device
    status = torch.tensor([int(local_success)], dtype=torch.int32, device=status_device)
    dist.all_reduce(status, op=dist.ReduceOp.MIN, group=ep_group)
    return bool(status.item())


@torch.no_grad()
def _placement_group_boolean_consensus(
    local_value: bool,
    *,
    device: torch.device,
    ep_size: int,
    ep_group: dist.ProcessGroup | None,
) -> tuple[bool, bool]:
    """Return whether every rank is true and whether all ranks agree."""

    if ep_size <= 1 or ep_group is None:
        return bool(local_value), True
    backend = str(dist.get_backend(ep_group)).lower().rsplit(".", maxsplit=1)[-1]
    status_device = torch.device("cpu") if backend == "gloo" else device
    status = torch.tensor([int(local_value)], dtype=torch.int32, device=status_device)
    dist.all_reduce(status, op=dist.ReduceOp.SUM, group=ep_group)
    true_count = int(status.item())
    return true_count == ep_size, true_count in (0, ep_size)


@torch.no_grad()
def _placement_group_all_true_mask(
    local_values: Sequence[bool],
    *,
    device: torch.device,
    ep_size: int,
    ep_group: dist.ProcessGroup | None,
) -> tuple[bool, ...]:
    """Return a rank-consistent mask that is true only when every rank is ready."""

    if ep_size <= 1 or ep_group is None:
        return tuple(bool(value) for value in local_values)
    backend = str(dist.get_backend(ep_group)).lower().rsplit(".", maxsplit=1)[-1]
    status_device = torch.device("cpu") if backend == "gloo" else device
    status = torch.tensor(local_values, dtype=torch.int32, device=status_device)
    dist.all_reduce(status, op=dist.ReduceOp.MIN, group=ep_group)
    return tuple(bool(value) for value in status.to(device="cpu").tolist())


@torch.no_grad()
def _cover_grouped_slot_entries_atomic(
    grouped_entries: dict[tuple[int, int], list[_CoverTensorEntry]],
    ep_rank: int,
    ep_size: int,
    ep_group: dist.ProcessGroup | None,
    *,
    zero_entry_groups: Iterable[tuple[int, Iterable[_CoverTensorEntry]]] = (),
    debug_validate: bool = False,
) -> None:
    """Stage all directed slot copies, then publish their destination tensors.

    Placement combines bidirectional swaps and one-way replica copies.  Treating
    both as directed copies lets us batch by peer, dtype, and slot shape while
    keeping destination tensors untouched until every communication succeeds.
    """

    zero_groups = tuple((int(rank), tuple(entries)) for rank, entries in zero_entry_groups)
    all_entries = tuple(entry for entries in grouped_entries.values() for entry in entries) + tuple(
        entry for _rank, entries in zero_groups for entry in entries
    )
    if not all_entries:
        return
    if ep_size > 1 and ep_group is None and any(src_rank != dst_rank for src_rank, dst_rank in grouped_entries):
        raise RuntimeError("HierMoE placement state migration requires an EP process group.")
    status_device = _local_tensor_view(all_entries[0].tensor).device

    def group_succeeded(local_success: bool) -> bool:
        if not debug_validate:
            return bool(local_success)
        return _placement_group_succeeded(
            local_success,
            device=status_device,
            ep_size=ep_size,
            ep_group=ep_group,
        )

    bucket_keys: set[tuple[torch.device, torch.dtype, tuple[int, ...]]] = set()
    send_buckets: dict[
        tuple[torch.device, torch.dtype, tuple[int, ...]], dict[int, list[tuple[torch.Tensor, int]]]
    ] = defaultdict(lambda: defaultdict(list))
    recv_buckets: dict[
        tuple[torch.device, torch.dtype, tuple[int, ...]], dict[int, list[tuple[torch.Tensor, int, int]]]
    ] = defaultdict(lambda: defaultdict(list))
    local_copies: list[tuple[torch.Tensor, int, torch.Tensor | None]] = []
    has_remote = False
    stage_error: Exception | None = None

    try:
        for (src_rank, dst_rank), entries in sorted(grouped_entries.items()):
            for entry in entries:
                local_tensor = _local_tensor_view(entry.tensor)
                src_view = local_tensor.detach()[entry.src_slot]
                dst_view = local_tensor.detach()[entry.dst_slot]
                if tuple(src_view.shape) != tuple(dst_view.shape):
                    raise RuntimeError("HierMoE placement tried to copy between incompatible expert slot shapes.")
                key = (src_view.device, src_view.dtype, tuple(int(value) for value in src_view.shape))
                bucket_keys.add(key)
                if src_rank == dst_rank:
                    if ep_rank == src_rank:
                        local_copies.append((local_tensor, int(entry.dst_slot), src_view.clone()))
                    continue
                has_remote = True
                if ep_rank == src_rank:
                    send_buckets[key][int(dst_rank)].append((src_view.contiguous().view(-1), int(src_view.numel())))
                elif ep_rank == dst_rank:
                    recv_buckets[key][int(src_rank)].append((local_tensor, int(entry.dst_slot), int(dst_view.numel())))
        for dst_rank, entries in zero_groups:
            for entry in entries:
                local_tensor = _local_tensor_view(entry.tensor)
                dst_view = local_tensor.detach()[entry.dst_slot]
                bucket_keys.add((dst_view.device, dst_view.dtype, tuple(int(value) for value in dst_view.shape)))
                if ep_rank == dst_rank:
                    # A ``None`` staged value below represents transactional zeroing.
                    local_copies.append((local_tensor, int(entry.dst_slot), None))
    except Exception as error:
        stage_error = error

    if not group_succeeded(stage_error is None):
        if stage_error is not None:
            raise RuntimeError("HierMoE placement transaction preflight failed.") from stage_error
        raise RuntimeError("Another EP rank rejected the HierMoE placement transaction preflight.")

    def bucket_sort_key(
        key: tuple[torch.device, torch.dtype, tuple[int, ...]],
    ) -> tuple[str, int, str, tuple[int, ...]]:
        device, dtype, shape = key
        return (device.type, -1 if device.index is None else int(device.index), str(dtype), shape)

    remote_commits: list[tuple[torch.Tensor, int, torch.Tensor]] = []
    try:
        if has_remote:
            for key in sorted(bucket_keys, key=bucket_sort_key):
                device, dtype, _shape = key
                peer_sends = send_buckets.get(key, {})
                peer_recvs = recv_buckets.get(key, {})
                send_splits_tensor = torch.zeros((ep_size,), dtype=torch.long, device=device)
                for peer_rank, items in peer_sends.items():
                    send_splits_tensor[int(peer_rank)] = sum(item[1] for item in items)
                recv_splits_tensor = torch.empty_like(send_splits_tensor)
                if ep_size > 1:
                    dist.all_to_all_single(recv_splits_tensor, send_splits_tensor, group=ep_group)
                else:
                    recv_splits_tensor.copy_(send_splits_tensor)
                input_splits = [int(value) for value in send_splits_tensor.detach().cpu().tolist()]
                output_splits = [int(value) for value in recv_splits_tensor.detach().cpu().tolist()]

                send_buffer = torch.empty((sum(input_splits),), dtype=dtype, device=device)
                offset = 0
                for peer_rank in range(ep_size):
                    items = peer_sends.get(peer_rank, ())
                    if not items:
                        continue
                    packed = torch.cat([item[0] for item in items], dim=0) if len(items) > 1 else items[0][0]
                    send_buffer[offset : offset + packed.numel()].copy_(packed)
                    offset += int(packed.numel())
                recv_buffer = torch.empty((sum(output_splits),), dtype=dtype, device=device)
                if ep_size > 1:
                    dist.all_to_all_single(
                        recv_buffer,
                        send_buffer,
                        output_split_sizes=output_splits,
                        input_split_sizes=input_splits,
                        group=ep_group,
                    )
                elif send_buffer.numel():
                    recv_buffer.copy_(send_buffer)

                offset = 0
                for peer_rank, split_size in enumerate(output_splits):
                    items = peer_recvs.get(peer_rank, ())
                    expected = sum(item[2] for item in items)
                    if int(split_size) != int(expected):
                        raise RuntimeError(
                            f"HierMoE placement state migration from rank {peer_rank} expected {expected} values, "
                            f"received {split_size}."
                        )
                    inner_offset = offset
                    for local_tensor, dst_slot, numel in items:
                        staged = recv_buffer[inner_offset : inner_offset + numel].view_as(
                            local_tensor.detach()[dst_slot]
                        )
                        remote_commits.append((local_tensor, dst_slot, staged))
                        inner_offset += numel
                    offset += split_size
    except Exception as error:
        stage_error = error

    if not group_succeeded(stage_error is None):
        if stage_error is not None:
            raise RuntimeError("HierMoE placement state migration staging failed.") from stage_error
        raise RuntimeError("Another EP rank failed to stage the HierMoE placement state migration.")

    publish_ops: list[tuple[torch.Tensor, int, torch.Tensor | None]] = [*local_copies, *remote_commits]
    destinations: set[tuple[int, int]] = set()
    publish_preflight_error: Exception | None = None
    try:
        for local_tensor, dst_slot, _staged in publish_ops:
            key = (id(local_tensor), int(dst_slot))
            if key in destinations:
                raise RuntimeError("HierMoE placement transaction writes one tensor slot more than once.")
            destinations.add(key)
    except Exception as error:
        publish_preflight_error = error
    if not group_succeeded(publish_preflight_error is None):
        if publish_preflight_error is not None:
            raise RuntimeError("HierMoE placement publish preflight failed.") from publish_preflight_error
        raise RuntimeError("Another EP rank rejected the HierMoE placement publish preflight.")

    publish_error: Exception | None = None
    try:
        for local_tensor, dst_slot, staged in publish_ops:
            if staged is None:
                local_tensor.detach()[dst_slot].zero_()
            else:
                local_tensor.detach()[dst_slot].copy_(staged)
    except Exception as error:
        publish_error = error

    publish_succeeded = group_succeeded(publish_error is None)
    if not publish_succeeded:
        if publish_error is not None:
            raise RuntimeError("HierMoE placement state migration publish failed.") from publish_error
        raise RuntimeError("Another EP rank failed to publish the HierMoE placement state migration.")


@torch.no_grad()
def _zero_slot_entries(entries: Iterable[_CoverTensorEntry], dst_rank: int, ep_rank: int) -> None:
    if ep_rank != dst_rank:
        return
    for entry in entries:
        _zero_local_slot(entry.tensor, entry.dst_slot)


def _iter_leaf_optimizers(optimizer: Any) -> Iterable[Any]:
    if optimizer is None:
        return ()
    if hasattr(optimizer, "optimizers_dict"):
        return tuple(optimizer.optimizers_dict.values())
    return (optimizer,)


def _step_device(param: torch.nn.Parameter, group: dict[str, Any]) -> torch.device:
    if bool(group.get("capturable", False)) or bool(group.get("fused", False)):
        return param.device
    return torch.device("cpu")


def _existing_optimizer_state(optimizer: Any, param: torch.nn.Parameter) -> dict[str, Any] | None:
    state = getattr(optimizer, "state", None)
    if state is None:
        return None
    existing = state.get(param)
    return existing if existing else None


def _build_optimizer_param_bindings(optimizer: Any) -> dict[int, tuple[_OptimizerParamBinding, ...]]:
    bindings: dict[int, list[_OptimizerParamBinding]] = defaultdict(list)
    for opt in _iter_leaf_optimizers(optimizer):
        for group in opt.param_groups:
            for param in group["params"]:
                bindings[id(param)].append(_OptimizerParamBinding(opt, group))
    return {param_id: tuple(items) for param_id, items in bindings.items()}


def _optimizer_state_slot_tensors(optimizer: Any, param: torch.nn.Parameter) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for opt in _iter_leaf_optimizers(optimizer):
        state = _existing_optimizer_state(opt, param)
        if not state:
            continue
        for value in state.values():
            if not torch.is_tensor(value):
                continue
            if tuple(_local_tensor_view(value).shape) != tuple(_local_tensor_view(param).shape):
                continue
            tensors.append(value)
    return tensors


def _optimizer_state_slot_tensors_from_bindings(
    bindings: Iterable[_OptimizerParamBinding], param: torch.nn.Parameter
) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for binding in bindings:
        state = _existing_optimizer_state(binding.optimizer, param)
        if not state:
            continue
        for value in state.values():
            if not torch.is_tensor(value):
                continue
            if tuple(_local_tensor_view(value).shape) != tuple(_local_tensor_view(param).shape):
                continue
            tensors.append(value)
    return tensors
