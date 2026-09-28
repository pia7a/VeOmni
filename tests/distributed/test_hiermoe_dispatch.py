import pytest
import torch

from veomni.distributed.moe.hiermoe.all_to_all import _local_expert_sort_indices


@pytest.mark.parametrize("device_type", ["cpu", "npu"])
@pytest.mark.parametrize("build_unsort", [False, True])
@pytest.mark.parametrize("pattern", ["empty", "single", "skewed", "random", "strided"])
def test_local_expert_sort_preserves_assignment_order(device_type, build_unsort, pattern):
    if device_type == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("Requires an NPU")
    device = torch.device(device_type)
    generator = torch.Generator().manual_seed(928)
    num_experts = 33
    if pattern == "empty":
        ids = torch.empty(0, dtype=torch.long)
    elif pattern == "single":
        ids = torch.tensor([32])
    elif pattern == "skewed":
        ids = torch.full((4097,), 17, dtype=torch.long)
        ids[::257] = 0
    else:
        ids = torch.randint(num_experts, (262144,), generator=generator)
    expected_order = torch.argsort(ids, stable=True)
    expected_counts = torch.bincount(ids, minlength=num_experts)
    device_ids = ids.to(device)
    if pattern == "strided":
        device_ids = torch.stack((device_ids, device_ids), dim=1)[:, 0]
    order, inverse, counts = _local_expert_sort_indices(device_ids, num_experts, device, build_unsort=build_unsort)
    torch.testing.assert_close(order.cpu(), expected_order, rtol=0, atol=0)
    torch.testing.assert_close(counts.cpu(), expected_counts, rtol=0, atol=0)
    if build_unsort:
        torch.testing.assert_close(order[inverse].cpu(), torch.arange(ids.numel()), rtol=0, atol=0)
    else:
        assert inverse.numel() == 0
