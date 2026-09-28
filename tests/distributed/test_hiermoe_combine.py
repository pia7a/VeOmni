import pytest
import torch

from veomni.distributed.moe.hiermoe.all_to_all import (
    _aggregate_weighted_outputs,
    _index_add_dim0_fp32,
    _NpuIndexAddDim0CastOutput,
    _NpuWeightedIndexAddDim0CastOutput,
)


def test_combine_aggregation_handles_different_source_token_counts() -> None:
    weighted_outputs = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    source_token_indices = torch.tensor([7, 7, 1, 8, 8])

    combined, output_splits = _aggregate_weighted_outputs(
        weighted_outputs,
        source_token_indices,
        split_sizes=[2, 3],
    )

    assert output_splits == [1, 2]
    torch.testing.assert_close(combined, torch.tensor([[3.0], [3.0], [9.0]]))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("chunk_rows", [1, 7, 65536])
@pytest.mark.parametrize("width", [11, 1025])
def test_combine_gather_backward_matches_chunked_autograd(monkeypatch, dtype, chunk_rows, width):
    """Exercise repeated indices, unused rows and a strided source on CPU."""
    monkeypatch.setenv("VEOMNI_HIERMOE_INDEX_ADD_FP32_CHUNK_ROWS", str(chunk_rows))
    generator = torch.Generator().manual_seed(923)
    source = torch.randn(width, 31, generator=generator, dtype=dtype).T.requires_grad_()
    reference_source = source.detach().clone().requires_grad_()
    index = torch.randint(0, 6, (31,), generator=generator)
    upstream = torch.randn(8, width, generator=generator, dtype=dtype)
    reference = _index_add_dim0_fp32(torch.zeros(8, width), index, reference_source).to(dtype)
    actual = _NpuIndexAddDim0CastOutput.apply(source, index, 8)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    reference.backward(upstream)
    actual.backward(upstream)
    torch.testing.assert_close(source.grad, reference_source.grad, rtol=0, atol=0)


def test_combine_empty_source_backward():
    source = torch.empty(0, 5, requires_grad=True)
    result = _NpuIndexAddDim0CastOutput.apply(source, torch.empty(0, dtype=torch.long), 3)
    torch.testing.assert_close(result, torch.zeros(3, 5))
    result.sum().backward()
    assert source.grad.shape == source.shape


def test_combine_backward_does_not_expand_each_chunk(monkeypatch):
    monkeypatch.setenv("VEOMNI_HIERMOE_INDEX_ADD_FP32_CHUNK_ROWS", "7")
    source = torch.randn(31, 11, dtype=torch.bfloat16, requires_grad=True)
    index = torch.arange(31) % 8
    output = _NpuIndexAddDim0CastOutput.apply(source, index, 8)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        output.sum().backward()
    assert not any(event.name == "aten::slice_backward" for event in trace.events())
    torch.testing.assert_close(source.grad, torch.ones_like(source))


@pytest.mark.parametrize("shape", [(31, 11), (16385, 33), (31, 1025), (0, 5)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("needs_grad", [(True, True), (True, False), (False, True)])
def test_weighted_combine_matches_broadcast_autograd(shape, dtype, needs_grad):
    generator = torch.Generator().manual_seed(928)
    rows, width = shape
    source = torch.randn(width, rows, dtype=dtype, generator=generator).T.requires_grad_(needs_grad[0])
    weights = torch.randn(rows, 1, dtype=dtype, generator=generator).requires_grad_(needs_grad[1])
    ref_source = source.detach().clone().requires_grad_(needs_grad[0])
    ref_weights = weights.detach().clone().requires_grad_(needs_grad[1])
    index = torch.randint(0, 9, (rows,), generator=generator)
    upstream = torch.randn(11, width, dtype=dtype, generator=generator)
    # The unweighted wrapper also keeps an empty source connected to autograd.
    reference = _NpuIndexAddDim0CastOutput.apply(ref_source * ref_weights, index, 11)
    actual = _NpuWeightedIndexAddDim0CastOutput.apply(source, weights, index, 11)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    reference.backward(upstream)
    actual.backward(upstream)
    for parameter, ref_parameter in ((source, ref_source), (weights, ref_weights)):
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, ref_parameter.grad, rtol=0, atol=0)


def test_weighted_combine_preserves_mixed_dtype_rounding():
    generator = torch.Generator().manual_seed(928)
    source = torch.randn(37, 1025, dtype=torch.bfloat16, generator=generator).requires_grad_()
    weights = torch.randn(37, 1, generator=generator).requires_grad_()
    index = torch.arange(37) % 9
    reference = _NpuIndexAddDim0CastOutput.apply(source * weights.to(source.dtype), index, 11)
    upstream = torch.randn_like(reference)
    expected = torch.autograd.grad(reference, (source, weights), upstream)
    actual = _NpuWeightedIndexAddDim0CastOutput.apply(source, weights, index, 11)
    gradients = torch.autograd.grad(actual, (source, weights), upstream)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for gradient, ref_gradient in zip(gradients, expected):
        torch.testing.assert_close(gradient, ref_gradient, rtol=0, atol=0)
