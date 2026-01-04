import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from from_scratch.syncbatchnorm import SyncBatchNorm


def init_process(rank, world_size, fn, *args):
    """Initialize distributed environment for worker processes."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29555"
    backend = "gloo"  # Use gloo for CPU testing
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    fn(rank, world_size, *args)
    dist.destroy_process_group()


def run_worker_syncbn(rank, world_size, hid_dim, batch_size, input_shard, results_queue):
    x = input_shard.clone().requires_grad_(True)
    local_batch_size = batch_size // world_size

    bn = SyncBatchNorm(hid_dim)
    bn.train()

    output = bn(x)

    samples_for_loss = batch_size // 2
    start_idx = rank * local_batch_size
    local_samples_for_loss = max(0, min(local_batch_size, samples_for_loss - start_idx))

    loss = output[:local_samples_for_loss].sum() if local_samples_for_loss > 0 else output.sum() * 0
    loss.backward()

    results_queue.put(
        {
            "rank": rank,
            "output": output.detach().cpu(),
            "grad_input": x.grad.detach().cpu() if x.grad is not None else torch.zeros_like(x),
            "running_mean": bn.running_mean.detach().cpu(),
            "running_var": bn.running_var.detach().cpu(),
        }
    )


def run_syncbn_test(num_workers, hid_dim, batch_size):
    """Run SyncBatchNorm test with multiple workers."""
    torch.manual_seed(42)
    full_input = torch.randn(batch_size, hid_dim)
    shards = full_input.chunk(num_workers)

    ctx = mp.get_context("spawn")
    results_queue = ctx.Queue()

    processes = []
    for rank in range(num_workers):
        p = ctx.Process(
            target=init_process,
            args=(rank, num_workers, run_worker_syncbn, hid_dim, batch_size, shards[rank], results_queue),
        )
        p.start()
        processes.append(p)

    results = []
    for _ in range(num_workers):
        results.append(results_queue.get())

    for p in processes:
        p.join()

    results.sort(key=lambda x: x["rank"])

    return results


def run_batchnorm1d_reference(num_workers, hid_dim, batch_size):
    """Run standard BatchNorm1d as reference (single process, full batch)."""
    torch.manual_seed(42)
    x = torch.randn(batch_size, hid_dim)
    x.requires_grad = True

    bn = torch.nn.BatchNorm1d(hid_dim, affine=False)
    bn.train()

    output = bn(x)

    samples_for_loss = batch_size // 2
    loss = output[:samples_for_loss].sum()

    loss.backward()

    return {
        "output": output.detach(),
        "grad_input": x.grad.detach(),
        "running_mean": bn.running_mean.detach(),
        "running_var": bn.running_var.detach(),
    }


@pytest.mark.parametrize("num_workers", [1, 4])
@pytest.mark.parametrize("hid_dim", [128, 256, 512, 1024])
@pytest.mark.parametrize("batch_size", [32, 64])
def test_batchnorm(num_workers, hid_dim, batch_size):
    """
    Test SyncBatchNorm against standard BatchNorm1d.

    Compares forward and backward passes with:
    - Different number of workers (1, 4)
    - Different hidden dimensions (128, 256, 512, 1024)
    - Different batch sizes (32, 64)

    Requirements from assignment:
    - FP32 inputs from standard Gaussian distribution
    - Loss function: sum over all dimensions for first B/2 samples
    - Should have atol <= 1e-3 and rtol = 0
    """
    syncbn_results = run_syncbn_test(num_workers, hid_dim, batch_size)

    bn1d_result = run_batchnorm1d_reference(num_workers, hid_dim, batch_size)

    syncbn_output = torch.cat([r["output"] for r in syncbn_results], dim=0)
    syncbn_grad_input = torch.cat([r["grad_input"] for r in syncbn_results], dim=0)

    syncbn_running_mean = syncbn_results[0]["running_mean"]
    syncbn_running_var = syncbn_results[0]["running_var"]

    assert torch.allclose(
        syncbn_output, bn1d_result["output"], atol=1e-3, rtol=0
    ), f"Output mismatch: max diff = {(syncbn_output - bn1d_result['output']).abs().max()}"

    assert torch.allclose(
        syncbn_grad_input, bn1d_result["grad_input"], atol=1e-3, rtol=0
    ), f"Gradient mismatch: max diff = {(syncbn_grad_input - bn1d_result['grad_input']).abs().max()}"
    assert torch.allclose(
        syncbn_running_mean, bn1d_result["running_mean"], atol=1e-3, rtol=0
    ), f"Running mean mismatch: max diff = {(syncbn_running_mean - bn1d_result['running_mean']).abs().max()}"

    assert torch.allclose(
        syncbn_running_var, bn1d_result["running_var"], atol=1e-3, rtol=0
    ), f"Running var mismatch: max diff = {(syncbn_running_var - bn1d_result['running_var']).abs().max()}"
