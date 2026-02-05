# group: N trajectories from the same prompt (training unit)
#
# Status transitions:
#   trajectory: PENDING → IN_PROGRESS → SAMPLED → COMPLETED
#   group.training: PENDING → ENQUEUED → DONE

async def do_group_rollout(builder, client):
    # run N rollouts in parallel, one per trajectory
    states = await gather(*[
        run_rollout(inp, client)  # traj: PENDING → IN_PROGRESS → SAMPLED
        for inp in builder.get_rollout_inputs()
    ])
    await score_group(states)  # traj: SAMPLED → COMPLETED
    return to_trajectory_group(states)

async def worker_pool(env_builders_q, groups_q, client):

    async def worker():
        while (builder := await env_builders_q.get()) is not None:
            group = await do_group_rollout(builder, client)
            groups_q.put_nowait(group)

    await gather(*[worker() for _ in range(N)])

async def dataloader(dataset, env_builders_q):
    for batch_idx in range(len(dataset)):
        for builder in dataset.get_batch(batch_idx):
            await env_builders_q.put(builder)
    await env_builders_q.put(DONE)

async def streaming_minibatch(groups_q, training_client):
    handoff_q = Queue()

    async def producer():
        while (group := await groups_q.get()) is not DONE:
            data = await prepare(group)  # CPU-bound but almost instantaneous
            group.training_status = ENQUEUED
            fwd_bwd_future = await training_client.forward_backward_async(data)
            await handoff_q.put((fwd_bwd_future, group))
        # all minibatches enqueued, now enqueue optim_step
        optim_future = await training_client.optim_step_async()
        await handoff_q.put(optim_future)

    async def consumer():
        while (item := await handoff_q.get()):
            if is_optim_future(item):
                await item.result()  # apply gradients
                break
            fwd_bwd_future, group = item
            await fwd_bwd_future.result()  # wait for TPU while producer prepares next
            group.training_status = DONE

    await gather(producer(), consumer())

async def main(dataset, training_client, sampling_client):
    env_builders_q = Queue(maxsize=N)
    groups_q = Queue()

    await gather(
        dataloader(dataset, env_builders_q),
        worker_pool(env_builders_q, groups_q, sampling_client),
        streaming_minibatch(groups_q, training_client),
    )



# possible bottlenecks

## Accurate Concerns

**1. Sampling throughput is likely the primary bottleneck**
- In `do_sync_training_with_stream_minibatch` (train.py:391-423), all `groups_per_batch` groups are launched simultaneously via `asyncio.create_task`
- Each group spawns `group_size` parallel rollouts inside `custom_do_group_rollout` (verifiers_rl/train.py:223-226)
- Net concurrency: `groups_per_batch × group_size` rollouts in flight (e.g., 32×8 = 256)

**2. `trajectory_groups_queue` is unbounded** (train.py:387)
- If training is slower than sampling, completed groups accumulate in memory
- This is mitigated by the producer/consumer pattern in `run_substep` which blocks after consuming `groups_per_minibatch` groups

**3. Queue size diagnostics are valuable**
- Track `trajectory_groups_queue.qsize()` to detect if training lags sampling
- Already have timing metrics (`time/trajectory_group_worker_loop/total`, `train/fwd_bwd_*`)

## Implementation-Specific Concerns

**1. Backpressure gap in streaming mode**
- `trajectory_groups_queue` is unbounded (train.py:387)
- All `groups_per_batch` sampling tasks are spawned immediately (train.py:419-423)
- If TPU is slow, all groups complete before first minibatch trains → memory spike
- Consider: bound the queue or spawn tasks incrementally

**2. Retry backoff can amplify latency** (verifiers_rl/train.py:197-216)
- On LLM errors, retries with exponential backoff (1s → 1.1× per retry, up to 30 retries)
- A few slow trajectories can block entire group completion
- Consider: timeout + skip trajectory instead of indefinite retry

**3. `remove_constant_reward_groups` reduces effective batch size**
- Groups with identical rewards are filtered (train.py:701-704, 867-869)
- If many groups are filtered, actual training batch < configured batch
- No learning rate scaling to compensate
