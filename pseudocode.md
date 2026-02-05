# group: N trajectories from the same prompt (training unit)
#
# Status transitions:
#   trajectory: PENDING → IN_PROGRESS → SAMPLED → COMPLETED
#   group.training: PENDING → ENQUEUED → DONE

async def worker_pool(env_builders_q, groups_q):

    async def worker():
        while (builder := await env_builders_q.get()) is not None:
            for traj in builder.trajectories:
                traj.status = IN_PROGRESS
            group = await rollout(builder)  # traj → SAMPLED after each rollout
            for traj in group.trajectories:
                traj.status = COMPLETED  # after reward assigned
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

async def main(dataset, training_client):
    env_builders_q = Queue(maxsize=N)
    groups_q = Queue()

    await gather(
        dataloader(dataset, env_builders_q),
        worker_pool(env_builders_q, groups_q),
        streaming_minibatch(groups_q, training_client),
    )
