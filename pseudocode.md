# 1. Pipelined Futures
async def pipelined_train(batches):
    curr_future = await enqueue(batches[0])
    for i, batch in enumerate(batches):
        next_future = await enqueue(batches[i+1]) if i+1 < len(batches) else None
        await curr_future.result()
        curr_future = next_future

# 2. Worker Pool
async def worker_pool():
    input_q, output_q = Queue(maxsize=N), Queue()

    async def worker():
        while (item := await input_q.get()) is not None:
            result = await process(item)
            output_q.put_nowait(result)

    await gather(feeder(), *[worker() for _ in range(N)], consumer())

# 3. Producer-Consumer with Future Handoff
async def streaming_minibatch():
    handoff_q = Queue()

    async def producer():
        for batch in batches:
            data = await prepare(batch)
            future = await enqueue_to_tpu(data)
            await handoff_q.put(future)  # hand off future, not result
        await handoff_q.put(DONE)

    async def consumer():
        while (future := await handoff_q.get()) is not DONE:
            await future.result()  # wait for TPU while producer prepares next

    await gather(producer(), consumer())

# 4. Event-Based Coordination
async def event_coordination():
    event = Event()
    shared_state = None

    async def trainer():
        nonlocal shared_state
        for batch in batches:
            shared_state = await train(batch)
            event.set()

    async def evaluator():
        while not done:
            await event.wait()
            event.clear()
            await evaluate(shared_state)

    await gather(trainer(), evaluator())

# 5. Stale Sample Requeue
async def training_loop():
    while i < end:
        sample = await output_q.get()
        if sample.age > max_age:
            create_task(input_q.put(sample.source))  # requeue, don't await
            continue
        await train(sample)
        i += 1
