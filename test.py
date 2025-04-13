import asyncio
import time


# async def say_after(delay, what):
#     await asyncio.sleep(delay)
#     print(what)

# async def main():
#     print(f"started at {time.strftime('%X')}")
#     await say_after(1,'hello')
#     await say_after(2,'world')
#     print(f"finished at {time.strftime('%X')}")

# asyncio.run(main())


async def say_after(delay, what):
    print(f"start: {what}, number of tasks: 1" )
    await asyncio.sleep(delay)
    print(f"end: {what}")

async def main():
    # print(asyncio.tasks.all_tasks())
    # task1 = asyncio.create_task(
    #     say_after(1,'hello')
    # )

    # task2 = asyncio.create_task(
    #     say_after(2,'world')
    # )

    print(f"started at {time.strftime('%X')}")
    # print(asyncio.tasks.all_tasks())
    # await task2
    # await task1

    await asyncio.create_task(
        say_after(1,'hello')
    )
    await asyncio.create_task(
        say_after(2,'world')
    )
    
    print(f"finished at {time.strftime('%X')}")

asyncio.run(main())