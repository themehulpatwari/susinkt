import asyncio
from selenium import webdriver
import getInformation
import action

async def main():
    driver = webdriver.Chrome()
    driver.maximize_window()

    driver.get("https://www.youtube.com/shorts")

    task1 = asyncio.create_task(getInformation.fetch_reel())
    task2 = asyncio.create_task(getInformation.fetch_url(driver))

    await asyncio.sleep(5)

    for i in range(1):
        task3 = asyncio.create_task(action.scroll())

    await asyncio.sleep(2)
    task4 = asyncio.create_task(action.click_like())
    await asyncio.sleep(2)
    task5 = asyncio.create_task(action.click_dislike())

    try:
        await asyncio.gather(task1,task2,task3,task4,task5)
    except asyncio.CancelledError:
        print("Tasks were cancelled.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Cancel any task that is still running
        for task in [task1, task2]:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    print(f"{task.get_name()} cancelled successfully.")

if __name__ == "__main__":
    asyncio.run(main())
