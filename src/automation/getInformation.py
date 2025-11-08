import pyautogui, asyncio
from pathlib import Path

# This function will get the url through the driver api

async def fetch_url(driver):
    last_url = driver.current_url
    print("Starting URL monitor. Initial:", last_url)

    try:
        while True:
            current = driver.current_url
            if current != last_url:
                print("URL changed to:", current)
                last_url = current
            await asyncio.sleep(2) 
    except asyncio.CancelledError:
        print("URL monitor stopped.")
        raise

# This function will get the url through a image

# async def fetch_url1():
#     local_dir = Path(__file__).parent
#     file_path = local_dir / "url.png"

#     loop = asyncio.get_running_loop()
#     try:
#         while True:
#             await asyncio.wait_for(loop.run_in_executor(None,lambda:pyautogui.screenshot(file_path,region=(0,0,100,100))),timeout=1)
#             print("successful")
#             await asyncio.sleep(1)
#     except asyncio.TimeoutError:
#         print("Screenshot took too long")
#         return None
    
async def fetch_reel():
    local_dir = Path(__file__).parent
    file_path = local_dir / "reel.png"

    loop = asyncio.get_running_loop()
    try:
        while True:
            await asyncio.wait_for(loop.run_in_executor(None,lambda:pyautogui.screenshot(file_path,region=(1000,300,1000,1300))),timeout=1)
            print("successful")
            await asyncio.sleep(1)
    except asyncio.TimeoutError:
        print("Screenshot took too long")
    






