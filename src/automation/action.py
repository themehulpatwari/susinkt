import pyautogui
import asyncio

async def scroll():
    dim = pyautogui.size()
    print("scrolled")
    pyautogui.moveTo(dim.width/2,dim.height/2)
    while True:
        pyautogui.vscroll(-1000)
        await asyncio.sleep(1)