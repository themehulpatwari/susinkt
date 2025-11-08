import pyautogui
import asyncio

dim = pyautogui.size()

async def scroll():
    dim = pyautogui.size()
    print("scrolled")
    pyautogui.moveTo(dim.width/2,dim.height/2)
    pyautogui.vscroll(-500)

async def click_like():
    pyautogui.moveTo(2*dim.width/3 + 50,dim.height/2)
    pyautogui.click()
    pyautogui.moveTo(dim.width/2,dim.height/2)

async def click_dislike():
    pyautogui.moveTo(2*dim.width/3 + 50,dim.height/2 + 75)
    pyautogui.click()
    pyautogui.moveTo(dim.width/2,dim.height/2)