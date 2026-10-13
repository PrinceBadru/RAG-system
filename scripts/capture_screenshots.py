import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By

os.makedirs("application_screenshots", exist_ok=True)

c_opts = ChromeOptions()
c_opts.add_argument("--headless=new")
c_opts.add_argument("--window-size=1600,1100")
c_opts.add_argument("--disable-gpu")
driver = webdriver.Chrome(options=c_opts)

try:
    url = "http://localhost:8501"
    print(f"Loading {url}...")
    driver.get(url)

    # Poll every 2 seconds for up to 60 seconds until tabs appear
    for attempt in range(30):
        tabs = driver.find_elements(By.XPATH, "//button[contains(@role, 'tab')]")
        if tabs and len(tabs) >= 2:
            print(f"Found {len(tabs)} tabs on attempt {attempt+1}!")
            break
        print(f"Attempt {attempt+1}: waiting for tabs (found {len(tabs)})...")
        time.sleep(2)

    time.sleep(4)  # Extra buffer to ensure all markdown, badges, and CSS render

    # 1. Capture Tab 1: Executive Overview & Risk Matrix
    img2_path = os.path.abspath("application_screenshots/image_2.png")
    driver.save_screenshot(img2_path)
    print(f"Captured Tab 1: {img2_path} ({os.path.getsize(img2_path)} bytes)")

    # 2. Click Tab 2: Interactive Audit Assistant and capture
    tabs = driver.find_elements(By.XPATH, "//button[contains(@role, 'tab')]")
    if len(tabs) >= 2:
        tabs[1].click()
        time.sleep(4)
        img1_path = os.path.abspath("application_screenshots/image.png")
        driver.save_screenshot(img1_path)
        print(f"Captured Tab 2: {img1_path} ({os.path.getsize(img1_path)} bytes)")

finally:
    driver.quit()
    print("Capture script finished.")
