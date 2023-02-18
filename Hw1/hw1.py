### Matthew Virgin
### Dr. Chaofan Chen
### COS 482
### 20 February 2023

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup as bs
import requests
import time

## website that we will be scraping
url = "https://www.bookdepository.com/bestsellers"

### selenium setup
options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
options.add_argument('--headless')
driver = webdriver.Chrome("Users/Matt/Desktop/School/DataSci/Hw1")
driver.get(url)
## give page time to load:
time.sleep(3)

### for each page, load in the html - use selenium to change pages
page_source = driver.page_source

num_pages = 34          # there are 34 pages of bestselling books

i = 0

while True:
    try:
        ## wait until next page button is displayed
        WebDriverWait(driver, 3).until(
            lambda s: s.find_element(By.CLASS_NAME, 'next').is_displayed()
        )
    except TimeoutException:
        break

    ## scrape data w bs, load into pandas****

    ## locate the next page button href
    next_page = driver.find_element(By.LINK_TEXT, "»")
    if next_page:
        ## click next page button if it exists
        # driver.execute_script("arguments[0].click;", next_page) - keeping this here because it's what prof did even though it seems I don't need it
        next_page.click()
    i = i + 1
    if i > num_pages:
        break
    
