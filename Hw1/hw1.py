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
import pandas as pd

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

num_pages = 34  # there are 34 pages of bestselling books
                # seems it stops at 34 no matter what/wont get last page

## create python dictionary
book_data = {'title': [],
            'author':[],
            'publication_date':[],
            'format':[],
            'current_price':[],
            'original_price':[]}

i = 1      # to track pages selenium has been through

while i <= num_pages:
    ## scrape data w bs
    soup = bs(page_source, 'lxml') 

    ## change url back so we can modify cleanly later
    url = "https://www.bookdepository.com/bestsellers"

    ## get book info
    book_info = soup.find_all("div", class_= "item-info") 

    ## break into title, author, pub date, format, curr price, orig price
    ## and add to dictionary 
    for info in book_info:
        book_title = info.find("h3", class_= "title")
        ## needs [] unless you want a list of chars
        book_data['title'] += [book_title.get_text(strip=True)]

        book_author = info.find("p", class_= "author")
        book_data['author'] += [book_author.get_text(strip=True)]

        ## some books do not have publication dates / prices
        ## check to see if they exist b4 adding
        book_pub_date = info.find("p", class_="published")
        if book_pub_date:
            book_data['publication_date'] += [book_pub_date.get_text(strip=True)]
        else:
            book_data['publication_date'] += ['']

        book_format = info.find("p", class_="format")
        book_data['format'] += [book_format.get_text(strip=True)]

        book_curr_price = info.find("span", class_="sale-price")
        if book_curr_price:
            book_data['current_price'] += [book_curr_price.get_text(strip=True)]
        else:
            book_data['current_price'] += ['']

        book_orig_price = info.find("span", class_="rrp")
        if book_orig_price:
            book_data['original_price'] += [book_orig_price.get_text(strip=True)]
        else:
            book_data['original_price'] += ['']

    ## There is no next on final page, so don't try to find/click it
    if i < 34:
        ## locate the next page button href
        next_page = driver.find_element(By.LINK_TEXT, "»")
        if next_page:
            ## click next page button if it exists
            next_page.click()
            ## update url and page_source
            extra_url = "?page={}".format(i+1)
            url = url + extra_url
            print(url)
            driver.get(url)
            page_source = driver.page_source
            ## I don't think I actually need selenium, given I'm modifying
            ## the url this way ... it looks cool, though
    i = i + 1
    print(i)

## now that dictionary is complete, turn into dataframe and export to .csv
book_dataframe = pd.DataFrame(book_data)
book_dataframe.to_csv('bestsellers.csv')
    
    
