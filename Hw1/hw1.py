### Matthew Virgin
### Dr. Chaofan Chen
### COS 482
### 20 February 2023

## optional selenium import to show pages changing as it scrapes
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait

## true requirements
from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
import matplotlib.pyplot as plt

## website that we will be scraping
url = "https://www.bookdepository.com/bestsellers"

### selenium setup - uncomment if you want to watch it change pages
### so you're not just sitting there. Uncomment line 102 as well
# options = webdriver.ChromeOptions()
# options.add_argument('--ignore-certificate-errors')
# options.add_argument('--incognito')
# options.add_argument('--headless')
# driver = webdriver.Chrome("Users/Matt/Desktop/School/DataSci/Hw1")
# driver.get(url)
# WebDriverWait(driver,3)

num_pages = 34  # there are 34 pages of bestselling books

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
    soup = bs(requests.get(url).content, 'lxml') 

    ## change url back so we can modify cleanly later
    url = "https://www.bookdepository.com/bestsellers"

    ## get book info
    book_info = soup.find_all("div", class_= "item-info") 

    ## break into title, author, pub date, format, curr price, orig price
    ## and add to dictionary 
    for info in book_info:
        ## some books do not have publication dates / prices
        ## so check to see if items they exist b4 adding
        ## author is sometimes '', so need to filter that out as well, checking
        ## all for '' just in case
        book_title = info.find("h3", class_= "title")
        if book_title and book_title.get_text(strip=True) != '':
            ## needs [] unless you want a list of chars
            book_data['title'] += [book_title.get_text(strip=True)]
        else:
            book_data['title'] += [None]

        book_author = info.find("p", class_= "author")
        if book_author and book_author.get_text(strip=True) != '':
            book_data['author'] += [book_author.get_text(strip=True)]
        else:
            book_data['author'] += [None]

        book_pub_date = info.find("p", class_="published")
        if book_pub_date and book_pub_date.get_text(strip=True) != '':
            book_data['publication_date'] += [book_pub_date.get_text(strip=True)]
        else:
            book_data['publication_date'] += [None]

        book_format = info.find("p", class_="format")
        if book_format and book_format.get_text(strip=True) != '':
            book_data['format'] += [book_format.get_text(strip=True)]
        else:
            book_data['format'] += [None]

        book_curr_price = info.find("span", class_="sale-price")
        if book_curr_price and book_curr_price.get_text(strip=True) != '':
            book_data['current_price'] += [book_curr_price.get_text(strip=True)]
        else:
            book_data['current_price'] += [None]

        book_orig_price = info.find("span", class_="rrp")
        if book_orig_price and book_orig_price.get_text(strip=True) != '':
            book_data['original_price'] += [book_orig_price.get_text(strip=True)]
        else:
            book_data['original_price'] += [None]

    ## There is no next on final page
    if i < 34:
        ## update url
        extra_url = "?page={}".format(i+1)
        url = url + extra_url

        ## optional - watch it change pages - uncomment to see selenium
        ## changing pages as it scrapes the data so you at least have something
        ## to look at while it loads

        # driver.get(url)

        ## I don't actually need Selenium, given there's no need to scroll,
        ## but I added all the code already so I figure I may as well use it

    i = i + 1

## now that dictionary is complete, turn into dataframe and export to .csv
book_df = pd.DataFrame(book_data)
book_df.to_csv('bestsellers.csv')

## I also noticed that the book website sometimes fails to load prices ...
## this happens just in my browser and refreshing will load them
## for example, the final book on page 34 as of 2/25/2023, 
## "Shooting an Elephant" loads it's discounted price only sometimes.

### Task 1 complete

### Task 2:
### a):

book_df.rename(columns = {'publication_date' : 'publication_year'}, 
                      inplace = True)

date = book_df['publication_year']

## removes the day and month from the strings contained in a pandas series
def removeDayMonth (someS):
    someS_list = someS.tolist()             # convert series to list
    for i in range(len(someS_list)):
        ## remove day and month chars, cast to int
        someS_list[i] = int(someS_list[i][7:])   
    return someS_list 

book_df['publication_year'] = removeDayMonth(date)

## remove rows containing empty values
book_df.dropna(inplace = True)

## get current and original price columns
curr_price = book_df['current_price']
orig_price = book_df['original_price']

## removes the first 3 chars from the strings contained in a pandas series
## used to remove US$ from curr_price and orig_price
def removeCash (someS):
    someS_list = someS.tolist()             # convert series to list
    for i in range(len(someS_list)):
        ## remove first 3 chars, cast to int
        someS_list[i] = float(someS_list[i][3:])    
    return someS_list                      

book_df['current_price'] = removeCash(curr_price)
book_df['original_price'] = removeCash(orig_price)

### b):
book_df.to_csv('bestsellers-cleaned.csv')

### c):

## new dataframe containing only bestsellers published during or after 2022
after_2022_df = book_df[book_df['publication_year'] >= 2022]

## drop unneeded columns as we only care about title, author, and current price
after_2022_df.drop(columns = ['publication_year', 'format', 'original_price'], 
                   inplace = True)

## save to csv for easy viewing
after_2022_df.to_csv('c.csv')

### d):

book_df.plot(x='current_price', y='original_price', kind='scatter',
              title = 'current vs. original price')

### e):
book_df[['current_price']].plot(kind='hist', title = 'Bestseller current prices')

### f):
book_df[['current_price','publication_year']].groupby(
    'publication_year').mean().sort_values(
    by = 'publication_year', ascending=True).plot.bar(
    title='Average current price by publication year')

plt.show()          # display plots