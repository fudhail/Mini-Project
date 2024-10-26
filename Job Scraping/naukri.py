
import time
import random
import os
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import chromedriver_autoinstaller

# Automatically install ChromeDriver
chromedriver_autoinstaller.install()

# Configure Chrome options
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")

# Launch Chrome
browser = webdriver.Chrome(options=options)

# Open Indeed job search page (modify keywords and location as needed)
browser.get(f"https://www.naukri.com/business-analyst-jobs-in-calicut?k=business%20analyst&l=calicut")
time.sleep(3)

# Set the number of pages to scrape
pages = 1

# List to store job data
jobs = []

# Loop through pages
for i in range(pages):
    print(f"Scraping page {i + 1}")

    # Scroll to the bottom to load dynamic content
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)  # Adjust time as needed for the page to load fully


    soup = BeautifulSoup(browser.page_source, "html.parser")

    # Attempt to find job listings (the selectors will need to be tailored to the site's structure)
    job_listings = soup.find_all('div', class_='cust-job-tuple')

    for job in job_listings:
        # Extract job title
        title = job.find('a', class_='title').text.strip()
        
        # Extract company name
        company = job.find('a', class_='comp-name').text.strip()
        
        # Extract the link to the job
        # link = job.find('a')['href']
        
        # Print the extracted information
        print(f'Job Title: {title}')
        print(f'Company: {company}')
        # print(f'Link: {link}\n')
else:
    print('Failed to retrieve the webpage.')