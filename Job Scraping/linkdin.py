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

chromedriver_autoinstaller.install()

# Configure Chrome options
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")

# Launch chromedriver maximized
browser = webdriver.Chrome(options=options)

# Part 3 - Open LinkedIn job search page (modify keywords as needed)
browser.get(f'https://www.linkedin.com/jobs/search/?keywords=Business%20Analyst&location=Toronto&position=1&pageNum=0')
input()
# Set the number of pages to scrape
pages: int = 1

# Part 4 - Loop through the specified number of pages to retrieve job postings
for i in range(pages):
    print(f'Scraping page {i + 1}')
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    try:
        # Click on the "see more jobs" button if present
        element = WebDriverWait(browser, 5).until(
            EC.presence_of_element_located(
                (By.XPATH, "/html/body/div[1]/div/main/section[2]/button")
            )
        )
        element.click()
    except Exception:
        pass

    # Part 5 - Scrape job postings
    jobs = []
    soup = BeautifulSoup(browser.page_source, "html.parser")
    job_listings = soup.find_all("div", class_="base-card")
    for job in job_listings:
        job_title = job.find("h3", class_="base-search-card__title").text.strip()
        job_company = job.find("h4", class_="base-search-card__subtitle").text.strip()
        job_location = job.find("span", class_="job-search-card__location").text.strip()
        apply_link = job.find("a", class_="base-card__full-link")["href"]
        job_ID = apply_link[apply_link.find('?position=') - 10:apply_link.find('?position=')]
        print(job_title)
        browser.get(apply_link)
        time.sleep(random.choice(list(range(5, 11))))

        try:
            description_soup = BeautifulSoup(browser.page_source, "html.parser")
            job_description = description_soup.find("div", class_="description__text description__text--rich").text.strip()
        except AttributeError:
            job_description = None

        jobs.append({
            "job ID": job_ID,
            "title": job_title,
            "company": job_company,
            "location": job_location,
            "link": apply_link,
            "job description": job_description,
        })

# Part 6 - Save data into a csv file, exclude index column
df = pd.DataFrame(jobs)
df.to_csv("jobs3.csv", index=False)