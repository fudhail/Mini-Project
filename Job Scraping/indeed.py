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
browser.get(f"https://in.indeed.com/jobs?q=business+analyst&l=Calicut%2C+Kerala")
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

    # Scrape job listings
    soup = BeautifulSoup(browser.page_source, "html.parser")

    # Find job cards (inspect the webpage to get the correct class)
    job_listings = soup.find_all("div", class_="job_seen_beacon")

    for job in job_listings:
        try:
            job_title = job.find("a").text.strip()
            job_company = job.find("div", class_="company_location").find("span").text.strip()
            job_location = job.find("div",attrs={'data-testid': 'text-location'}).text.strip()
            # apply_link = job.find("a", class_="jcs-JobTitle")["href"]
            # job_ID = apply_link.split("jk=")[1]
            print(job_title, job_company, job_location)
            # Open the job page for the full job description
            # browser.get("https://in.indeed.com" + apply_link)
            time.sleep(random.choice(list(range(5, 11))))

            # try:
            #     description_soup = BeautifulSoup(browser.page_source, "html.parser")
            #     job_description = description_soup.find("div", class_="jobsearch-jobDescriptionText").text.strip()
            # except AttributeError:
            #     job_description = None

            jobs.append(
                {
                    # "job ID": job_ID,
                    "title": job_title,
                    "company": job_company,
                    "location": job_location,
                    # "link": "https://in.indeed.com" + apply_link,
                    # "job description": job_description,
                }
            )

        except Exception as e:
            print(f"Error processing job: {e}")

    # Try to click on the "Next" button (adjust XPath based on the website structure)
    try:
        next_button = WebDriverWait(browser, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//a[@aria-label='Next']"))
        )
        next_button.click()
        time.sleep(3)
    except Exception as e:
        print(f"Could not find next button")
        pass

# Close browser
browser.quit()

# Save data into a csv file, excluding the index column
df = pd.DataFrame(jobs)
df.to_csv("indeed_jobs.csv", index=False)

print("Job scraping completed and saved to 'indeed_jobs.csv'.")


