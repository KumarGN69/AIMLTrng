import google.generativeai as genai
import os, dotenv, pandas, csv
import requests, certifi
from serpapi import GoogleSearch


def search_jobs(job_details: str) -> str:
    """
    This is a tool for agent to perform the requested task
    :param job_details:
    :return: details compiled by search and query using serpapi
    """
    dotenv.load_dotenv()
    search_query= f"http://serpapi.com/search?q={job_details}&api_key={os.getenv('SERPAPI_API_KEY')}"
    response = requests.get(url=search_query,verify=False)
    return response.json()['organic_results']

if __name__ == '__main__':
    job_title= input (f"Enter your job title: ")
    job_location = input (f"Enter the location for search: ")

    query = f"Currently open jobs with {job_title} in {job_location} posted on LinkedIn')"
    open_jobs = [job for job in search_jobs( query)]
    df = pandas.DataFrame(open_jobs)
    df.to_csv('./jobs_list.csv',index=False,quoting=csv.QUOTE_ALL,quotechar='"')


