import os, dotenv, pandas, csv, warnings, requests

def search_jobs() -> list:
    """
    This is a tool for agent to perform search jobs task
    :param:  none
    :return: list of jobs based on search query
    """
    dotenv.load_dotenv()
    warnings.filterwarnings("ignore")

    job_title = input(f"Enter your job title: ")
    job_location = input(f"Enter the location for search: ")
    query = f"Current open jobs with {job_title} in {job_location} from all job websites')"

    search_query= f"http://serpapi.com/search?q={query}&api_key={os.getenv('SERPAPI_API_KEY')}"
    response = requests.get(url=search_query,verify=False)
    open_jobs= response.json()['organic_results']
    return open_jobs

if __name__ == '__main__':

    df = pandas.DataFrame(search_jobs())
    df.to_csv('./jobs_list.csv',index=False,quoting=csv.QUOTE_ALL,quotechar='"')


