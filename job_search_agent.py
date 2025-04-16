import os, dotenv, pandas, csv, warnings, requests, json
from langchain_ollama import ChatOllama
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage

def search_jobs(role:str, location:str, type:str) -> list:
    """
    This is a tool for agent to perform search jobs task
    :param:  role, type and location
    :return: list of jobs based on search query
    """
    dotenv.load_dotenv()
    warnings.filterwarnings("ignore")
    #construct the query for searching the job sites
    query = f"Current open {role} in {location} of {type} from all job websites')"
    # print(query)
    # generate the search query for serpapi get http call
    search_query= f"http://serpapi.com/search?q={query}&api_key={os.getenv('SERPAPI_API_KEY')}"
    # make the hhtp api call to serpapi service
    response = requests.get(url=search_query,verify=False)
    #convert and extract the relevant search details
    open_jobs= response.json()['organic_results']
    return open_jobs

class JobSearchAgent(Runnable):
    def invoke(self,query:str):
        """
        This is an agent that extract the role, type and location from the user input and invokes
        the search_jobs tool to get the list of jobs that user queries for
        This is using langchain v0.2 interfaces

        :param query: the user input
        :return: list of jobs by using the search job tool
        """

        dotenv.load_dotenv()
        # instantiate the local llm model
        llm = ChatOllama(model=os.getenv('INFERENCE_MODEL'))
        # create the prompt to extract the job type, role and location from user input
        extraction_prompt = (f"Extract the type, role, location from this query {query}."
                             f"Respond in JSON keys 'type', 'role' and 'location"
                             )
        # get the job role, type and location using llm
        response = llm.invoke([HumanMessage(content=extraction_prompt)])
        try:
            job_info = json.loads(response.content)
            job_type = job_info['type']
            job_role = job_info['role']
            job_location = job_info['location']
        except Exception as e:
            return f"Error parsing llm response: {e} \n. Raw output: {response.content}"
        # invoke the search job tools with the extracted info
        jobs = search_jobs(role=job_role, type=job_type, location=job_location)
        return jobs

if __name__ == '__main__':
    # get user input
    query = input('Enter your job search query with job title, location and type: ')
    #invoke the agent to get the results
    df = pandas.DataFrame(JobSearchAgent().invoke(query))
    #store the results to a csv file
    df.to_csv('./jobs_list.csv',index=False,quoting=csv.QUOTE_ALL,quotechar='"')

