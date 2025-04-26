import pandas as pd
import csv
# read projects mapped to cloud
df = pd.read_csv('./Inputs/ProjectMapping.csv')
cloud_projects = df.dropna(subset=['Cloud (Y/N)'])

# read the PP report data
project_financials = pd.read_csv('./Inputs/ERS-TECHNOLOGY-FY25_TREND_REPORT.csv')

# merge the two data frames and filter on only cloud projects
merged_df = pd.merge(
    cloud_projects,
    project_financials,
    left_on='Project Code',   # column in df_projects
    right_on='Project',  # column in df_financials
    how='left'                # or 'inner' as needed
)
# convert to ensure reveneue and HC are float numbers
merged_df['Total Revenue'] = merged_df['Total Revenue'].astype(float)
merged_df['Total BFTE'] = merged_df['BFTE + UC/FC'].astype(float)

# save the financial data from PP report for the cloud projects to a separate csv file
merged_df.to_csv("./Reports/cloud_project_list.csv", index=False,quoting=csv.QUOTE_ALL,quotechar='"')


# create sub dataframes for revenue and bfte
revenue_df=  merged_df.groupby(['PA','Sub- PA','Month'])['Total Revenue'].sum().reset_index()
bfte_df=  merged_df.groupby(['PA','Sub- PA',"Month"])['Total BFTE'].sum().reset_index()

# merge the sub data frames into one dataframe
summary_df=pd.merge(revenue_df,bfte_df,on=['PA', 'Sub- PA','Month'])
summary_pivot = summary_df.pivot_table(index=['PA','Sub- PA'],columns='Month', values=['Total Revenue', 'Total BFTE'], aggfunc='sum')
summary_pivot = summary_pivot.reset_index()
summary_pivot.to_csv('./Reports/summary_projects.csv', index=False,quoting=csv.QUOTE_ALL,quotechar='"')

summary_pivot = merged_df.pivot_table(index=['PA'],columns='Month', values=['Project_5'], aggfunc='nunique')
summary_pivot = summary_pivot.reset_index()
summary_pivot.to_csv('./Reports/summary_projects_byPA.csv', index=False,quoting=csv.QUOTE_ALL,quotechar='"')

summary_pivot = merged_df.pivot_table(index=['Revised Component Group'],columns='Month', values=['Project_5'], aggfunc='nunique')
summary_pivot = summary_pivot.reset_index()
summary_pivot.to_csv('./Reports/summary_projects_byONOFF.csv', index=False,quoting=csv.QUOTE_ALL,quotechar='"')

summary_pivot = merged_df.pivot_table(index=['Sub- PA'],columns='Month', values=['Project_5'], aggfunc='nunique')
summary_pivot = summary_pivot.reset_index()
summary_pivot.to_csv('./Reports/summary_projects_bySubPA.csv', index=False,quoting=csv.QUOTE_ALL,quotechar='"')

# create pivots and save to separate files for subsequent trends
summary_pivot = merged_df.pivot_table(index=['PA'],columns='Month', values=['Total Revenue'], aggfunc='sum')
summary_pivot = summary_pivot.reset_index()
summary_pivot.to_csv('./Reports/summary_pivot_spend_byPA.csv', index=False,quoting=csv.QUOTE_ALL,quotechar='"')


# create pivots and save to separate files for subsequent trends
summary_pivot = merged_df.pivot_table(index=['Revised Component Group'],columns='Month', values=['Total Revenue'], aggfunc='sum')
summary_pivot = summary_pivot.reset_index()
summary_pivot.to_csv('./Reports/summary_pivot_spend_byONOFF.csv', index=False,quoting=csv.QUOTE_ALL,quotechar='"')

summary_pivot = merged_df.pivot_table(index=['Sub- PA'],columns='Month', values=['Total Revenue'], aggfunc='sum')
summary_pivot = summary_pivot.reset_index()
summary_pivot.to_csv('./Reports/summary_pivot_spend_bySubPA.csv', index=False,quoting=csv.QUOTE_ALL,quotechar='"')

summary_pivot = merged_df.pivot_table(index=['Employee Comp Code PP_8'],columns='Month', values=['Total Revenue'], aggfunc='sum')
summary_pivot = summary_pivot.reset_index()
summary_pivot.to_csv('./Reports/summary_pivot_spend_by_country.csv', index=False,quoting=csv.QUOTE_ALL,quotechar='"')

summary_pivot = merged_df.pivot_table(index=['Employee Comp Code PP_8'],columns="Month", values=['Total BFTE'], aggfunc='sum')
summary_pivot = summary_pivot.reset_index()
summary_pivot.to_csv('./Reports/summary_pivot_bfte_by_country.csv', index=False,quoting=csv.QUOTE_ALL,quotechar='"')

summary_pivot = merged_df.pivot_table(index=['Revised Component Group','PA'],columns=['Month'], values=['Total BFTE'], aggfunc='sum')
summary_pivot = summary_pivot.reset_index()
summary_pivot.to_csv('./Reports/summary_pivot_bfte_geos_ON_OFF.csv', index=False,quoting=csv.QUOTE_ALL,quotechar='"')

summary_pivot = merged_df.pivot_table(index=['PA'],columns=['Month'], values=['Total BFTE'], aggfunc='sum')
summary_pivot = summary_pivot.reset_index()
summary_pivot.to_csv('./Reports/summary_pivot_bfte_geos_month_byPA.csv', index=False,quoting=csv.QUOTE_ALL,quotechar='"')

summary_pivot = merged_df.pivot_table(index=['Sub- PA'],columns=['Month'], values=['Total BFTE'], aggfunc='sum')
summary_pivot = summary_pivot.reset_index()
summary_pivot.to_csv('./Reports/summary_pivot_bfte_geos_month_bySubPA.csv', index=False,quoting=csv.QUOTE_ALL,quotechar='"')

summary_pivot = merged_df.pivot_table(index=['Major Proj Cat'],columns=['Month'], values=['Total BFTE'], aggfunc='sum')
summary_pivot = summary_pivot.reset_index()
summary_pivot.to_csv('./Reports/summary_pivot_proj_category.csv', index=False,quoting=csv.QUOTE_ALL,quotechar='"')

summary_pivot = merged_df.pivot_table(index=['Project_5'],columns=['Month'], values=['Total BFTE'], aggfunc='sum')
summary_pivot = summary_pivot.reset_index()
summary_pivot.to_csv('./Reports/summary_pivot_proj_bfte.csv', index=False,quoting=csv.QUOTE_ALL,quotechar='"')
# # load the cloud projects data
df = pd.read_csv('./Reports/cloud_project_list.csv')

# create a sub dataframe for a specific month (latest)
filtered_df = df[df['Month'] == "Mar'25"]
summary_pivot = filtered_df.pivot_table(index=['Employee Comp Code PP_8'],columns=['PA'], values=['Total BFTE'], aggfunc='sum')
summary_pivot = summary_pivot.reset_index()
summary_pivot.to_csv('./Reports/summary_pivot_bfte_geos_latest_month.csv', index=False,quoting=csv.QUOTE_ALL,quotechar='"')

summary_pivot = filtered_df.pivot_table(index=['Major Proj Cat'],columns=['PA'], values=['Total BFTE'], aggfunc='sum')
summary_pivot = summary_pivot.reset_index()
summary_pivot.to_csv('./Reports/summary_pivot_bfte_projCategory_latest_month.csv', index=False,quoting=csv.QUOTE_ALL,quotechar='"')

summary_pivot = filtered_df.pivot_table(index=['Major Proj Cat'],columns=['Sub- PA'], values=['Total BFTE'], aggfunc='sum')
summary_pivot = summary_pivot.reset_index()
summary_pivot.to_csv('./Reports/summary_pivot_bfte_projCategory_latest_month_bySubPA.csv', index=False,quoting=csv.QUOTE_ALL,quotechar='"')

summary_pivot = filtered_df.pivot_table(index=['Project_5'],columns=['Revised Component Group'], values=['Total BFTE'], aggfunc='sum')
summary_pivot = summary_pivot.reset_index()
summary_pivot.to_csv('./Reports/summary_pivot_bfte_projectname_latestmonth.csv', index=False,quoting=csv.QUOTE_ALL,quotechar='"')


# # create files for trending by employee band
# emp_types = ['E1','E2','E3','E4','E5','E6','E7','Third Party','Fixed Term Contract',"TSS Contract Trainee"]
# for emp_type in emp_types:
#     # print(type(emp_type))
#     temp_df = df[df['Employee Subgroup'] == emp_type].copy()
#     summary_pivot = temp_df.pivot_table(index=['Employee Comp Code PP_8','Revised Component Group'],columns=['Month'], values=['Total Revenue','Total BFTE'], aggfunc='sum')
#     summary_pivot = summary_pivot.reset_index()
#     summary_pivot.to_csv(f'./summary_pivot_bfte_{emp_type}_band.csv', index=False,quoting=csv.QUOTE_ALL,quotechar='"')

#
# realization_pivot = df.pivot_table(index=['Employee Comp Code PP_8', 'Employee Subgroup'],columns =['Month'],values=['Total Revenue','Total BFTE'] , aggfunc='sum')
# realization_pivot = realization_pivot.reset_index()
# realization_pivot.to_csv("./realization.csv",index=False,quoting=csv.QUOTE_ALL,quotechar='"')

