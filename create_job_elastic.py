#This script creates the csv file to feed into logstash and insert the data into elasticsearch engine set locally
import joblib

pickling_on = open("C:/Users/nataz/Downloads/job_prop_data/dataset/jobs_df", 'rb')
jobs_df = joblib.load(pickling_on)
pickling_on.close()

jobs_df.to_csv("C:/Users/nataz/Downloads/job_prop_data/dataset/elastic/jobs_index.csv", sep='|', encoding='utf-8-sig',header=False, index=False)