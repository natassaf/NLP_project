# NLP_project
This repository contains a project done as capstone project during my master. It is a recommendation system which given information on a  candidate uses a deep learning model called Deep Relevance Matching Model to suggest the most relevant jobs from a database of job announcements. In order to use the deep learning model we first retrieve the top 100 most relevant jobs using Elasticsearch engine. The way that the data are indexed within Elasticsearch is not demonstrated here but the data can be ingested using logstash. 

### Data
The data used on this project are not publicly available due to confidentiality agreement. It is possible though to find a detailed description on the capstone project pdf. 

### DRMM
The paper describing Deep Relevance Matching Model is by Guo et al. and can be found in the link below:

https://dl.acm.org/citation.cfm?id=2983769

For the implementation of DRMM dynet tool was used and multiple versions were calculated as mentioned in the experimental results of the pdf.
But only one version is uploaded containing lines needed to be added to construct variations of the main model in comments.

### Dependencies
python 3

dynet

cuda
