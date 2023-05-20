# ClothingSimilarity Search
 
Overview: 
- Using webscraping (Selenium) from 3 sites to extract data about apparel and the corresponding links.
- Common preprocessing applied on the sentences extracted.
- State-of-the-art Sentence BERT used to find out similarity of the description given by the user to the database we already have and suggested the top N=10 links.
- Deployment code deployed on Google Cloud Run.

Contents:
- Project.ipynb:   Consists of the main code used to web scrape data, do prepprocessing, and prepare csv files.
- dataset.csv:     Contains the description and links of apparels from multiple sites.
- embeddings.csv:  Contains the sentence embeddinigs of the descriptions
- imgs:            Images of those sites showing html code which is helpful for web scraping   

Deployment Code: 
- A Docker file,
- Requirements file mentioning all the required libraries
- the .csv files mentioned above
- main.py file which does the computation
- test.py file to confirm if code has been deployed correctly or not.


Link:
https://getprediction-bhdhvw323q-el.a.run.app
