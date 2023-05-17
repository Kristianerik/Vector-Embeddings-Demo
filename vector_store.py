from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import pandas as pd
import os

#Open Api Key insert your api key here
os.environ["OPENAI_API_KEY"] = "sk-*****"

#Loading Documents
loader1 = CSVLoader(file_path='profile.csv')
loader2 = CSVLoader(file_path='portfolio.csv')
loader3 = CSVLoader(file_path='transcript.csv')

#Creating Index with documents
index_creator = VectorstoreIndexCreator()
docsearch = index_creator.from_loaders([loader1, loader2, loader3])

#Creating a QA chain
chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever, input_key = "question")

#Creating Querey and passing into chain

query = "What is the median income in this dataset?"
response = chain({"question": query})

print(response['result'])
