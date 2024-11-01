# Install Dependencies
# %%capture
# pip -q install openai
# pip -q install langchain-openai
# pip -q install langchain-core
# pip -q install langchain-community
# pip -q install sentence-transformers
# pip -q install langchain-huggingface
# pip -q install langchain_experimental
# ************************************************************************************************
# Import required packages
# ************************************************************************************************
import os
import openai
import pickle
import sqlite3
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import Timestamp
from getpass import getpass
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool

# ************************************************************************************************
# Understanding the data
# ************************************************************************************************
# Nifty 50 Constituent Price Data
csp = pickle.load(open(r"G:\My Drive\AI Datasets\constituent_stock_prices.pkl", 'rb'))

# Its a dictionary with each company-name as a key
print(type(csp))
print('*' * 100)

print(csp.keys())    # each key contains a dataframe
print('*' * 100)
# Total companies
print(len(csp.keys()))
print('*' * 100)

# Checking no. of rows and columns in each dataframe
rows = []
cols = []

for key in csp.keys():
    print(f"{key:<15} {csp[key].shape}")
    rows.append(csp[key].shape[0])
    cols.append(csp[key].shape[1])

# Visualize no. of rows in each dataframe
plt.figure(figsize=(20,4))
sns.barplot(x=csp.keys(), y=rows, hue=csp.keys())
plt.xticks(rotation=80)
plt.show()

# Checking a dataframe
adani = csp['ADANIENT.NS']
print(adani.shape)
print(adani.head())
print('*' * 100)

# Checking TCS dataframe and plot 'Close' column values
tcs = csp['TCS.NS']
print('tcs : ', tcs.shape)
print(tcs.head())
print('*' * 100)
tcs['Close'].plot(kind='line', figsize=(8, 4), title='tcs stock Close value')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()

# Checking Wipro dataframe and plot 'Close' column values
wipro = csp['WIPRO.NS']
print('wipro : ', wipro.shape)
print(wipro.head())
print('*' * 100)
wipro['Close'].plot(kind='line', figsize=(8, 4), title='Wipro stock Close value')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()

# Combine Prices data

comb_df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol'])

for key in csp.keys():
    tmp_df = csp[key].copy()
    tmp_df.reset_index(inplace=True)
    tmp_df['Symbol'] = key.split('.')[0]
    comb_df = pd.concat([comb_df, tmp_df], ignore_index=True)

print(comb_df.head())
print('*' * 100)
print(comb_df.shape)
print('*' * 100)
# Cross-check no. of total rows
print(sum(rows))
print('*' * 100)
print(comb_df.tail())
print('*' * 100)
# Save the combined dataframe as csv file
comb_df.to_csv('all_stock_prices.csv', index=False)
print(comb_df.info())
print('*' * 100)

# ************************************************************************************************
# Create a SQLite Database (in-memory)
# ************************************************************************************************

print(sqlite3.sqlite_version)
print('*' * 100)

# Connect to a sqlite DB (It will create it if it doesn't exists)
conn = sqlite3.connect('stock_db.sqlite')
print("Opened database successfully")
print('*' * 100)

#Create table stock_prices
print(comb_df.columns)

# Create a table 'stock_prices' in DB
conn.execute('''
CREATE TABLE IF NOT EXISTS stock_prices(
                      date DATE,
                      open DOUBLE,
                      high DOUBLE,
                      low DOUBLE,
                      close DOUBLE,
                      volume INT,
                      symbol VARCHAR(20));''')

conn.commit()

print("Table created successfully");
print('*' * 100)

# Show tables

cursor = conn.execute('''
SELECT name FROM sqlite_master WHERE type='table';
''')

for row in cursor:
    print(row)
print('*' * 100)

# ************************************************************************************************
# Insert data into stock_prices table
# ************************************************************************************************

# Function to convert the 'date' from Timestamp to String yyyy-mm-dd

def convert_date(date):
    if isinstance(date, Timestamp):
        return date.strftime('%Y-%m-%d')
    else:
        raise ValueError("Input must be a Pandas Timestamp object")

print(f"Original Date : {comb_df['Date'][0]} , Converted Date : {convert_date(comb_df['Date'][0])}")
print('*' * 100)

comb_df['Date'] = comb_df['Date'].apply(convert_date)
print(comb_df.head(3))
print('*' * 100)


# Insert stock prices data

conn.executemany('''
INSERT INTO stock_prices (date, open, high, low, close, volume, symbol) VALUES (?, ?, ?, ?, ?, ?, ?)
''', comb_df.values)

conn.commit()

print("Data inserted successfully!")
print('*' * 100)

# ************************************************************************************************
# Query the Database
# ************************************************************************************************

# Show table content

cursor = conn.execute('''
SELECT * from stock_prices limit 10;
''')

for row in cursor:
    print(row)
print('*' * 100)

# Show table content for WIPRO stock

cursor = conn.execute('''
SELECT * from stock_prices WHERE symbol='WIPRO' limit 10;
''')

for row in cursor:
    print(row)
print('*' * 100)

# Show number of rows in table

cursor = conn.execute('''
SELECT count(*) from stock_prices;
''')

for row in cursor:
    print(row)
print('*' * 100)

# Show columns info of table

cursor = conn.execute('''
PRAGMA table_info(stock_prices);
''')

for row in cursor:
    print(row)
print('*' * 100)

# ************************************************************************************************
# SQL Query Generation using an LLM
# ************************************************************************************************

# Read OpenAI key
f = open('G:\My Drive\AI Datasets\key\openaikey.txt')
api_key = f.read().strip()          # Remove Blank Spaces
os.environ['OPENAI_API_KEY'] = api_key
openai.api_key= os.getenv('OPENAI_API_KEY')

# Load the model

llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)

output = llm.invoke("what is 2 plus 3")
print('test output from LLM : what is 2 plus 3 ? Ans : ', output)
print('*' * 100)

# Create Chain for Query Generation using LCEL (LangChain Expression Language)

# Build a prompt
template = """Use the following pieces of context to generate the SQL query / 
with column names for the request given at the end. / 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Request: {request}
Generate query:"""

PROMPT = PromptTemplate(input_variables=['context', 'request'], template=template)

# Query Generation Chain - created using LCEL (LangChain Expression Language)

chain = (PROMPT
         | llm
         | StrOutputParser()       # to get output in a more usable format
         )
# Context
table_info = """CREATE TABLE IF NOT EXISTS stock_prices (date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume INT, symbol VARCHAR(20));"""

# Generate SQL Query then query the Database

# Generate sql query

response1 = chain.invoke({"request": "How many total records there in table?",
                          "context": table_info})

print(response1)
print('*' * 100)

# Generate another query

response2 = chain.invoke({"request": "What are unique companies symbols present in table?",
                          "context": table_info})

print(response2)
print('*' * 100)

# Generate another query

response3 = chain.invoke({"request": "Give me any ten records for Wipro company?",
                          "context": table_info})

print(response3)
print('*' * 100)

# Generate another query

response4 = chain.invoke({"request": "Need open, high prices for any ten Wipro records",
                          "context": table_info})

print(response4)
print('*' * 100)

# ************************************************************************************************
# Query the Database using generated SQL queries
# ************************************************************************************************

def format_query(query):
    query = re.sub(r"```sql\n|\n```", "", query).strip()
    query = re.sub(r"\n", " ", query).strip()
    return query

print(format_query(response1))

# Use generated query to get data from database

query = format_query(response1)
cursor = conn.execute(query)

for row in cursor:
    print(row)
print('*' * 100)

# Generated Response2
print(format_query(response2))
print('*' * 100)

# Use generated query to get data from database

query = format_query(response2)
cursor = conn.execute(query)

for row in cursor:
    print(row)
print('*' * 100)

# Generated Response3
print(format_query(response3))
print('*' * 100)
# Use generated query to get data from database

query = format_query(response3)
cursor = conn.execute(query)

for row in cursor:
    print(row)
print('*' * 100)


# Generated Response4
print(format_query(response4))
print('*' * 100)

# Use generated query to get data from database

query = format_query(response4)
cursor = conn.execute(query)

for row in cursor:
    print(row)
print('*' * 100)

# ************************************************************************************************
# Python Code generation using an LLM
# ************************************************************************************************
# The Python REPL is a powerful tool for interactive programming, making it easy to experiment 
# with Python code and get immediate feedback. # It allows you to enter Python code, evaluate it
# immediately, and see the results right away.
#
# Components of Python REPL:
#
# Read: The REPL reads the input you provide. This can be a single line of code, a block of code, 
# or even multiple commands in succession.
#
# Eval: The input is evaluated (executed) by the Python interpreter. This means that the code you 
# entered is processed, and any calculations or operations are performed.

# Print: The results of the evaluation are printed to the screen. This allows you to see the 
# output immediately after entering your code.

# Loop: After printing the result, the REPL loops back to read the next input, allowing you to
# continue interacting with the interpreter.
#
# Libraries to be imported
# from langchain_experimental.utilities import PythonREPL
# from langchain_core.tools import Tool
# ************************************************************************************************

python_repl = PythonREPL()
print(python_repl.run("print(1+1)"))
print('*' * 100)

# You can create the tool to pass to an agent

repl_tool = Tool(
    name="python_repl",
    description="""A Python shell. Use this to execute python commands. Input should be a valid /
     python command. If you want to see the output of a value, you should print it out with `print(...)`.""",
    func=python_repl.run,
)

print(repl_tool.run("print(1+1)"))
print('*' * 100)

# To plot a sine wave

repl_tool.run("""

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.show()

""")

# Generate code using llm

response = llm.invoke('generate and return python code only, no additional text, a code to create sine waves')
print(response.content)
print('*' * 100)

# Execute the generated code

repl_tool.run(response.content)

# ************************************************************************************************
# Create another Chain for Code Generation for Stock Prices
# ************************************************************************************************
# Build prompt to use the user_request, generated_sql_query, extracted_data to generate Python 
# code for insights

template2 = """Use the following pieces of user request and sql query to generate python code /
 to show insights related to the data given at the end.
Generate and return python code only, no additional text.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Request: {request}
Sql query: {sql_query}
Data: {data}
Generate code:"""

PROMPT2 = PromptTemplate(input_variables=["request", "sql_query", "data"], template=template2)

# Code Generation Chain

chain2 = (PROMPT2
          | llm
          | StrOutputParser()
          )

# First, Generate SQL Query for a user request
# Generate sql query

user_request = "Need insights on the trend present in any 50 Wipro records"

generated_query = chain.invoke({"request": user_request,
                                "context": table_info})

print(generated_query)
print('*' * 100)
print(format_query(generated_query))
print('*' * 100)

# Use generated query to get data from database

query = format_query(generated_query)
print(query)
print('*' * 100)
cursor = conn.execute(query)

extracted_data = []

for row in cursor:
    extracted_data.append(row)

print(extracted_data[:2])
print('*' * 100)

# Generate code

response1A = chain2.invoke({"request": user_request,
                            "sql_query": generated_query,
                            "data": extracted_data
                            })

# See the Generated code
print(response1A)
print('*' * 100)

# Execute the generated Code for the user request
# Execute the generated code

print(repl_tool.run(response1A))