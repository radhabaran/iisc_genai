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