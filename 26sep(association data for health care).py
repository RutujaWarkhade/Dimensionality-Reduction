# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 08:36:05 2024

@author: om
"""
#import required libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder

#sample dataset
transations=[
    ['Milk','Bread','Butter'],
    ['Bread','Eggs'],
    ['Milk','Bread','Eggs','Butter'],
    ['Bread','Eggs','Butter'],
    ['Milk','Break','Eggs']
    ]
#step1:convert the dataset into a format suitable for Apriori 
    
te=TransactionEncoder()
te_ary=te.fit(transations).transform(transations)
df=pd.DataFrame(te_ary, columns=te.columns_)

#step2:Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets=apriori(df, min_support=0.5, use_colnames=True)

#step3: Generate assosiation rules from the frequent itemsets
rules=association_rules(frequent_itemsets, metric="lift", min_threshold=1)

#step4: Output the results
print("Frequent Itemsets")
print(frequent_itemsets)

print("\nAssociation Rules")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]) 


#import required libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder

#sample dataset
transations=[
    ['Fever','Cough','COVID-19'],
    ['Cough','Sore Throat', 'Flu'],
    ['Fever','Cough','Shortness of Breath','COVID-19'],
    ['Cough','Sore Throat','Flu','Headache'],
    ['Fever','Body Ache','Flu'],
    ['Fever','Cough','COVID-19','Shortness of Breath'],
    ['Sore Throat','Headache','Cough'],
    ['Body Ache','Fatigue','Flu']
    ]

#step1:convert the dataset into a format suitable for Apriori 
    
te=TransactionEncoder()
te_ary=te.fit(transations).transform(transations)
df=pd.DataFrame(te_ary, columns=te.columns_)

#step2:Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets=apriori(df, min_support=0.3, use_colnames=True)

#step3: Generate assosiation rules from the frequent itemsets
rules=association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)

#step4: Output the results
print("Frequent Itemsets")
print(frequent_itemsets)

print("\nAssociation Rules")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]) 
"""


"""


import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder

#sample dataset
transations=[
    ['Laptop','Mouse','Keyboard'],
    ['Smartphone','Headphones'],
    ['Laptop','Mouse','Headphones'],
    ['Smartphone','Charger','Phone Case'],
    ['Laptop','Mouse','Monitor'],
    ['Smartphone','Charger','Phone Case'],
    ['Mouse','Keyboard','Monitor'],
    ['Smartphone','Headphones','Smartwatch']
    ]

#step1:convert the dataset into a format suitable for Apriori 
    
te=TransactionEncoder()
te_ary=te.fit(transations).transform(transations)
df=pd.DataFrame(te_ary, columns=te.columns_)

#step2:Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets=apriori(df, min_support=0.2, use_colnames=True)

#step3: Generate assosiation rules from the frequent itemsets
rules=association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)

#step4: Output the results
print("Frequent Itemsets")
print(frequent_itemsets)

print("\nAssociation Rules")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]) 







