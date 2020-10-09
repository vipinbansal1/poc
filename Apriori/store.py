import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import math
#from efficient_apriori import apriori




dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)


transactions = [('eggs', 'bacon', 'soup'),
                ('eggs', ' banana', 'apple'),
                ('soup', 'apple', 'banana'),
                ('soup', 'milk', 'eggs'),
                ('soup', 'bread', 'eggs')]
                
#itemsets, rules = apriori(transactions, min_support=0.2,  min_confidence=0.6)
#print(rules)  # [{eggs} -> {bacon}, {soup} -> {bacon}]
#print(itemsets)
#records = []  
#for i in range(0, file.shape[0]):  
#    records.append([str(file.values[i,j]) for j in range(0, 4)])








import numpy as np
import pandas as pd
from apyori import apriori  
#store_data.csv"
def data_Preparation():
    df  = pd.read_csv("ikeafull.csv")
    print(df.head())
    df = df[df['Name'].notnull()]
    
   
    grouped = df.groupby('orderid')
    group = grouped.groups
    product_list = []
    for name,group in grouped:
        if(group.shape[0] == 1):
            continue
        plist = group['Name'].tolist()
        product_list.append(plist)
    return product_list

products = data_Preparation()


association_rules = apriori(products, 
                            min_support=0.001, 
                            min_confidence=0.6,
                            min_lift=2, 
                            max_length=3)  

print("length:",(association_rules))
association_results = list(association_rules)
print(len(association_results))

#association_results = association_results[155:]

AssociationList = []
for item in association_results:
    #print(item)
    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    #print("Rule between items: ", items)

    #second index of the inner list
    #print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list
    
    for i in range(len(item[2])):
        singleAssociation = []
        '''
        print("Associations of:", item[2][i][0])
        print(" With:", item[2][i][1])
        print("Confidence:", round(item[2][i][2],2))
        print("lift:", round(item[2][i][3],2))
        '''
        singleAssociation.append([product for product in item[2][i][0]])
        singleAssociation.append([product for product in item[2][i][1]])
        singleAssociation.append(round(item[2][i][2],2))
        singleAssociation.append(round(item[2][i][3],2))
        AssociationList.append(singleAssociation)
#print(AssociationList)


cart_product = ['ESARP','RANARP','KOPPLA']
suggested_p_list = []
for i in range(len(AssociationList)):
    if (set(cart_product).issubset(set(AssociationList[i][0]))):
        suggested_p_list.append(AssociationList[i])
print(suggested_p_list)