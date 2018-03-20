
# coding: utf-8

# In[1]:


#imports
import time
from collections import deque
import csv
import numpy as np
import tensorflow as tf
from six import next
from tensorflow.core.framework import summary_pb2
import pandas as pd
import dataio
import ops


# In[2]:


#set varibales 
np.random.seed(13575)

BATCH_SIZE = 100
USER_NUM =  206202
ITEM_NUM = 32
DIM = 15
EPOCH_MAX = 100
DEVICE = "/cpu:0"


# In[ ]:


#clip -ve values
def clip(x):
    return np.clip(x, 1.0, None)

#create scalar summary of frame
def make_scalar_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])

#read data and split test-train
def get_data():
    df = dataio.read_process("data/user_basket_size.csv", sep=",")
    df['group_indicator'] = (df.ix[:,0] != df.ix[:,0].shift(-1)).astype(int)

    df_train = df.loc[df.group_indicator==0]
    df_train = df_train.drop('group_indicator', axis=1)

    df_test =  df.loc[df.group_indicator==1]
    df_test = df_test.drop('group_indicator', axis=1)
    df = df.drop('group_indicator', axis=1)

    return df_train, df_test


# In[ ]:


#Implementaion of SVD, create Tensorflow session
def svd(train, test):
    samples_per_batch = len(train) // BATCH_SIZE
    print test.head(10)
    iter_train = dataio.ShuffleIterator([train["user"],
                                         train["days_since_prior_order"],
                                         train["basket_size"]],
                                        batch_size=BATCH_SIZE)

    iter_test = dataio.OneEpochIterator([test["user"],
                                         test["days_since_prior_order"],
                                         test["basket_size"]],
                                        batch_size=-1)

    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    days_since_prior_order_batch = tf.placeholder(tf.int32, shape=[None], name="id_days_since_prior_order")
    basket_size_batch = tf.placeholder(tf.float32, shape=[None])

    infer, regularizer = ops.inference_svd(user_batch, days_since_prior_order_batch, user_num=USER_NUM, item_num=ITEM_NUM, dim=DIM,
                                           device=DEVICE)
    global_step = tf.contrib.framework.get_or_create_global_step()
    _, train_op = ops.optimization(infer, regularizer, basket_size_batch, learning_rate=0.001, reg=0.05, device=DEVICE)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(logdir="/tmp/svd/log", graph=sess.graph)
        errors = deque(maxlen=samples_per_batch)
        start = time.time()
        min = 100
        predList = []
        actList = []
        finalPred = []
        finalAct = []
        finalpr = []
        finalac = []
        for i in range(EPOCH_MAX * samples_per_batch):

            users, days_since_prior_orders, basket_sizes = next(iter_train)
            _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                                   days_since_prior_order_batch: days_since_prior_orders,
                                                                   basket_size_batch: basket_sizes})
            pred_batch = clip(pred_batch)
            errors.append(np.power(pred_batch - basket_sizes, 2))
            if i % samples_per_batch == 0:
                train_err = np.sqrt(np.mean(errors))
                test_err2 = np.array([])
                for users, days_since_prior_orders, basket_sizes in iter_test:
                    pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                            days_since_prior_order_batch: days_since_prior_orders})
                    #pred_batch = clip(pred_batch)
                    test_err2 = np.append(test_err2, np.power(pred_batch - basket_sizes, 2))

                    pr = pred_batch
                    ac = basket_sizes
                end = time.time()
                test_err = np.sqrt(np.mean(test_err2))
                train_err_summary = make_scalar_summary("training_error", train_err)
                test_err_summary = make_scalar_summary("test_error", test_err)
                summary_writer.add_summary(train_err_summary, i)
                summary_writer.add_summary(test_err_summary, i)
                start = end

                if train_err < min:
                    min = train_err
                    finalpr = pr
                    finalac = ac

        return finalpr, finalac


# In[ ]:


if __name__ == '__main__':
    df_train, df_test = get_data()
    pr, ac = svd(df_train, df_test)
    prdf = pd.DataFrame(pr)
    acdf = pd.DataFrame(ac)
    result = pd.concat([prdf, acdf], axis=1)


# In[ ]:


import numpy as np # linear algebra
import pprint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import csv


# In[ ]:


orders = pd.read_csv("data/orders_train_test.csv")
prior = pd.read_csv("data/order_products__prior.csv")
train = pd.read_csv("data/order_products__train.csv")


# In[ ]:


frames = [prior, train]
products = result = pd.concat(frames)


# In[ ]:


orders = orders.loc[orders['group_indicator'] == 1]

# find the products that were actually bought in the last order of every user
test_orders = pd.merge(orders, products, on='order_id')

test_orders2 = test_orders[['user_id', 'order_id', 'product_id']]

# create a list of lists
# list of the last order of each user
# containing lists of the products that each user bought in his last order
test_orders2 =  test_orders2.groupby(['user_id', 'order_id'])['product_id'].apply(list)
#filename = 'actual_products.csv'
#test_orders2.to_csv(filename, index=False, encoding='utf-8', header=False)


test_set = pd.read_csv("data/test_set_.csv", names = ["user_id", "days_since_prior_order", "basket", "order_id"])
# the next dataset contains the predicted basket size of the next basket (output of svd_train_val.py)
preds = pd.read_csv("data/pred-actual.csv",  names = ["pred", "actual"])
# this dataset contains statistics concerning users' consumer behaviour
user_prod_stats = pd.read_csv("data/user_product_stats.csv")
#act_prods = pd.read_csv("data/actual_products.csv")


# In[ ]:


test_preds = pd.concat([test_set, preds], axis=1)

pred_prods =pd.DataFrame()
l=int(len(test_set))
c=int(1)
final_pred_prods = []
final_pred_prods2 = pd.DataFrame()


# In[ ]:


i = 0

# iterate through the dataframe containing the user_id and the predicted number of his next basket

# for every user check the predicted size of his next basket and accordingly predict the products that he buy

# the prediction of the next basket products depends on the following:
# 1. predicted basket size
# 2. the preferences of the user,
sample_user = ''
sample_size = ''
sample_reco = ''
for index, row in test_preds.iterrows():
     user_stats = []
     basket_size = int(round(row['pred'],0))
     user = row['user_id']

     user_stats = user_prod_stats.loc[user_prod_stats['user_id'] == user]
     user_products = user_stats['product_id']

     pred_prods  =  user_products.head(basket_size)
     df_row = pred_prods.tolist()
     sample_user = user
     sample_size = basket_size
     sample_reco = pred_prods
     final_pred_prods.append(df_row)

     i = i+1

print sample_user
print sample_size
print sample_reco
# print type(final_pred_prods)
# print type(final_pred_prods[1])

# print 'results'
# for xs in final_pred_prods:
#     print ",".join(map(str, xs))


# In[ ]:


# create a list of lists
# list of the last order of each user
# containing lists of the predicted products for this order
with open('data/pred_products.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #wr = ",".join(map(str, wr))
    wr.writerow(final_pred_prods)


# In[ ]:



import csv

#preds = pd.read_csv("pred_products2.csv")
predss = []


# read CSV file & load into list
with open("data/pred_products.csv", 'r') as my_file:
    reader = csv.reader(my_file, delimiter='\t')
    preds = list(reader)

with open("data/actual_products.csv", 'r') as my_file:
    reader = csv.reader(my_file, delimiter='\t')
    acts = list(reader)

acts = [l[0] for l in acts]


# In[ ]:


TTP = 0
TFP = 0
TFN = 0
TT = 0

i= 0


# In[ ]:


for pred, act in zip(preds, acts):
        act = str(act)
        act = act.replace(" ", "")
        pred = str(pred)

        pred = pred.replace(" ", "")
        pred = pred.replace("'", "")
        pred = pred.replace("[", "")
        pred = pred.replace("]", "")
        act = act.replace("[", "")
        act = act.replace("]", "")

        act = act.split(",")
        pred = pred.split(",")


        pred = set(pred)
        act = set(act)

        TP = len(set.intersection(act, pred))

        UN = len(set.union(act, pred))

        FP = len(pred)-TP
        FN = len(act)-TP
        T = len(act)

        AC = TP/float(T)
        #print TP, UN, FP, FN, T
        TTP=TTP+TP
        TFP=TFP+FP
        TFN=TFN+FN
        TT=TT+T
        #print TTP, TFP, TFN, TT


# In[ ]:


TAC = TTP/float(TT)
#print TTP,TT
PRE = TTP/float((TTP+TFP))
REC = TTP/float((TTP+TFN))
F1 = (2*(PRE*REC))/float((PRE+REC))

i = i+1
print 'true positives', TTP, '\nfalse positives', TFP, '\nfalse negatives', TFN, '\ntotal products bought', TT
print '\naccuracy', TAC, '\nprecision', PRE, '\nrecall', REC, '\nf1', F1

