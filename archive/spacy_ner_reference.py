#!/usr/bin/env python
# coding: utf-8

# # Resume NER
# ## Extract Information from Resumes using NER (Named Entity Recognition)

# ### Part 2 - NER with Spacy
# In this second part of the challenge, we will be using the preprocessed data from part one to start training NER models. We will be using spacy (https://spacy.io/) here to "get our feet wet" with NER, as training spacy can be reasonably done on our laptops and does not yet necessarily require a GPU. Spacy is a powerful, effective, and resource-efficient NLP library - It might surprise us with its performance on the challenge!
# 
# We will run spacy's pretrained models on our data to get a feel for NER, and then we will perform some additional preprocessing on our data before we start training our own NER model using the labelled entities we have identified in part one. 
# We will also explore evaluation metrics for NER, and decide how we want to quantify the performance of our trained models. 
# 
# * *If you need help setting up python or running this notebook, please get help from the  assistants to the professor*
# * *It might be helpful to try your code out first in a python ide like pycharm before copying it an running it here in this notebook*
# * *For solving the programming tasks, use the python reference linked here (Help->Python Reference) as well as Web-searches.* 

# ##### Reload preprocessed data
# Here, we will load the data we saved in part one and save it to a variable. Provide code below to load the data and store it as a list in a variable. (Hint - use 'open' and the json module)

## import json module
import json
import spacy
from flair.data import Sentence
from flair.models import SequenceTagger


def load_resumes():
    path = "../dataset/converted_resumes.json"
    ## TODO open file and load as json
    with open(path,encoding="utf8") as f:
        resumes = json.load(f)
    ## TODO print length of loaded resumes list to be sure everything ok
    print("Loaded: {} resumes".format(len(resumes)))
    return resumes


# ##### Take Spacy for a spin
# Before we train our own NER model to recognize the resume-specific entities we want to capture, let's see how spacy's pretrained NER models do on our data. These pretrained models can't recognize our entities yet, but let's see how they do. Run the next code block to load spacy's English language model 
def load_spacy():
    nlp = spacy.load('en')
    print(nlp)


# Now we get the EntityRecognizer in the loaded nlp pipeline and display the labels it supports

def ner_labels(nlp):
    ner = nlp.get_pipe('ner')
    labels = ner.labels
    print(labels)
    return ner


# ##### Question: What do the 'GPE', 'FAC' and 'NORP' labels stand for? (Tipp: use either the spacy.explain method, or google the spacy.io api docs) 
# *Answer here*

# In[4]:


### if you choose to use spacy's 'explain' method to get the answer to the question above, provide your code here
#for label in labels:
#    print("{}:  {}".format(label,spacy.explain(label)))


# As we can see, the entities are different than the entities we will train our custom model on. 
# ##### Question: what entities do you think this model will find in an example resume?
# *Answer here*

# Now we will work with one of our resumes, and get spacy to tell us what entities it recognizes. Complete the code block below to get a single resume text out of our resume list. 

# In[5]:


### TODO get a single resume text and print it out
#restxt = resumes[200][0]
#print("\n".join(restxt.split('\n\n')))


# Extracting entities with spacy is easy with a pretrained model. We simply call the model (here 'nlp') with our text to get a spacy Document. See https://spacy.io/api/doc for more detail. 
# 
# Execute the code below to process the resume txt.

# In[6]:


#doc = nlp(restxt)


# The doc object has a list of entities predicted by spacy 'ents'. We would like to loop through all of these entities and print their label and associated text to see what spacy predicted for this resume.
# 
# Complete the code below to do this. You will probably need to google the spacy api docs to find the solution (Tipp: look for 'Doc.ents'). Also, trying code in your ide (for example pycharm) before copying it here might help with exploring and debugging to find the solution. 

# In[7]:


##TODO loop through the doc's entities, and print the label and text for each entity found. 
#for ent in doc.ents:
#    print("{:10} {}".format(ent.label_,ent))


# ##### Questions: What is your first impression of spacy's NER based on the results above? Does it seem accurate/powerfull?
# ##### Does it make many mistakes? Do some entity types seem more accurate than others? 
# *Answers here*

# Now as a comparison, we will list the entities contained in the resume's original annotated training data (remember, the existing annotations were created by a human-annotator, and not predicted by a machine like the entities predicted above) 
# 
# Complete the code below to do the following: 
# * Access the 'entities' list of the example resume you chose, loop through the entities and print them out. 
# * *Tip: one entity in the list is a tuple with the following structure: (12,1222,"label") where the first element is the start index of the entity in the resume text, the second element is the end index, and the third element is the label.
# * Use this Tip to print out a formatted list of entities 
# 
# 

# In[8]:


##TODO access entities
#res = resumes[200]
#restext = res[0]
#labeled_ents = res[1]['entities']
## TDOD print out formatted list of entities
#for ent in labeled_ents:
#    enttext = restext[ent[0]:ent[1]]
#    enttext = " ".join(enttext.split())
#    print("{:20} {}".format(ent[2],enttext))


# As we already know, the annotated entities in the training data are different than the entities spacy can recognize with it's pretrainied models, so we need to train a custom NER model. We will get started with that now. 

# ##### Prepare Training Data for NER model training
# We need to do some more preprocessing of our training data before we can train our model.

# Remember the entity labels you chose in part 1 of the challenge? We will be training a model to predict those entities.
# As a first step, we will gather all resumes that contain at least one training annotation for those entities.
# 
# Complete and execute the code below to gather your training data. 

# In[25]:


##TODO Store the entity labels you want to train for as array in chosen_entity_labels
#chosen_entity_labels = ["Companies worked at","Degree","Skills","Designation"]
chosen_entity_labels = ["Companies worked at","Degree","Designation"]

## this method gathers all resumes which have all of the chosen entites above.
def gather_candidates(dataset,entity_labels):
    candidates = list()
    for resume in dataset:
        res_ent_labels = list(zip(*resume[1]["entities"]))[2]
        if set(entity_labels).issubset(res_ent_labels):
            candidates.append(resume)
    return candidates
## TODO use the gather candidates methods and store result in training_data variable
#training_data = gather_candidates(resumes,chosen_entity_labels)
#print("Gathered {} training examples".format(len(training_data)))


# Now we have those training examples which contain the entities we are interested in. Do you have at least a few hundred examples? If not, you might need to re-think the entities you chose or try just one or two of them and re-run the notebooks. It is important that we have several hundred examples for training (e.g. more than 200. 3-500 is better). 

# ##### Remove other entity annotations from training data
# Now that we have our training data, we want to remove all but relevant (chosen) entity annotations from this data, so that the model we train will only train for our entities. Complete and execute the code below to do this. 

# In[26]:


## filter all annotation based on filter list
def filtered_data(training_data):
    def filter_ents(ents, filter):
        filtered = [ent for ent in ents if ent[2] in filter]
        return filtered
    X = [[dat[0], dict(entities=filter_ents(dat[1]['entities'], chosen_entity_labels))] for dat in training_data]
    return X

## now remove all but relevant (chosen) entity annotations and store in X variable


# ##### Remove resumes that cause errors in spacy
# Depending on what entities you chose, some of the resumes might cause errors in spacy. We don't need to get into details as to why, suffice to say it has to do with whitespace and syntax in the entity annotations. If these resumes are not removed from our training data, spacy will throw an exception during training, so we need to remove them first. 
# 
# We will use the remove_bad_data function below to do this. This function does the following:
# * calls train_spacy_ner with debug=True and n_iter=1. This causes spacy to process the documents one-by-one, and gather the documents that throw an exception in a list of "bad docs" which it returns. 
# * You will complete the function to remove any baddocs (returned by remove_bad_data) from your training data list. 
# 
# You may or may not have any bad documents depending on the entities you chose. In any case, there should not be more than a dozen or so bad docs.  

# In[27]:


from resume_ner.spacy_train_resume_ner import train_spacy_ner

def remove_bad_data(training_data):
    model, baddocs = train_spacy_ner(training_data, debug=True, n_iter=1)
    ## training data is list of lists with each list containing a text and annotations
    ## baddocs is a set of strings/resume texts.
    ## TODO complete implementation to filter bad docs and store filter result (good docs) in filtered variable
    filtered = [data for data in training_data if data[0] not in baddocs]
    print("Unfiltered training data size: ",len(training_data))
    print("Filtered training data size: ", len(filtered))
    print("Bad data size: ", len(baddocs))
    return filtered

## call remove method. It may take a few minutes for the method to complete.
## you will know it is complete when the print output above. 
#X = remove_bad_data(X)


# ##### Question: How many bad docs did you have? What is the size of your new (filtered) training data? 
# *Answer here*

# ##### Train/Test Split
# Now before we train our model, we have to split our available training data into training and test sets. Splitting our data into train and test (or holdout) datasets is a fundamental technique in machine learning, and essential to avoid the problem of overfitting.
# Before we go on, you should get a grasp of how train/test split helps us avoid overfitting. Please take the time now to do a quick web search on the topic. There are many resources available. You should search for "train test validation overfitting" or some subset of those terms.
# 
# Here are a few articles to start with:
# * https://machinelearningmastery.com/a-simple-intuition-for-overfitting/
# * https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation (Note: you are free to install scikit learn and use the train_test_split method documented here, but it is not necessary. It is the concept that is important)
# 
# ##### Question: What is overfitting and how does doing a train/test split help us avoid overfitting when training our models? Please answer in your own words. 
# *Answer here*

# Now that we understand why we do a train/test split, we will write some code that splits our data into train and test sets. Usually we want around 70-80% of the data for train, and the rest for test. 
# ##### TODO: Complete the code below to create a train and test dataset

# In[28]:


##TODO complete the implementation  of the train test split function below
def train_test_split(X,train_percent):
    train_size = int(len(X)*train_percent)
    train = X[:train_size]
    test = X[train_size:]
    assert len(train)+len(test)==len(X)
    return train,test
## do train test split
#train,test = train_test_split(X,0.7)
## TODO print the size of train and test sets. Do they add up to length of X? 
#print("Train size: ",len(train))
#print("Test size: ",len(test))
#print("X size: ",len(X))


# ##### Train a spacy ner model with our training data
# OK, now it is (finally) time to train our own custom NER model using spacy. Because our training data has been preprocessed to only include annotations for the entities we are interested in, the model will only be able to predict/extract those entities. 
# *Depending on your computer, this step may take a while.* Training 20 epochs (iterations) using 480 training examples takes around 10 minutes on my machine (core i7 CPU). You will see output like *Losses {'ner':2342.23342342}* after each epoch/iteration. The default number of iterations is 20, so you will see this output 20 times. When this step is done, we will use the trained ner model to perform predictions on our test data in our test set.  

# In[29]:



## run this code to train a ner model using spacy
#custom_nlp,_= train_spacy_ner(train,n_iter=20)


# ##### Inspect NER predictions on one sample resume
# Now that we have a trained model, let's see how it works on one of our resumes. 

# In[30]:


## TODO fetch one resume out of our test dataset and store to the "resume" variable
#resume = test[73]
## TODO create a spacy doc out of the resume using our trained model and save to the "doc" variable 
#doc = custom_nlp(resume[0])


# Now we will output the predicted entities and the existing annotated entities in that doc

# In[31]:


## TODO output predicted entities (in "ents" variable of the spacy doc created above)
#print("PREDICTED:")
#for ent in doc.ents:
#    print("{:20} {}".format(ent.label_,ent))
#print()
## TODO output labeled entities (in "entities" dictionary of resume)
#print("LABELED:")
#for ent in resume[1]["entities"]:
#    print("{:20} {}".format(ent[2],resume[0][ent[0]:ent[1]]))


# #### Evaluation Metrics for NER
# Now that we can predict entities using our trained model, we can compare our predictions with the original annotations in our training data to evaluate how well our model performs for our task. The original annotations have been annotated manually by human annotators, and represent a "Gold Standard" against which we can compare our predictions. 
# 
# For a simple classification task, the most common evaluation metrics are:
# * accuracy
# * precision
# * recall
# * f1 score
# 
# In order to understand these metrics, we need to understand the following concepts:
# * True positives - How many of the predicted entities are "true" according to the Gold Standard? (training annotation) 
# * True negatives - How many entities did the model not predict which are actually not entities according to the Gold Standard?
# * False positives - How many entities did the model predict which are NOT entities according to the Gold Standard?  
# * False negatives - How many entities did the model "miss" - e.g. did not recognize as entities which are entities according to the Gold Standard? 
# 
# Before we go on, it is important that you understand true/false positives/negatives as well as the evaluation metrics above. Take some time now to research the web in order to find answers to the following questions:
# 
# ##### Question: How are the evaluation metrics above defined in the context of evaluating Machine Learning models? How do they relate to True/False Positives/Negatives above? Please provide an intuitive description as well as the mathmatical formula for each metric. 
# *Answers here*
# 

# ##### Calculating Metrics based on token-level annotations or full entity-level. 
# The concepts above are our first step toward understanding how to evaluate our model effectively. However, in NER, we need to take into account that we can calculate our metrics either based on all tokens (words) found in the document, or only on the entities found in the document.  
# 
# ##### Token-Level evaluation. 
# Token level evaluation evaluates how accurately did the model tag *each individual word/token* in the input. In order to understand this, we need to understand something called the "BILUO" Scheme (or BILOU or BIO). The spacy docs have a good reference. Please read and familiarize yourself with BILUO. 
# 
# https://spacy.io/api/annotation#biluo
# 
# Up to now, we have not been working with the BILUO scheme, but with "offsets" (for example: (112,150,"Email") - which says there is an "Email" entity between positions 112 and 150 in the text). We would like to be able to evaluate our models on a token-level using BILUO - so we need to convert our data to BILUO. Fortunately, Spacy provides a helper method to do this for us.
# 
# *Execute the code below to see how our "Gold Standard" and predictions for our example doc above look in BILUO scheme.* 
# Note: some of the lines might be ommited for display purposes. 

# In[32]:


from spacy.gold import biluo_tags_from_offsets
import pandas as pd
from IPython.display import display, HTML

## returns a pandas dataframe with tokens, prediction, and true (Gold Standard) annotations of tokens
def make_bilou_df(nlp,resume):
    doc = nlp(resume[0])
    bilou_ents_predicted = biluo_tags_from_offsets(doc, [(ent.start_char,ent.end_char,ent.label_)for ent in doc.ents])
    bilou_ents_true = biluo_tags_from_offsets(doc,
                                                   [(ent[0], ent[1], ent[2]) for ent in resume[1]["entities"]])

    
    doc_tokens = [tok.text for tok in doc]
    bilou_df = pd.DataFrame()
    bilou_df["Tokens"] =doc_tokens
    bilou_df["Tokens"] = bilou_df["Tokens"].str.replace("\\s+","")
    bilou_df["Predicted"] = bilou_ents_predicted
    bilou_df["True"] = bilou_ents_true
    return bilou_df

#bilou_df = make_bilou_df(custom_nlp,test[110])
#display(bilou_df)


# Based on this output, it should be very easy to calculate a token-level accuracy. We simply compare the "Predicted" to "True" columns and calculate what percentage are the same. 

# In[33]:


## TODO bilou_df is a pandas dataframe. Use pandas dataframe api to get a subset where predicted and true are the same. 
#same_df = bilou_df[bilou_df["Predicted"]==bilou_df["True"]]
## accuracy is the length of this subset divided by the length of bilou_df
#accuracy = float(same_df.shape[0])/bilou_df.shape[0]
#print("Accuracy on one resume: ",accuracy)


# The accuracy might seem pretty good... if it is not 100%, then let's print out those tokens where the model predicted something different than the gold standard by running the code below. 
# 
# Note - if your score on one doc is 100%, pick another document and re-run the last few cells above. 

# In[34]:


## find all rows in bilou_df where "Predicted" not equal to "True" column. 
#diff_df = bilou_df[bilou_df["Predicted"]!=bilou_df["True"]]
#display(diff_df)


# Now let's calculate the accuracy on all our test resumes and average them for an accuracy score. 
# 
# Please complete the code below to report an accuracy score on our test resumes

# In[35]:


import numpy as np
def accuracy(nlp,test):
    doc_accuracy = []
    for tres in test:
        tres_df = make_bilou_df(nlp,tres)
        same_df = tres_df[tres_df["Predicted"]==tres_df["True"]]
        ## accuracy is len of same_df/len of bilou_df
        accuracy = float(same_df.shape[0])/tres_df.shape[0]
        doc_accuracy.append(accuracy)

    total_acc = np.mean(doc_accuracy)
    print("Accuracy: ",total_acc)

    


# ##### Question: how does the model perform on token-level accuracy? What did it miss? In those cases where the predictions didn't match the gold standard, were the predictions plausible or just "spurious" (wrong)? 
# *Answer here* 

# ##### Question: What might the advantages and disadvantages be of calculating accuracy on token-level? Hint: think about a document with 1000 tokens where only 10 tokens are annotated as entities. What might the accuracy be on such a document?  

# ##### Entity-Level evaluation #####
# Another method of evaluating the performance of our NER model is to calculate metrics not on token-level, but on entity level. There is a good blog article that describes this method. 
# 
# http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
# 
# The article goes into some detail, the most important part is the scenarios described in the section "Comparing NER system output and golden standard". 

# ##### Question: how do the first 3 scenarios described in the section "Comparing NER system output and golden standard" correlate to  true/false positives/negatives? 
# *Answer here* 

# ##### Precision, Recall, F1 #####

# Now we would like to calculate precision, recall, and f1 for each entity type we are interested in (our chosen entities). To do this, we need to understand the formulas for each. A good article for this is https://skymind.ai/wiki/accuracy-precision-recall-f1. 

# ##### Question: how can we calculate precision, recall and f1 score based on the information above? Please provide the formulas for each #####
# *Answer here*

# Now supply code below which calculates precision and recall and F1 on our test data for each entity type we are interested in. 
# 
# 

# In[42]:


## TODO cycle through chosen_entity_labels and calculate metrics for each entity using test data
## Tip: use make_bilou_df on each resume in our test set, and calculate for each entity true and false positives, and false negatives. 
## Then use the formulas you learned to calculate metrics and print them out
## Also - store the precisions, recalls, and f1s for each entity in a data structure to access later. 
def calc_metrics(custom_nlp,test):
    data = []
    for label in chosen_entity_labels:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for tres in test:
            tres_df = make_bilou_df(custom_nlp,tres)
            tp = tres_df[(tres_df["Predicted"]==tres_df["True"]) & (tres_df["Predicted"].str.contains(label))]
            fp = tres_df[(tres_df["Predicted"]!=tres_df["True"]) & (tres_df["Predicted"].str.contains(label))]
            fn = tres_df[(tres_df["Predicted"]!=tres_df["True"]) & (tres_df["True"].str.contains(label))]
            true_positives += tp.shape[0]
            false_positives += fp.shape[0]
            false_negatives += fn.shape[0]


        print("For label '{}' tp: {} fp: {} fn: {}".format(label,true_positives,false_positives,false_negatives))

        precision = 0.0 if true_positives==0 else float(true_positives)/(true_positives+false_positives)
        recall = 0.0 if true_positives==0 else float(true_positives)/(true_positives+false_negatives)
        f1 = 0.0 if precision+recall==0 else 2*((precision*recall)/(precision+recall))
        #print("Precision: ",precision)
        #print("Recall: ",recall)
        #print("F1: ",f1)
        row = [precision,recall,f1]
        data.append(row)

    metric_df = pd.DataFrame(data,index=chosen_entity_labels,columns=["Precision","Recall","f1"])
    #display(metric_df)
    return metric_df


# If you stored the individual metrics in a data structure or variables, you can easily average the precision, recall, and f1 for all entities to compute an average score for each metric. 

# In[43]:


## TODO compute average metrics
#print("Average Precision: ",metric_df["Precision"].mean())
#print("Average Recall: ",metric_df["Recall"].mean())
#print("Average f1: ",metric_df["f1"].mean())


# ##### Question: how do the average metrics here (computed on entity-level) compare to the token-level accuracy score above? Which metric(s) would you prefer to use to evaluate the quality of your model? Why? 

# We are almost Done with part II! We just need to save our BILUO training data for reuse in Part III. Complete the code below to do this. 

# In[44]:


def full_biluo_sentences(nlp, train, test):
    ## TODO persist BILUO data as text
    print("Make bilou dfs")
    training_data_as_bilou = [make_bilou_df(nlp,res) for res in train]
    test_data_as_bilou = [make_bilou_df(nlp,res) for res in test]
    print("Done!")
    training_file = ""
    test_file = ""
    for idx,df in enumerate(training_data_as_bilou):
        df = df[df["Tokens"]!=""]
        df = df[["Tokens","True"]]
        df.loc[df["Tokens"]==".","Tokens"]="\r\n"
        df.loc[df["Tokens"] == ".", "True"] = ""
        ascsv = df.to_csv(None,sep=" ",encoding="utf-8",index=False,header=False,line_terminator="\n")
        training_file = training_file+"\r\n"+ascsv
    for idx, df in enumerate(test_data_as_bilou):
        df = df[df["Tokens"] != ""]
        df = df[["Tokens", "True"]]
        df.loc[df["Tokens"] == ".", "Tokens"] = "\r\n"
        df.loc[df["Tokens"] == ".", "True"] = ""
        ascsv = df.to_csv(None, sep=" ", encoding="utf-8", index=False, header=False, line_terminator="\n")
        test_file = test_file + "\r\n" + ascsv
    return training_file,test_file


def full_biluo_resumes(nlp, train, test):
    ## TODO persist BILUO data as text
    print("Make bilou dfs")
    training_data_as_bilou = [make_bilou_df(nlp,res) for res in train]
    test_data_as_bilou = [make_bilou_df(nlp,res) for res in test]
    print("Done!")
    training_file = ""
    for idx,df in enumerate(training_data_as_bilou):
        df = df[df["Tokens"]!=""]
        df = df[["Tokens","True"]]
        ascsv = df.to_csv(None,sep=" ",encoding="utf-8",index=False,header=False,line_terminator="\n")
        training_file = training_file+"\r\n"+ascsv
    for idx, df in enumerate(test_data_as_bilou):
        df = df[df["Tokens"] != ""]
        df = df[["Tokens", "True"]]
        ascsv = df.to_csv(None, sep=" ", encoding="utf-8", index=False, header=False, line_terminator="\n")
        training_file = training_file + "\r\n" + ascsv
    return training_file


def load_corpus():
    from flair.data_fetcher import NLPTaskDataFetcher
    columns = {1:"ner",3:"text"}
    corpus = NLPTaskDataFetcher.load_column_corpus(
        "../dataset/flair",
        column_format=columns,
        train_file="train_res_bilou.txt",
        test_file = "test_res_bilou.txt")
    return corpus


def run():
    resumes = load_resumes()
    resumes = gather_candidates(resumes,chosen_entity_labels)
    resumes = filtered_data(resumes)
    resumes = remove_bad_data(resumes)
    train,test = train_test_split(resumes,0.7)
    #custom_nlp, _ = train_spacy_ner(train, n_iter=20)
    custom_nlp = spacy.load("C:\Projects\SAKI_NLP\models\custom_nlp")
    metric_df = calc_metrics(custom_nlp,test)
    training_file,test_file = full_biluo_resumes(custom_nlp, train, test)
    return training_file,test_file
    #persist(training_df,test_df)
    #corpus = load_corpus()
    #return corpus

def persist(traindf,testdf):
    with open("../dataset/flair/train_res_bilou.txt",'w+',encoding="utf-8") as f:
        traindf.to_csv(f,sep=" ",encoding="utf-8",index=False)
    with open("../dataset/flair/test_res_bilou.txt",'w+',encoding="utf-8") as f:
        testdf.to_csv(f,sep=" ",encoding="utf-8",index=False)



def test():
    #from flair.data import TaggedCorpus
    from flair.data_fetcher import NLPTaskDataFetcher
    columns = {1: "ner", 3: "text"}
    corpus = NLPTaskDataFetcher.load_column_corpus("../dataset/flair", column_format=columns,
                                                   train_file="train_res_bilou.txt",
                                                   test_file="test_res_bilou.txt")

def load_model_and_test():
    # load the model you trained
    model = SequenceTagger.load('resources/taggers/example-ner/final-model.pt')

    # create example sentence
    sentence = Sentence('I love Berlin')

    # predict tags and print
    model.predict(sentence)

    print(sentence.to_tagged_string())