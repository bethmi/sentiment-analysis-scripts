from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import nltk
import csv
from nltk import FreqDist,ConditionalFreqDist 
from nltk.probability import FreqDist, DictionaryProbDist, ELEProbDist, sum_logs, LaplaceProbDist, SimpleGoodTuringProbDist
from nltk.classify import NaiveBayesClassifier,MaxentClassifier
import nltk.classify.util
import re
import operator
from operator import itemgetter, attrgetter
from nltk.corpus import stopwords
import collections
import numpy as np
import itertools
from pylab import figure,axes,pie,show
from tkinter import PhotoImage
from nltk import pos_tag
from decimal import Decimal

## range of values to be considerd since 4 is considered as neutral
maximum_negative_value=4.0
minimum_positive_value=4.0


def clear_gui():
    training_set.set('stanfordMOOCForumPostsSet.csv') 
    num_fold.set('')

    accuracy.set('')
    pos_precision.set('')
    pos_recall.set('')
    neg_precision.set('')
    neg_recall.set('')
    classifier_selection.set('')
    pos_var.set(0)

##   check box response for POS     
def show_pos_status():
    print('Pos feature is ', pos_var.get())
     
##   mode(bigram/unigram)
def show_mode_status():
    print('Bigram mode is ', mode_var.get())

##  get the sentiment based on the value
def is_not_neutral(value):
    if (Decimal(value)> minimum_positive_value) or (Decimal(value)< maximum_negative_value):
      return True;
    else:
      return False;
      
##  get the course name according to the dataset
def get_course_name(value):
    if (value == 'How to learn maths'):
        return 'Education';
    elif (value == 'Humanities Science'):
        return 'Humanities';
    elif (value == 'Statistics in Medicine'):
        return 'Medicine';

##  get the sentiment based on the value
def get_sentiment(value):
    if (Decimal(value)> minimum_positive_value):
      return "positive"
    elif (Decimal(value)< maximum_negative_value):
      return "negative"

## Cross Validation
def cross_validation(*args):
    try:
        ts=training_set.get()
        subject=course.get()
        bigram_flag=mode_var.get() ## if 1, then do bigram mode
        pos_flag=pos_var.get() ## if 1, then do pos
        classifier_type=classifier_selection.get()  
        stopset = set(stopwords.words('english'))

        training=[]
        tot_accuracy=0
        tot_pos_precision=0
        tot_pos_recall=0
        tot_neg_precision=0
        tot_neg_recall=0
        tot_pos_fmeasure=0
        tot_neg_fmeasure=0
        

        for t in csv.reader(open(ts, errors="ignore"), delimiter=','):
            ## comment should belong to the selected course and neutral sentiments are ignored
            if(get_course_name(subject)==t[7] and is_not_neutral(t[4])): 
                training.append(t)

        num_folds = (int)(num_fold.get())
        subset_size =round((len(training))/num_folds)
        print('data-setsize is ', len(training))
        print('subset_size is ', subset_size)

        for i in range(num_folds):
            txt=0
            negative_comments = []
            positive_comments = []
            negative_comments_list = []
            positive_comments_list = []
            comments = []
            top_pos=[]
            top_neg=[]
            neutral=[]
            temp_store=[]
            test_ref_store=[]
            test_store=[]
            temp_ts_store=[]
            test_sentiment=[]
            posBigramFeatureVector = []
            negBigramFeatureVector = []
            testing_this_round = training[i*subset_size:][:subset_size]
            training_this_round = training[:i*subset_size] + training[(i+1)*subset_size:]

            ## for each iteration
            print('Now i is ',i)
            print('training_this_round: ', len(training_this_round))
            print('testing_this_round: ', len(testing_this_round))

            print('...Training begins...')

            for t in training_this_round:
                sign=get_sentiment(t[4])
                content = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', t[0])
                content = re.sub('\s+', ' ', content)  # condense all whitespace    
                content = re.sub('[^A-Za-z0-9 ]+', '', content)  # remove non-alpha chars excluding numbers
                content = content.lower()
                ## print(content,sign)

                try:
                    ## check if the content is already available in the temp_ts_store to avoid duplicates
                    index =  temp_ts_store.index(content)
                except ValueError: ## Exception occurs when content is not in list
                    index=-1
                       
                if index==-1:
                    temp_ts_store.append(content)
                    txt=txt+1
                    if sign=='negative':
                        negative_comments_list.append(content)
                    if sign=='positive':
                        positive_comments_list.append(content)                                
                    
            np=len(positive_comments_list)
            nn=len(negative_comments_list)
            
            print(nn,' negative text is read')

            print(np,' positive text is read')

            if nn>np :
                positive_comments=positive_comments_list
                for text in negative_comments_list:
                    index =  negative_comments_list.index(text)
                    if index < np:
                        negative_comments.append(text)


            if np>nn :
                negative_comments=negative_comments_list 
                for text in positive_comments_list:
                    index =  positive_comments_list.index(text)
                    if index < nn:
                        positive_comments.append(text)

            if np==nn :
                positive_comments=positive_comments_list
                negative_comments=negative_comments_list 
                
            np=len(positive_comments)
            nn=len(negative_comments)
            
            print(nn,' negative text is considered')
            print(np,' positive text is considered')
            
            
            ##   bigram without pos

            if(pos_flag==0 and bigram_flag==1):            
                for (words) in negative_comments:
                    for item in nltk.bigrams ([e.lower() for e in words.split() if e not in stopset]):
                        negBigramFeatureVector.append('_'.join(item))      
                
                print('negative words are filtered')

                for (words) in positive_comments:
                    for item in nltk.bigrams ([e.lower() for e in words.split() if e not in stopset]):
                        posBigramFeatureVector.append('_'.join(item))

                print('positive words are filtered')
                
                for (words) in negBigramFeatureVector:
                    words_filtered = words.split()
                    comments.append((words_filtered, 'negative'))

                for (words) in posBigramFeatureVector:
                    words_filtered = words.split()       
                    comments.append((words_filtered, 'positive'))

            ##   unigram without pos

            if(pos_flag==0 and bigram_flag==0):
                for (words) in negative_comments:
                   words_filtered=[e.lower() for e in words.split() if e not in stopset]
                   comments.append((words_filtered, 'negative'))
                     
                print('negative words are filtered')

                for (words) in positive_comments:
                   words_filtered=[e.lower() for e in words.split() if e not in stopset]
                   comments.append((words_filtered, 'positive'))
                  
                print('positive words are filtered')   

            ##   pos +  bigram

            if(pos_flag==1 and bigram_flag==1):            
                for (words) in negative_comments:
                    cc=[]
                    cc.extend(words.lower().split())
                    ed_sentence = nltk.pos_tag(cc)                
                    for item in nltk.bigrams ([e.lower() for e,pos in ed_sentence if ((e not in stopset) & (pos in ['NN','JJ','JJR','JJS']))]):
                        negBigramFeatureVector.append('_'.join(item))      
                    
                print('negative words are filtered')

                for (words) in positive_comments:
                    cc=[]
                    cc.extend(words.lower().split())
                    ed_sentence = nltk.pos_tag(cc)                
                    for item in nltk.bigrams ([e.lower() for e,pos in ed_sentence if ((e not in stopset) & (pos in ['NN','JJ','JJR','JJS']))]):
                        posBigramFeatureVector.append('_'.join(item))

                print('positive words are filtered')
                    
                for (words) in negBigramFeatureVector:
                    words_filtered = words.split()
                    comments.append((words_filtered, 'negative'))

                for (words) in posBigramFeatureVector:
                    words_filtered = words.split()       
                    comments.append((words_filtered, 'positive'))

            ##  pos + unigram

            if(pos_flag==1 and bigram_flag==0):
                
                for (words) in negative_comments:
                    cc=[]
                    cc.extend(words.lower().split())
                    ed_sentence = nltk.pos_tag(cc)                

                    words_filtered=[e.lower() for e,pos in ed_sentence if ((e not in stopset) & (pos in ['NN','JJ','JJR','JJS']))]
                    comments.append((words_filtered, 'negative'))
                     
                print('negative words are filtered')

                for (words) in positive_comments:
                    cc=[]
                    cc.extend(words.lower().split())
                    ed_sentence = nltk.pos_tag(cc)                

                    words_filtered=[e.lower() for e,pos in ed_sentence if ((e not in stopset) & (pos in ['NN','JJ','JJR','JJS']))]
                    comments.append((words_filtered, 'positive'))
                  
                print('positive words are filtered')                      


            print('...Testing begins...')
            
            num=0
            
            for t in testing_this_round:
                num=num+1
                content = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', t[0])
                content = re.sub('\s+', ' ', content)  # condense all whitespace
                content = re.sub('[^A-Za-z ]+', '', content)  # remove non-alpha chars excluding numbers
                content = content.lower()
                assigned_sign=get_sentiment(t[4])
                test_sentiment.append((content,assigned_sign))

            print('comments are collected')

            def get_words_in_comments(comments):
                all_words = []
                for (words, sentiment) in comments:
                  all_words.extend(words)
                return all_words
            def get_word_features(wordlist):
                wordlist = nltk.FreqDist(wordlist)
                word_features = wordlist.keys()
                return word_features

            word_features = get_word_features(get_words_in_comments(comments))

            def extract_features(document):
                document_words = set(document)
                features = {}
                for word in word_features:
                    features['contains(%s)' % word] = (word in document_words)
                return features

            training_set_classification = nltk.classify.apply_features(extract_features, comments)  
                   

            if(classifier_type=='Naive Bayes'): 
                print('Naive Bayes Classifier is used')
                classifier = nltk.NaiveBayesClassifier.train(training_set_classification)
                print(classifier.show_most_informative_features(),'\n Start Checking...\n')

            if(classifier_type=='MaxEnt'):   
                print('Max Entropy Classifier is used')         
                classifier = nltk.classify.maxent.MaxentClassifier.train(training_set_classification, 'GIS', trace=3, encoding=None, labels=None, gaussian_prior_sigma=0, max_iter = 10)
                print(classifier.show_most_informative_features(),'\n Start Checking...\n')            

            refsets = collections.defaultdict(set)
            testsets = collections.defaultdict(set)

            max_neg_prob = 0.0
            max_pos_prob = 0.0
            most_neg_comment=''
            most_pos_comment=''
            num=0

            for (sentence,sign) in test_sentiment:
                sen=[]
                cc=[]
                num=num+1
                sentence = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', sentence)
                ed_sentence = re.sub('[^A-Za-z ]+', '', sentence)  # remove non-alpha chars excluding numbers
                ed_sentence = re.sub('\s+', ' ', ed_sentence)  # condense all whitespace
                
                if(pos_flag==1):
                    cc.extend(ed_sentence.lower().split())
                    ed_sentence = nltk.pos_tag(cc)                

                ## bigram mode
                if(bigram_flag==1):
                    if(pos_flag==1):
                        for item in nltk.bigrams ([e.lower() for e,pos in ed_sentence if ((e not in stopset) & (pos in ['NN','JJ','JJR','JJS']))]):
                            sen.append('_'.join(item))
                    else:
                        for item in nltk.bigrams ([e.lower() for e in ed_sentence.split() if e not in stopset]):
                            sen.append('_'.join(item))

                    try:
                        index =  temp_store.index(ed_sentence)
                    except ValueError: index=-1
                    if index==-1:
                        temp_store.append(ed_sentence)
                        test_store.append((sen,sign))
                        test_ref_store.append((num,sign))
                        refsets[sign].add(num)
                        dist=classifier.prob_classify(extract_features(sen))
                    for label in dist.samples():
                        if label == 'positive':
                            if dist.prob(label) > max_pos_prob :
                                max_pos_prob = dist.prob(label)
                                most_pos_comment= sentence
                            if dist.prob(label)>0.5:
                                top_pos.append((sentence,dist.prob(label)))                                         
                                testsets['positive'].add(num)
                        if label == 'negative':
                            if dist.prob(label) > max_neg_prob :
                                max_neg_prob = dist.prob(label)
                                most_neg_comment= sentence
                            if dist.prob(label)> 0.5:
                                top_neg.append((sentence,dist.prob(label)))
                                testsets['negative'].add(num)
                                         

                     # else :
                     #     print('Duplicate Value')
                          

              ## unigram mode
                if(bigram_flag==0):
                    if(pos_flag==1):
                        sen=[e.lower() for e,pos in ed_sentence if ((e not in stopset) & (pos in ['NN','JJ','JJR','JJS']))]
                    else:
                        sen=[e.lower() for e in ed_sentence.split() if e not in stopset]

                   
                    try:
                        index =  temp_store.index(ed_sentence)
                    except ValueError: index=-1
                    if index==-1:
                        temp_store.append(ed_sentence)
                        test_store.append((sen,sign))
                        test_ref_store.append((num,sign))
                        refsets[sign].add(num)
                        
                    dist=classifier.prob_classify(extract_features(sen))
                    for label in dist.samples():
                        if label == 'positive':
                            if dist.prob(label) > max_pos_prob :
                                max_pos_prob = dist.prob(label)
                                most_pos_comment= sentence
                            if dist.prob(label)>0.5:
                                top_pos.append((sentence,dist.prob(label)))
                                testsets['positive'].add(num)
                        if label == 'negative':
                            if dist.prob(label) > max_neg_prob :
                                max_neg_prob = dist.prob(label)
                                most_neg_comment= sentence
                            if dist.prob(label)> 0.5:
                                top_neg.append((sentence,dist.prob(label)))
                                testsets['negative'].add(num)
                                    

                   # else :
                   #  print('Duplicate Value')
                    

            test_features = get_word_features(get_words_in_comments(test_store))
            testing_set = nltk.classify.apply_features(extract_features, test_store)

            acc=(nltk.classify.util.accuracy(classifier, testing_set))*100
            pp =(nltk.precision(refsets['positive'], testsets['positive']))
            pr = nltk.recall(refsets['positive'], testsets['positive'])
            np = nltk.precision(refsets['negative'], testsets['negative'])
            nr = nltk.recall(refsets['negative'], testsets['negative'])
            fmp = nltk.f_measure(refsets['positive'], testsets['positive'], alpha=0.5)
            fmn = nltk.f_measure(refsets['negative'], testsets['negative'], alpha=0.5)
            
            print ('\naccuracy:',acc ,'%\n')
            print ('pos precision:',pp )
            print ('pos recall:',pr)
            print ('neg precision:',np )
            print ('neg recall:',nr)
            print ('f measure (pos):',fmp)
            print ('f measure (neg):',fmn)
            print('\n')

            if pp==None:
                pp=0.0
            if pr==None:
                pr=0.0
            if np==None:
                np=0.0
            if nr==None:
                nr=0.0
            if fmp==None:
                fmp=0.0
            if fmn==None:
                fmn=0.0

            tot_accuracy        =tot_accuracy     +(float)(acc)
            tot_pos_precision   =tot_pos_precision+(float)(pp)
            tot_pos_recall      =tot_pos_recall   +(float)(pr)
            tot_pos_fmeasure    =tot_pos_fmeasure    +(float)(fmp)
            tot_neg_precision   =tot_neg_precision+(float)(np)
            tot_neg_recall      =tot_neg_recall   +(float)(nr)
            tot_neg_fmeasure    =tot_neg_fmeasure    +(float)(fmn)

    
        print('Done')

        final_accuracy      =tot_accuracy/num_folds
        final_pos_precision =tot_pos_precision/num_folds
        final_pos_recall    =tot_pos_recall/num_folds
        final_pos_fmeasure  =tot_pos_fmeasure/num_folds
        final_neg_precision =tot_neg_precision/num_folds
        final_neg_recall    =tot_neg_recall/num_folds
        final_neg_fmeasure  =tot_neg_fmeasure/num_folds

        print('Accuracy\t',final_accuracy)
        print('Pos pre\t',final_pos_precision)
        print('Pos rec\t',final_pos_recall)
        
        print('Neg pre\t',final_neg_precision)
        print('Neg recall\t',final_neg_recall)

        print('Pos fmeasure\t',final_pos_fmeasure)
        print('Neg fmeasure\t',final_neg_fmeasure)

        accuracy.set(final_accuracy)
        pos_precision.set(final_pos_precision)
        pos_recall.set(final_pos_recall)
        neg_precision.set(final_neg_precision)
        neg_recall.set(final_neg_recall)


    except Exception as e:
        print(e)

    
root = Tk()
root.title('Testing Environment')

mainframe = ttk.Frame(root, padding='20 20 20 20') ## L T R B
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

training_set  =StringVar()
course        =StringVar()
num_fold      =StringVar()
analyze_text  =StringVar()
accuracy      =StringVar()
pos_precision =StringVar()
pos_recall    =StringVar()
neg_precision =StringVar()
neg_recall    =StringVar()
classifier_selection    =StringVar()


opt_row=1

##  radio button options
mode_var = IntVar()

mode_var.set(1) 
options = [
    ("Unigram",0),
    ("Bigram",1)    
]

##  check box
pos_var = IntVar()

training_set.set('stanfordMOOCForumPostsSet.csv')
training_set_entry = ttk.Entry(mainframe, width=50, textvariable=training_set)
ttk.Label(mainframe, text='Training set\t\t: ').grid(column=1, row=2, sticky=(N, W))
training_set_entry.grid(column=2, row=2, sticky=(W, E))

course_entry = ttk.Combobox(mainframe, width=50, textvariable=course)
ttk.Label(mainframe, text='Course name\t\t: ').grid(column=1, row=3, sticky=(N, W))
course_entry['values']=(
    'How to learn maths',
    'Humanities Science',
    'Statistics in Medicine'
    )
course_entry.current(0)
course_entry.grid(column=2, row=3, sticky=(W, E))

num_folds_entry = ttk.Entry(mainframe, width=50, textvariable=num_fold)
ttk.Label(mainframe, text='No. of Folds\t\t: ').grid(column=1, row=4, sticky=(N, W))
num_folds_entry.grid(column=2, row=4, sticky=(W, E))

classifier_entry = ttk.Combobox(mainframe, width=50, textvariable=classifier_selection)
ttk.Label(mainframe, text='Classifier\t\t\t: ').grid(column=1, row=5, sticky=(N, W))
classifier_entry['values']=(
    'Naive Bayes',
    'MaxEnt'
    )
classifier_entry.current(0)
classifier_entry.grid(column=2, row=5, sticky=(W, E))

##  Radio Button
##  =============
##Select the mode (unigram/bigram) <- current = bigram

for txt, val in options:
    opt_row=opt_row+1
    Radiobutton(mainframe, text=txt, variable=mode_var, command=show_mode_status, value=val).grid(column=3, row=opt_row, sticky=(N, W))

ttk.Checkbutton(mainframe, text="POS", command=show_pos_status, variable=pos_var, onvalue=1, offvalue=0).grid(column=3, row=5, sticky=(N, W))

ttk.Button(mainframe, text="Start", command=cross_validation).grid(column=3, row=7, sticky=W)

ttk.Button(mainframe, text=" Clear  ", command=clear_gui).grid(column=3, row=9, sticky=W)

ttk.Label(mainframe, text='Accuracy    %\t\t: ').grid(column=1, row=6, sticky=(N, W))
ttk.Label(mainframe, textvariable=accuracy ).grid(column=2, row=6, sticky=(N,W))

ttk.Label(mainframe, text='Positive Precision \t\t: ').grid(column=1, row=7, sticky=(N, W))
ttk.Label(mainframe, textvariable=pos_precision ).grid(column=2, row=7, sticky=(N,W))

ttk.Label(mainframe, text='Positive Recall \t\t: ').grid(column=1, row=8, sticky=(N, W))
ttk.Label(mainframe, textvariable=pos_recall ).grid(column=2, row=8, sticky=(N,W))


ttk.Label(mainframe, text='Negative Precision \t: ').grid(column=1, row=9, sticky=(N, W))
ttk.Label(mainframe, textvariable=neg_precision ).grid(column=2, row=9, sticky=(N,W))

ttk.Label(mainframe, text='Negative Recall \t\t: ').grid(column=1, row=10, sticky=(N, W))
ttk.Label(mainframe, textvariable=neg_recall ).grid(column=2, row=10, sticky=(N,W))


for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)

training_set_entry.focus()


root.bind('<Return>', cross_validation)

root.mainloop()
