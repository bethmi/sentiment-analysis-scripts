from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import nltk
import csv
from nltk import FreqDist,ConditionalFreqDist 
from nltk.probability import FreqDist, DictionaryProbDist, ELEProbDist, sum_logs, LaplaceProbDist
from nltk.classify import NaiveBayesClassifier
import re
import operator
from operator import itemgetter, attrgetter
from nltk.corpus import stopwords
import collections
import numpy as np
import itertools
from pylab import figure,axes,pie,show
from tkinter import PhotoImage
from decimal import Decimal


def clear_gui():
    clear_highlights()
    training_set.set('stanfordMOOCForumPostsSet.csv')
    testing_data.set('')
    pos_var.set(1)

##   check box response for POS     
def show_pos_status():
    print('Pos feature is ', pos_var.get())
     
##   mode(bigram/unigram)
def show_mode_status():
    print('Bigram mode is ', mode_var.get())

##  get the course name according to the dataset
def get_course_name(value):
    if (value == 'How to learn maths'):
        return 'Education';
    elif (value == 'Humanities Science'):
        return 'Humanities';
    elif (value == 'Statistics in Medicine'):
        return 'Medicine';

def calculate(single_text_only_flag):
    try:
        negative_comments = []
        positive_comments = []
        negative_comments_list = []
        positive_comments_list = []
        comments = []
        temp_ts_store=[]
        posBigramFeatureVector = []
        negBigramFeatureVector = []

        matched=0
        
        ts=training_set.get()
        subject=course.get()
        bigram_flag=mode_var.get() ## if 1, then do bigram mode
        pos_flag=pos_var.get() ## if 1, then do pos

        stopset = set(stopwords.words('english'))
        clear_highlights()

        txt=0

        for t in csv.reader(open(ts, errors="ignore"), delimiter=','):
          if(get_course_name(subject)==t[7] and t[4]!=4): ## comment should belong to the selected subject and neutral sentiments are ignored
            sign=get_sentiment(t[4])

            content = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', t[0])
            content = re.sub('\s+', ' ', content)  # condense all whitespace    
            content = re.sub('[^A-Za-z0-9 ]+', '', content)  # remove non-alpha chars excluding numbers
            content = content.lower()

            ##  mark as matched
            matched = matched+1
            
            try:
                ## check if the content is already available in the temp_ts_store to avoid duplicates
                index =  temp_ts_store.index(content) 
            except ValueError: ## Exception occurs when content is not in list
                index=-1
                
            if index==-1:
                temp_ts_store.append(content)
                print(txt,'\t',sign,'\t',content)
                txt=txt+1
                if sign=='negative':
                    negative_comments_list.append(content)
                if sign=='positive':
                    positive_comments_list.append(content)                                


        print("Total number of collected comments for {0} = {1}".format(subject, matched))    
        
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

        ##   bigram + pos

        if(pos_flag==1 and bigram_flag==1):            
            for (words) in negative_comments:
                temp_store=[]
                temp_store.extend(words.lower().split())
                ed_sentence = nltk.pos_tag(temp_store)                
                for item in nltk.bigrams ([e.lower() for e,pos in ed_sentence if ((e not in stopset) & (pos in ['NN','JJ','JJR','JJS']))]):
                    negBigramFeatureVector.append('_'.join(item))      
                
            print('negative words are filtered')

            for (words) in positive_comments:
                temp_store=[]
                temp_store.extend(words.lower().split())
                ed_sentence = nltk.pos_tag(temp_store)                
                for item in nltk.bigrams ([e.lower() for e,pos in ed_sentence if ((e not in stopset) & (pos in ['NN','JJ','JJR','JJS']))]):
                    posBigramFeatureVector.append('_'.join(item))
            print('positive words are filtered')

            for (words) in negBigramFeatureVector:
                words_filtered = words.split()
                comments.append((words_filtered, 'negative'))

            for (words) in posBigramFeatureVector:
                words_filtered = words.split()       
                comments.append((words_filtered, 'positive'))

        ##   unigram + pos

        if(pos_flag==1 and bigram_flag==0):
            for (words) in negative_comments:
                temp_store=[]
                temp_store.extend(words.lower().split())
                ed_sentence = nltk.pos_tag(temp_store)                

                words_filtered=[e.lower() for e,pos in ed_sentence if ((e not in stopset) & (pos in ['NN','JJ','JJR','JJS']))]
                comments.append((words_filtered, 'negative'))
            print('negative words are filtered')

            for (words) in positive_comments:
                temp_store=[]
                temp_store.extend(words.lower().split())
                ed_sentence = nltk.pos_tag(temp_store)                

                words_filtered=[e.lower() for e,pos in ed_sentence if ((e not in stopset) & (pos in ['NN','JJ','JJR','JJS']))]
                comments.append((words_filtered, 'positive'))

            print('positive words are filtered')
        
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

## save results to file

        filename="result-{0}-{1}-{2}.csv".format(ts,bigram_flag,pos_flag)    
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(comments)    

        word_features = get_word_features(get_words_in_comments(comments))
        word_features = get_word_features(get_words_in_comments(comments))

        def extract_features(document):
            document_words = set(document)
            features = {}
            for word in word_features:
                features['contains(%s)' % word] = (word in document_words)
            return features


        training_set_classification = nltk.classify.apply_features(extract_features, comments)        

        classifier = nltk.NaiveBayesClassifier.train(training_set_classification)
        
        print(classifier.show_most_informative_features(),'\n Start Checking...\n')

        if(single_text_only_flag):
            print("prceeding with text...")
            check_text(bigram_flag, pos_flag, classifier, stopset, extract_features, word_features)
        else:
            print("prceeding with dataset...")
            check_dataset(bigram_flag, pos_flag, classifier, stopset, extract_features, word_features)                    
    except ValueError:
        pass


def check_dataset(bigram_flag, pos_flag, classifier, stopset, extract_features, word_features):

    ds=testing_data.get()
    test_comments = []
    temp_store=[]
    top_pos=[]
    top_neg=[]
    neutral=[]
    max_neg_prob = 0.0
    max_pos_prob = 0.0
    most_neg_comment=''
    most_pos_comment=''

    for t in csv.reader(open(ds, errors="ignore"), delimiter=','):     
        test_comments.append(t[0])

    print("comments collected from the dataset")

    num=1
    for (sentence) in test_comments:
        word_list=[]
        temp_store=[]
        print('\n',num)
        num=num+1
        sentence = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', sentence)
        ed_sentence = re.sub('[^A-Za-z ]+', '', sentence)  # remove non-alpha chars excluding numbers
        ed_sentence = re.sub('\s+', ' ', ed_sentence)  # condense all whitespace
        
        if(pos_flag==1):
            temp_store.extend(ed_sentence.lower().split())
            print(num)
            print('\n splitting\n')
            ed_sentence = nltk.pos_tag(temp_store)                
            print(ed_sentence)

        ## bigram
        if(bigram_flag==1):
            if(pos_flag==1):
                for item in nltk.bigrams ([e.lower() for e,pos in ed_sentence if ((e not in stopset) & (pos in ['NN','JJ','JJR','JJS']))]):
                    word_list.append('_'.join(item))
            else:
                for item in nltk.bigrams ([e.lower() for e in ed_sentence.split() if e not in stopset]):
                    word_list.append('_'.join(item))

            try:
                index =  temp_store.index(ed_sentence)
            except ValueError: index=-1
          
            if index==-1:
                temp_store.append(ed_sentence)
                dist=classifier.prob_classify(extract_features(word_list))
                for label in dist.samples():
                    if label == 'positive':
                        if dist.prob(label) > max_pos_prob :
                            max_pos_prob = dist.prob(label)
                            most_pos_comment= sentence
                        if dist.prob(label)>0.5:
                            top_pos.append((sentence,dist.prob(label)))

                    if label == 'negative':
                        if dist.prob(label) > max_neg_prob :
                            max_neg_prob = dist.prob(label)
                            most_neg_comment= sentence
                        if dist.prob(label)> 0.5:
                            top_neg.append((sentence,dist.prob(label)))                                    
                

        ## unigram
        if(bigram_flag==0):
            if(pos_flag==1):
                word_list=[e.lower() for e,pos in ed_sentence if ((e not in stopset) & (pos in ['NN','JJ','JJR','JJS']))]
            else:
                word_list=[e.lower() for e in ed_sentence.split() if e not in stopset]
       
            
            try:
                ## check if the content is already available in the temp_ts_store to avoid duplicates
                index =  temp_store.index(ed_sentence)
            except ValueError: ## Exception occurs when content is not in list
                index=-1
            
            if index==-1:
                temp_store.append(ed_sentence)

                dist=classifier.prob_classify(extract_features(word_list))
                for label in dist.samples():
                ##        print('\t%s:%f'%(label,dist.prob(label)))
                  if label == 'positive':
                      if dist.prob(label) > max_pos_prob :
                          max_pos_prob = dist.prob(label)
                          most_pos_comment= sentence
                      if dist.prob(label)>0.5:
                          top_pos.append((sentence,dist.prob(label)))

                  if label == 'negative':
                      if dist.prob(label) > max_neg_prob :
                          max_neg_prob = dist.prob(label)
                          most_neg_comment= sentence
                      if dist.prob(label)> 0.5:
                          top_neg.append((sentence,dist.prob(label)))
      
    print('\n\nMost Positive comment :-\n\n',most_pos_comment,'\n positivity score = ',max_pos_prob,'\n','-'*40)
    print('\nMost Negative comment :- \n\n',most_neg_comment,'\n negativity score = ',max_neg_prob,'\n','-'*40)

    top_neg =sorted(top_neg, key=itemgetter(1), reverse=True )
    top_pos=sorted(top_pos, key=itemgetter(1), reverse=True )

    print('Top Negative Stories\n\n')
    for sentence,score in top_neg[0:6]:
        print(sentence,'\t - ',score,'\n','-'*35)

    if(len(top_neg)>0):
        hn.set('Top Negative Story\t:')
        hn1.set('Top Negative Stories')
        mn1.set(top_neg[0][0])
    if(len(top_neg)>1):
        mn2.set( top_neg[1][0])
    if(len(top_neg)>2):
        mn3.set(top_neg[2][0])
    if(len(top_neg)>3):
        mn4.set(top_neg[3][0])
    if(len(top_neg)>4):
        mn5.set(top_neg[4][0])     

    print('\nTop Positive Stories\n\n')
    for sentence,score in top_pos[0:6]:
        print(sentence,'\t - ',score,'\n','-'*35)

    print('='*35)
    if(len(top_pos)>0):
        hp.set('Top Positive Story\t\t:')
        hp1.set('Top Positive Stories\n=====================')
        mp1.set(top_pos[0][0])
    if(len(top_pos)>1):
        mp2.set(top_pos[1][0])
    if(len(top_pos)>2):
        mp3.set(top_pos[2][0])
    if(len(top_pos)>3):
        mp4.set(top_pos[3][0])
    if(len(top_pos)>4):
        mp5.set(top_pos[4][0])

    fp=len(top_pos)
    fn=len(top_neg)

    print('Total num of positive\t',fp)
    print('Total num of negative\t',fn)
    
    generate_sentiment_chart(fp, fn)
    generate_word_distribution_chart(bigram_flag, pos_flag, word_list)  

def generate_sentiment_chart(fp, fn):
    print('Generating pie chart...')
    
    # make a square figure and axes
    fig = figure(1, figsize=(6,6))
    ax = axes([0.1, 0.1, 0.8, 0.8])

    # The slices will be ordered and plotted counter-clockwise.
    labels = 'Positive','Negative'
    fracs = [fp, fn]
    explode=(0, 0.05)
    colors=['green','red']
    pie(fracs, explode=explode, labels=labels,colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    
def generate_word_distribution_chart(bigram_flag, pos_flag, words):
    print('Generating distribution chart...')
    
    fig = figure(2, figsize=(6,6))
    if(bigram_flag==0):
        print('Most Used Words\n====================\n')
        counter_word=collections.Counter(words)
        for comment, count in counter_word.most_common(50):
            print("{0}: {1}".format(comment, count)) 
        fdist_words=FreqDist(words)
        fdist_words.plot(25)

    if(bigram_flag==1):
        print('Top Bigrams in the feed\n=====================\n')
        freq_big = FreqDist(words)
        print('Top Bigrams\n\n')
        print(freq_big)
        print('-'*35)
        counter=collections.Counter(words)
        for phrase, count in counter.most_common(50):
            print("{0}: {1}".format(phrase, count)) 
        print('\n')
        freq_big.plot(25)

## Check the text
def check_text(bigram_flag, pos_flag, classifier,stopset,extract_features,word_features):

    test_comments = testing_data.get()

    word_list=[]
    temp_store=[]

    sentence = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', test_comments)
    ed_sentence = re.sub('[^A-Za-z ]+', '', sentence)  # remove non-alpha chars excluding numbers
    ed_sentence = re.sub('\s+', ' ', ed_sentence)  # condense all whitespace
  
    if(pos_flag==1):
        temp_store.extend(ed_sentence.lower().split())
        print('\n splitting\n')
        ed_sentence = nltk.pos_tag(temp_store)
        print(ed_sentence)
    else:
        ed_sentence = ed_sentence.lower()
    
    ## bigram + pos
    if(pos_flag==1 and bigram_flag==1):
        for item in nltk.bigrams ([e.lower() for e,pos in ed_sentence if ((e not in stopset) & (pos in ['NN','JJ','JJR','JJS']))]):
            word_list.append('_'.join(item))

    ## unigram + pos
    if(pos_flag==1 and bigram_flag==0):
        word_list=[e.lower() for e,pos in ed_sentence if ((e not in stopset) & (pos in ['NN','JJ','JJR','JJS']))]

    ## bigram without pos
    if(pos_flag==0 and bigram_flag==1):
        for item in nltk.bigrams ([e.lower() for e in ed_sentence.split() if e not in stopset]):
            word_list.append('_'.join(item))

    ## unigram without pos        
    if(pos_flag==0 and bigram_flag==0):
        word_list=[e.lower() for e in ed_sentence.split() if e not in stopset]

    print("Analyzing Sentiment...")
    print('\n',sentence,'\t - ',classifier.classify(extract_features(word_list)),'\n',word_list,'\n')

    dist=classifier.prob_classify(extract_features(word_list))
    for label in dist.samples():
        print('\t%s:%f'%(label,dist.prob(label)))
        
    overall_sentiment=classifier.classify(extract_features(word_list))

    hp.set('Overall Sentiment\t\t:')
    mp1.set(overall_sentiment)
    
    feedback_heading.set('Feedback\t\t:')
    feedback_value.set(generate_feedback(overall_sentiment))


def get_words_in_comments(comments):
    all_words = []
    for (words, word_list) in comments:
        all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

def get_sentiment(value):
    if (Decimal(value)>4.0):
        return "positive"
    else:
        return "negative"

def generate_feedback(value):
    if (value=='negative'):
        return "Sorry for the inconvenience. Our instructors will contact you soon."
    else:
        return "We are glad to hear that you enjoyed this course!"   

def clear_highlights():
    hp.set('')     ##   headline - 'Top Positive Story'
    hp1.set('')    ##   headline - 'Top Positive Stories' 
    hn.set('')     ##   headline - 'Top Negative Story'
    hn1.set('')    ##   headline - 'Top Negative Stories'

    mp1.set('')    ##   most positive story 1
    mp2.set('')
    mp3.set('')
    mp4.set('')
    mp5.set('')

    mn1.set('')    ##   most negative story 1
    mn2.set('')
    mn3.set('')
    mn4.set('')
    mn5.set('')
    
    feedback_heading.set('')
    feedback_value.set('')

## UI
root = Tk()
root.title('Demonstration')

mainframe = ttk.Frame(root, padding='20 20 20 20') ## L T R B
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

training_set=StringVar()
testing_data=StringVar()
analyze_text=StringVar()
course      =StringVar()

hn=StringVar()
hn1=StringVar()
hp=StringVar()
hp1=StringVar()
feedback_heading=StringVar()
feedback_value=StringVar()

mn1=StringVar()
mn2=StringVar()
mn3=StringVar()
mn4=StringVar()
mn5=StringVar()
mp1=StringVar()
mp2=StringVar()
mp3=StringVar()
mp4=StringVar()
mp5=StringVar()
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
pos_var.set(1)

trainingset_entry = ttk.Entry(mainframe, width=50, textvariable=training_set)
ttk.Label(mainframe, text='Training Dataset\t\t: ').grid(column=1, row=2, sticky=(N, W))
trainingset_entry.grid(column=2, row=2, sticky=(W, E))
training_set.set('stanfordMOOCForumPostsSet.csv')

course_entry = ttk.Combobox(mainframe, width=50, textvariable=course)
ttk.Label(mainframe, text='Course name\t\t: ').grid(column=1, row=3, sticky=(N, W))
course_entry['values']=(
    'How to learn maths',
    'Humanities Science',
    'Statistics in Medicine'
    )
course_entry.current(0)
course_entry.grid(column=2, row=3, sticky=(W, E))

testing_data_entry = ttk.Entry(mainframe, width=50, textvariable=testing_data)
ttk.Label(mainframe, text='Testing Data\t\t: ').grid(column=1, row=4, sticky=(N, W))
testing_data_entry.grid(column=2, row=4, sticky=(W, E))

##  Radio Button
##  =============
##Select the mode (unigram/bigram) <- current = bigram

for txt, val in options:
    opt_row=opt_row+1
    Radiobutton(mainframe, text=txt, variable=mode_var, command=show_mode_status, value=val).grid(column=3, row=opt_row, sticky=(N, W))

ttk.Checkbutton(mainframe, text="POS", command=show_pos_status, variable=pos_var, onvalue=1, offvalue=0).grid(column=3, row=5, sticky=(N, W))

ttk.Button(mainframe, text="Analyze Dataset", command=lambda: calculate(False)).grid(column=2, row=6, sticky=(N,E))

ttk.Button(mainframe, text=" Analyze Text  ", command=lambda: calculate(True)).grid(column=2, row=6, sticky=(N,W))

ttk.Button(mainframe, text=" Clear  ", command=clear_gui).grid(column=3, row=16, sticky=W)


ttk.Label(mainframe, textvariable=hp).grid(column=1, row=10, sticky=(N, W))
ttk.Label(mainframe, textvariable=mp1, wraplength=1000).grid(column=2, row=10, sticky=(N,W))

ttk.Label(mainframe, textvariable=feedback_heading).grid(column=1, row=11, sticky=(N, W))
ttk.Label(mainframe, textvariable=feedback_value).grid(column=2, row=11, sticky=(N, W))


ttk.Label(mainframe, textvariable=hn).grid(column=1, row=11, sticky=(N, W))
ttk.Label(mainframe, textvariable=mn1, wraplength=1000).grid(column=2, row=11, sticky=(N,W))

for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)

trainingset_entry.focus()

root.bind('<Return>', calculate)

root.mainloop()
