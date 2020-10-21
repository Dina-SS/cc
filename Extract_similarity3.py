
import numpy as np
import pandas as pd
import json
import re

import nltk
from nltk.corpus import stopwords
stop_words= stopwords.words('english')
from nltk.stem.wordnet import WordNetLemmatizer
lemma= WordNetLemmatizer()
from nltk import pos_tag
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk import ngrams

import string

import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)

import argparse
import os
import io
import docx2txt
import textract
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EDUCATION =['4 year degree', 'hsc', 'ssc',  'bachelors', 'icse', 'postdoctoral', 'bs', 'msc', 'masters', 
  'mtech', 'graduate', 'phd', 'btech', 'ms', 'bachelor', 'undergraduate', 'postdoc',
  'advanced degree', 'mba', 'master', 'ba', 'ms degree', 'ma', '4year degree', 'doctorate', 'cbse','bsc']

EDU=['be','me']
punc={'`', '~', '$', '|', '.', '@', '%', '#', ']', '=', ':', ';', '}', '[', '/', ')', '^', '>', '{', '+', '<', "'", '!', '"', '_', '(', '\\', '&', '*', '-', '?'}

#open json file
#with open('skills.json') as f:
    #skills = json.load(f)
    
skills_df=pd.read_csv('dir/skills.csv')
skills=list(skills_df.skill)




#Function converting pdf to string
def extract_text_from_pdf(pdf_path):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    with open(pdf_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh,caching=True,check_extractable=True):
            page_interpreter.process_page(page)

        text = fake_file_handle.getvalue()

    # close open handles
    converter.close()
    fake_file_handle.close()

    return text
#------------------------------------------------------------------------------------
def extract_text_from_docx(doc_path):
    try:
        temp = docx2txt.process(doc_path)
        return temp
    except KeyError:
        return ' '

#------------------------------------------------------------------------------------
def extract_text_from_doc(doc_path):
    try:
        try:
            import textract
        except ImportError:
            return ' '
        temp = textract.process(doc_path).decode('utf-8')
        return temp
    except KeyError:
        return ' '

#------------------------------------------------------------------------------------
def extract_text(file_path):

    text = ''
    if file_path.endswith('pdf') == True:
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith('docx') == True:
        text = extract_text_from_docx(file_path)
    elif file_path.endswith('doc') == True :
        text = extract_text_from_doc(file_path)
    else:
        text= file_path
    return text
#------------------------------------------------------------------------------------
def extract_name(nlp_text, matcher):

    pattern = [[{'POS': 'PROPN'}, {'POS': 'PROPN'}]]
    
    matcher.add('NAME', None, *pattern)
    
    matches = matcher(nlp_text)
    
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        return span.text
#------------------------------------------------------------------------------------
def extract_email(text):
    email = re.findall("([^@|\s]+@[^@]+\.[^@|\s]+)", text)
    if email:
        try:
            return email[0].split()[0].strip(';')
        except IndexError:
            return None

#------------------------------------------------------------------------------------
def extract_mobile_number(text):

    # Found this complicated regex on : https://zapier.com/blog/extract-links-email-phone-regex/
    phone = re.findall(re.compile(r'(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|[0-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?'), text)
    if phone:
        number = ''.join(phone[0])
        if len(number) > 11:
            return '+' + number
        else:
            return number
        
#------------------------------------------------------------------------------------
def extract_education(nlp_text):

    edu = []
    # Extract education degree
    for tx in nlp_text.split('\n'):
        for t in tx.split():
            tex = re.sub(r'[^\w\s]', '', t)
            if (t.upper or t.find(".") > 0) and tex in EDU:
                edu.append(tx.split("   ")[0])

            elif tex.lower() in EDUCATION:
                edu.append(tx.split("   ")[0])          
    return edu
#------------------------------------------------------------------------------------
def extract_experience(resume_text):
    
    # word tokenization 
    word_tokens = nltk.word_tokenize(resume_text)
    # remove stop words and lemmatize  
    filtered_sentence = [w for w in word_tokens if not w in stop_words and w not in stop_words] 
    sent = nltk.pos_tag(filtered_sentence)
    
    # parse regex
    cp = nltk.RegexpParser('P: {<NNP>+}')
    cs = cp.parse(sent)
    
    test = []
    for vp in list(cs.subtrees(filter=lambda x: x.label()=='P')):
        test.append(" ".join([i[0] for i in vp.leaves() if len(vp.leaves()) >= 2]))
        

    # Search the word 'experience' in the chunk and then print out the text after it
    x = [x[x.lower().index('experience') + 11:] for i, x in enumerate(test) if x and 'experience' in x.lower()]
    return x

#------------------------------------------------------------------------------------

def extract_skills(nlp_text):
    lemma_text= [" ".join(lemma.lemmatize(word.strip()) for word 
                          in word_tokenize(sent) if word.strip() not in stop_words) for sent in sent_tokenize(nlp_text)]
    skillset = []  

    for i in range (len(lemma_text)):

        #3_n_gram
        threegrams = ngrams(lemma_text[i].split(), 3)
        for grams in threegrams:
            gram=(' '.join(grams))
            if gram.lower() in skills:
                skillset.append(gram)
                lemma_text[i]=lemma_text[i].replace(gram, "")

        #2_n_gram
        twograms = ngrams(lemma_text[i].split(), 2)
        for grams in twograms:
            gram=(' '.join(grams))
            if gram.lower() in skills:
                skillset.append(gram)
                lemma_text[i]=lemma_text[i].replace(gram, "")

        #1_n_gram
        onegrams = ngrams(lemma_text[i].split(), 1)
        for grams in onegrams:
            gram=(' '.join(grams))
            if (gram.lower() in skills):
                skillset.append(gram)
                lemma_text[i]=lemma_text[i].replace(gram, "")
    
    return list(set(skillset))
#------------------------------------------------------------------------------------
def find_similarity(vect, list1,list2):
    """ Vectorise text and compute the cosine similarity """
    query_1 = vect.transform(list1)
    query_2 = vect.transform(list2)
    cos_sim = cosine_similarity(query_1.A, query_2.A)  # displays the resulting matrix
    return np.round(cos_sim.squeeze(), 3)
#------------------------------------------------------------------------------------

def get_info(file_path_or_cv_text,desc_text):
    
    nlp = spacy.load('en')
    doc = nlp(extract_text(file_path_or_cv_text))
    desc_doc = nlp(desc_text)
    
    name=extract_name(doc,matcher)
    phone=extract_mobile_number(doc.text)
    email=extract_email(doc.text)
    edu=extract_education(doc.text)
    exp=extract_experience(doc.text)
    
    cv_skills=extract_skills(doc.text)
    desc_skills=extract_skills(desc_doc.text)
    cv_skills_l=[x.lower() for x in cv_skills]
    desc_skills_l=[x.lower() for x in desc_skills]
    try:
        vectorizer = CountVectorizer().fit(desc_skills_l)

    except:
        vectorizer = CountVectorizer().fit(skills)

    
    similarity = find_similarity(vectorizer, [' '.join(desc_skills_l)],[' '.join(cv_skills_l)])

    #skill_gap 
    skill_gap = desc_skills_l
    for a in cv_skills_l: 
        if a in desc_skills_l: 
            skill_gap.remove(a)  

    dict_info={
               'Name':name,
               'Phone':phone,
               'Email':email,
               'Education':edu,
               'Experience':exp,
               'CV_Skills':cv_skills,
               'Desc_Skills':desc_skills,
               'Skill_Gap':skill_gap,
               'MatchingRate':similarity
        
    }
    json_output=json.dumps(dict_info)
    return (json_output)








