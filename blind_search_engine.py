
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.ldamulticore import LdaMulticore

import multiprocessing
import pickle

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


import pathlib

class blind_search_engine:
    def __init__(self):
        print("Loading data.....")
        self.lda_model,self.df,self.dictionary,self.topicid_to_ids = self.load()
        print('finished loading data')

    def preprocess(self,text):
        text = text.lower()
        words = word_tokenize(text)
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return words

    def load(self):
        
        chunks =[]
        df_address = str(pathlib.Path(__file__).parent.resolve()) + '\\saved_data\\sorted_df.csv'

        for chunk in pd.read_csv(df_address, chunksize=10000):
            chunks.append(chunk)

        loaded_df = pd.concat(chunks, ignore_index=True)

        dict_address = str(pathlib.Path(__file__).parent.resolve()) + '\\saved_data\\dictionary.dict'
        loaded_dictionary = Dictionary.load(dict_address)

        model_address = str(pathlib.Path(__file__).parent.resolve()) + '\\saved_data\\lda_model_80.model'
        #model_file = 'lda_model_80.model'
        loaded_lda_model = LdaModel.load(model_address)

        topicid_toids_address = str(pathlib.Path(__file__).parent.resolve()) + '\\saved_data\\topicid_to_ids.pkl'
        with open(topicid_toids_address, 'rb') as f:
            loaded_topicid_to_ids = pickle.load(f)

        return loaded_lda_model,loaded_df,loaded_dictionary,loaded_topicid_to_ids

            
    def bow_to_topicid(self,bow,no_lower_than = 0.2,num=3):
        p_topics = self.lda_model.get_document_topics(bow, minimum_probability = no_lower_than)
        top = sorted(p_topics, key=lambda x: x[1], reverse=True)[:num]
        return top

    def search_to_topicid(self,search,no_lower_than = 0.2,num=3):
        
        preprocess_search = self.preprocess(search)
        bow_search = self.dictionary.doc2bow(preprocess_search)
        top = self.bow_to_topicid(bow_search,no_lower_than,num)
        return top


    def id_to_contents(self,id):
        result = self.df.loc[self.df['id'] == id]
        return result

    def search_results(self,top,num_result = 10, topics = 3):
        results = self.df.iloc[:2]
        results = results.drop([0,1])


        
        for topicid, prob in top[:topics]:
            ids = self.topicid_to_ids[topicid]
        
            for id in ids:
                next = self.id_to_contents(id)
            
                if num_result == 0 :
                    break
                elif next is None:
                    break
                else:
                    num_result = num_result -1
                results = pd.concat([results, next],axis = 0)
    

        return results
    


    def search(self,text:str,no_lower_than=0.2,num=3,num_result=10,topics=3):
        top = self.search_to_topicid(text,no_lower_than,num)
        results = self.search_results(top,num_result,topics)
        return results
    

    def turnOn(self,no_lower_than=0.2,num_result=10,topics=3):
        x = input('Search >>: ')
        while x != 'off' and x != 'end' and x!= 'close':
            
            results = self.search(x,no_lower_than,num=3,num_result=10,topics=3)
            print(results)
            x = input('Search >>: ')
        print('Search Ended')
    

