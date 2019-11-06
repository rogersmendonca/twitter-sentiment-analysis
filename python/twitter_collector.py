import csv
from datetime import datetime,timedelta
import re
import string
import sys

from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from scipy import stats
import tweepy
from textblob import TextBlob

import twitter_util as util

''' 

  Twitter Collector class for Sentiment Analysis

  Developer: Rogers Reiche de Mendonca
  Date: 01/09/2019

'''
class TwitterCollector(object): 
    STOPWORDS = set(stopwords.words('english')).union({'amp','gt'})
    STOPWORDS_LOWER = {s.lower() for s in STOPWORDS}
    
    # HappyEmoticons
    EMOTICONS_HAPPY = set([
        ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
        ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
        '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
        'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
        '<3'
        ])
    
    # Sad Emoticons
    EMOTICONS_SAD = set([
        ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
        ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
        ':c', ':{', '>:\\', ';('
        ])

    # Emoji patterns
    EMOJI_PATTERN = re.compile("["
             u"\U0001F600-\U0001F64F"  # emoticons
             u"\U0001F300-\U0001F5FF"  # symbols & pictographs
             u"\U0001F680-\U0001F6FF"  # transport & map symbols
             u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
             u"\U00002702-\U000027B0"
             u"\U000024C2-\U0001F251"
             "]+", flags=re.UNICODE)

    # combine sad and happy emoticons
    EMOTICONS = EMOTICONS_HAPPY.union(EMOTICONS_SAD)
    
    SPLITTER = {'artificialintelligence':'Artificial Intelligence',
                'bigdata':'Big Data',
                'cybersecurity':'Cyber Security',
                'datascience':'Data Science',
                'datascientist':'Data Scientist',
                'datascientists':'Data Scientists',
                'deeplearning':'Deep Learning',
                'digitaltransformation':'Digital Transformation',
                'donaldtrump':'Donald Trump',
                'intelligentautomation':'Intelligent Automation',
                'intelligententerprise':'Intelligent Enterprise',
                'katemiddleton':'Kate Middleton',
                'machinelearning':'Machine Learning',
                'neuralnetworks':'Neural Networks',
                'princegeorge':'Prince George',
                'princesscharlotte':'Princess Charlotte',
                'princewilliam':'Prince William',
                'roboticprocessautomation':'Robotic Process Automation',
                'royalfamily':'Royal Family'}
    
    def __init__(self, consumer_key, consumer_secret, access_token, 
                 access_token_secret, wait_on_rate_limit=True, 
                 wait_on_rate_limit_notify=True, proxy=None): 

        # Authentication 
        try: 
            # create OAuthHandler object 
            self.auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 

            # set access token and secret 
            self.auth.set_access_token(access_token, access_token_secret) 

            # create tweepy API object to fetch tweets 
            self.api = tweepy.API(auth_handler=self.auth,
                                  wait_on_rate_limit=wait_on_rate_limit,
                                  wait_on_rate_limit_notify=wait_on_rate_limit_notify,
                                  proxy=proxy)            
            print(f"{self.api.me().screen_name} - {self.api.me().name}")
            print(f"{self.api.me().description}")
            print('')
        except: 
            print("Error: Authentication Failed")
            print('Confira as informacoes do "consumer key" e ' \
                  'do "access token" utilizados na autenticacao.')
            sys.exit(-1)

    def _splitter(self, tweet):        
        for key, value in TwitterCollector.SPLITTER.items():
            tweet = re.sub(key, value, tweet, flags = re.IGNORECASE)
        return tweet
            
    def _remove_link_special_character(self, tweet):        
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])" \
                               "|(\w+:\/\/\S+)", " ", tweet).split())
    def _remove_mention(self, tweet):        
        return re.sub(r'‚Ä¶', '', re.sub(r':', '', tweet))

    def _replace_consecutive_non_ASCII(self, tweet):        
        return re.sub(r'[^\x00-\x7F]+',' ', tweet)
    
    def _remove_emoji(self, tweet):        
        return TwitterCollector.EMOJI_PATTERN.sub(r'', tweet)
    
    def _remove_stopword(self, tweet):
        tweet_tk = [w for w in word_tokenize(tweet)]                        
        filtered_tweet = [w for w in tweet_tk 
                          if w.lower() not in TwitterCollector.STOPWORDS_LOWER \
                          and w not in TwitterCollector.EMOTICONS \
                          and w not in string.punctuation]
        return ' '.join(filtered_tweet)
    
    def _remove_digit(self, tweet):
        tweet = tweet.replace('\t',' ').replace('\r\n',' \r\n ')
        return ' '.join([x for x in tweet.split(' ') if not x.isdigit()])
    
    def _remove_short_word(self, tweet, min_len):
        tweet_tk = [w for w in word_tokenize(tweet)]
        filtered_tweet = [w for w in tweet_tk 
                          if len(w) >= min_len]
        return ' '.join(filtered_tweet)

    def clean_tweet(self, tweet):
        tweet = self._splitter(tweet)
        
        tweet = self._remove_link_special_character(tweet)

        tweet = self._remove_mention(tweet)
        
        tweet = self._replace_consecutive_non_ASCII(tweet)
        
        tweet = self._remove_emoji(tweet)

        tweet = self._remove_stopword(tweet)
        
        tweet = self._remove_digit(tweet)
        
        tweet = self._remove_short_word(tweet, 2)
        
        return tweet

    def get_tweets(self, query, since=None, until=None, count = 10): 
        if since is None:
            since = datetime.now().date() - timedelta(days=1)
        if until is None:
            until = datetime.now().date()
                    
        try: 
            tweets = tweepy.Cursor(self.api.search,
                                   q = f"\"{query}\" since:{since} until:{until} -filter:retweets",
                                   result_type = 'recent',
                                   include_entities = False,
                                   count = 100,
                                   tweet_mode = 'extended').items(count)
            
            return tweets 
        except tweepy.TweepError as e: 
            print("Error : " + str(e))
            sys.exit(-1)
            
    def save_tweets(self, dest_dir, file_name, query, since=None, until=None, count=10):
        file_csv = f"{dest_dir}/{file_name}"
        tweets = self.get_tweets(query, since, until, count)
        
        count_t = 0
        count_en = 0
        
        m_score_history = []
        df_ = None
        for tweet in tweets:
            count_t += 1
            m_score_history.append([None]*2)
    
            langs_dict = util.detect_langs(tweet.full_text)
            detected_lang = util.detect(tweet.full_text)
            has_en = (tweet.lang == 'en') or util.is_english(langs_dict)            
            
            full_text_clean = ''			
            if (has_en):			
                count_en += 1
                full_text_clean = self.clean_tweet(tweet.full_text)
                
                tb_full_text_clean = TextBlob(full_text_clean)
                m_score_history[count_t-1][0] = tb_full_text_clean.sentiment.polarity
                m_score_history[count_t-1][1] = tb_full_text_clean.sentiment.subjectivity
                
            print(f"{count_t}. {tweet.created_at} [en? {has_en}] {m_score_history[count_t-1]}")
                                        
            dict_ = {'full_text': tweet.full_text,
                     'full_text_clean': full_text_clean,
                     'retweet_count': tweet.retweet_count,
                     'favorite_count': tweet.favorite_count,
                     'lang': tweet.lang,
                     'detected_lang':detected_lang,
                     'detected_langs': langs_dict,
                     'has_english': has_en}
            dict_sentiments = {0:'polarity', 1:'subjectivity'}
            for k,v in dict_sentiments.items():
                dict_[v] = m_score_history[count_t-1][k] \
                            if m_score_history[count_t-1][k] is not None \
                            else 0
            
            if df_ is None:
                df_ = pd.DataFrame(columns=dict_)            
            df_ = df_.append(dict_,ignore_index=True)
        
        df_ = df_.sample(frac=1).reset_index(drop=True)
        df_.to_csv(file_csv,
                   sep=';',
                   header=True,
                   mode='w',
                   index=False,
                   quoting=csv.QUOTE_ALL)
        
        pol = [row[0] for row in m_score_history if row[0] is not None]
        stat_pol = [stats.describe(pol, ddof=0)]
        subj = [row[1] for row in m_score_history if row[1] is not None]
        stat_subj = [stats.describe(subj, ddof=0)]
        
        print(f"\n[{util.now()}] QUERY: \"{query}\" since:{since} until:{until}")
        print(f"[{util.now()}] TOTAL: {count_t} [{count_en} ENGLISH ({(count_en/count_t)*100:.2f} %)]")
        print(f"[{util.now()}] ESTATISTICA - POLARIDADE:")
        print(f"[{util.now()}] {stat_pol}; median={np.median(pol)}; std={np.std(pol,ddof=0)}")
        print(f"[{util.now()}] ESTATISTICA - SUBJETIVIDADE:")
        print(f"[{util.now()}] {stat_subj}; median={np.median(subj)}; std={np.std(subj,ddof=0)}")            
            
        with open(f"{dest_dir}/{util.date_to_str(datetime.now().date())}_EXECUTION_LOG.txt", 'a') as fd:                        
            print(f"\n[{util.now()}] QUERY: \"{query}\" since:{since} until:{until}", file=fd)
            print(f"[{util.now()}] TOTAL: {count_t} [{count_en} ENGLISH ({(count_en/count_t)*100:.2f} %)]", file=fd)
            print(f"[{util.now()}] ESTATISTICA - POLARIDADE:", file=fd)
            print(f"[{util.now()}] {stat_pol}; median={np.median(pol)}; std={np.std(pol,ddof=0)}", file=fd)
            print(f"[{util.now()}] ESTATISTICA - SUBJETIVIDADE:", file=fd)
            print(f"[{util.now()}] {stat_subj}; median={np.median(subj)}; std={np.std(subj,ddof=0)}", file=fd)
            print(' ', file=fd)

        del df_
        del m_score_history          
        del pol
        del stat_pol
        del subj
        del stat_subj
        
def help():
    print('''
          Uso: python twitter_collector.py <dest_dir> <max_limit> <prev_days_ini> <prev_days_end> <term_1> <prefix_1> ... <term_n> <prefix_n> 
          
          Parametros:
              dest_dir Diretorio de destino das informacoes coletadas
              max_limit Limite maximo de tweets para coletar por dia
              prev_day_ini Numero de dias anteriores para iniciar a coleta
              prev_day_end Numero de dias anteriores para parar a coleta (Obs: prev_day_end = 0, termina no dia anterior)              
              term_1 Termo de consulta 1
              prefix_1 Prefixo do arquivo relativo a consulta 1
              (...)
              term_n Termo de consulta n
              prefix_n Prefixo do arquivo relativo a consulta n
              
          Exemplo: python twitter_collector.py c:/twitter_collector 100000 7 0 "donald trump" DT "machine learning" ML neymar NJ "princess charlotte" PC "robotic process automation" RPA
          ''')
        
def main(): 
    NOW_DATE = datetime.now().date()
    
    if len(sys.argv) < 6:
        help()
        sys.exit(-1)
    else:
        consumer_key = '[API key]'
        consumer_secret = '[API secret key]'
    
        access_token = '[Access token]'
        access_token_secret = '[Access token secret]'
    	
        api = TwitterCollector(consumer_key, consumer_secret, access_token, access_token_secret) 
    
        dest_dir = sys.argv[1]
        try:
            max_limit = int(sys.argv[2])
            prev_ini = int(sys.argv[3])
            prev_end = int(sys.argv[4])
        except:
            max_limit = 10
            prev_ini = 0
            prev_end = 0
        terms = {}
        for i in range(5, len(sys.argv), 2):
            term = sys.argv[i]
            prefix = sys.argv[i+1] if (i+1) < len(sys.argv) else 'UNDEFINED'
            terms.update({term : prefix})

        for n in range(prev_ini, prev_end, -1):
            for term, file_prefix in terms.items():
                ts = util.date_to_str(NOW_DATE - timedelta(days=(n)))
                file_name=f"{ts}_{file_prefix}.csv"
                with open(f"{dest_dir}/{util.date_to_str(datetime.now().date())}_EXECUTION_LOG.txt", 'a') as fd:
                    print(f"[{util.now()}] Gerando {dest_dir}/{file_name} ...")
                    print(f"[{util.now()}] Gerando {dest_dir}/{file_name} ...", file = fd)
    
                api.save_tweets(dest_dir,
                                file_name,
                                term,
                                since = NOW_DATE - timedelta(days=n),
                                until = NOW_DATE - timedelta(days=(n-1)),
                                count = max_limit)
            
if __name__ == "__main__": 
    main() 