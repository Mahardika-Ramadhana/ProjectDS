import certifi
import os
os.environ['SSL_CERT_FILE'] = certifi.where()
import snscrape.modules.twitter as sntwitter
import pandas as pd
from textblob import TextBlob

# Query pencarian tweet
query = "fauzan lang:id"
max_tweets = 100  # Jumlah maksimal tweet yang diambil

tweets_data = []
print("Mengambil data tweet dari Twitter dengan snscrape...")
for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
    if i >= max_tweets:
        break
    tweets_data.append({'Konten': tweet.content})

if not tweets_data:
    print("Tidak ada data tweet yang ditemukan.")
else:
    # Analisis sentimen
    def analisis_sentimen(teks):
        analisis = TextBlob(teks)
        polaritas = analisis.sentiment.polarity
        if polaritas > 0:
            return 'Positif'
        elif polaritas < 0:
            return 'Negatif'
        else:
            return 'Netral'

    df = pd.DataFrame(tweets_data)
    print("Menganalisis sentimen...")
    df['Sentimen'] = df['Konten'].apply(analisis_sentimen)
    df.to_csv("hasil_sentimen_twitter.csv", index=False)
    print("Selesai! Hasil disimpan di 'hasil_sentimen_twitter.csv'.")
    print(df['Sentimen'].value_counts())

