import json
import random
import re
import sqlite3
from newspaper import Article
import time

def time_from_fraction(f):
    i_part = int(f)
    f_part = f - i_part
    f_part *= 60
    return f"{str(i_part)}:{str(int(f_part))}"

def truncate(num):
    return re.sub(r'^(\d+\.\d{,2})\d*$',r'\1',str(num))

with open('./../news.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

avg_time=0
data=data[9113:]
conn = sqlite3.connect('./../messages.db')
cursor = conn.cursor()
cnt=0
size_d=len(data)
batches_size=size_d/100
# random.shuffle(data)
cycle_start = time.time()
cycle_end=0
for d in data:
    url = d[0]
    ref_ttl=d[1]
    ref_src=d[2]
    date=d[3]
    parseOk=False
    cnt+=1
    try:
        a = Article(url, language='ru') # Chinese
        a.download()
        a.parse()
        print(f"[{cnt}] {a.title}")
        parseOk=True
    except:
        print(url)
    if parseOk:
        cursor.execute( "INSERT INTO news (url,ref_ttl, ref_src, date, text, tags,authors,title) VALUES (?, ?, ?, ?, ?,?,?,?)", (url, ref_ttl, ref_src, date, a.text, ','.join(a.tags), ','.join(a.authors), a.title))

    if cnt%100==0:
        cycle_end=time.time()
        hop_time = (cycle_end - cycle_start)
        avg_time+=hop_time*2 if avg_time==0 else hop_time
        avg_time/=2

        print(f"{cnt} from {len(data)} received... spent {truncate(hop_time/60)} minutes, estimated {time_from_fraction(((size_d - cnt) / 100) * avg_time / 60 / 60)} left")
        cycle_start=time.time()
        conn.commit()


conn.close()