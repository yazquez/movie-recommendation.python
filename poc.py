from collections import defaultdict
text = "from collections import defaultdict from collections import defaultdict from from import"
d = defaultdict(float)
for w in text.split():
  d[w] += 1.5

for w in sorted(d, key=d.get, reverse=True)[:2]:
  print (w, d[w])



import pandas as pd


c_names = ['post1', 'post2', 'post3', 'post4']
words = ['ice', 'snow', 'tahoe', 'goal', 'puck']
post_words = pd.DataFrame([[4, 4, 6, 2],
                           [6, 1, 0, 5],
                           [3, 0, 0, 5],
                           [0, 6, 5, 1],
                           [0, 4, 5, 0]],
                          index = words,
                          columns = c_names)
post_words.index.names = ['word:']
post_words