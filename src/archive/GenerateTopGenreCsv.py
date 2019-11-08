'''
The genre section has too many sections. If we use 1-hot encoding to find the distance, then we have too many 0 distances. Need to parse non-important categories to "Others"
'''

import pandas as pd
import numpy as np
import json
from collections import Counter
import sys

if len(sys.argv) < 2:
	assert Error("Not enough input arguments.")

input_csv = sys.argv[1] # '../movies_short2.csv' # modify here.
print("Input CSV=", input_csv)

# Read csv + get genre column
X = np.array(pd.read_csv(input_csv))
X_genres = X[:,1]

# Parse hash
genre_name_count_hash = {}
genre_name_id_hash = {}
for x_genre in X_genres:
	genres = json.loads(x_genre)    # every X data's genres
	for g in genres:                # every genre in X data's genre
		if g['name'] in genre_name_count_hash.keys():
			genre_name_count_hash[g['name']] += 1
		else:
			genre_name_count_hash[g['name']] = 1

		genre_name_id_hash[g['name']] = g['id']

# print(genre_name_count_hash)
# print(genre_name_id_hash)

# Get top 5 genre
k = Counter(genre_name_count_hash)
top_genres = dict(k.most_common(5)) # Finding 5 highest values


# Modify existing data.
new_X_genres = []
for x_genre in X_genres:
	genres = json.loads(x_genre)    # every X data's genres

	added_genres = []
	for g in genres:                # every genre in X data's genre
		if g['name'] in top_genres.keys():
			added_genres.append(g['name'])

	genres_json = []
	for ag in added_genres:
		genres_json.append({"id": genre_name_id_hash[ag], "name": ag})

	if len(genres_json) == 0:
		empty_genre = [{"id": 99, "name": "Others"}]
		new_X_genres.append(json.dumps(empty_genre))
	else:
		new_X_genres.append(json.dumps(genres_json))

new_X_genres = np.array(new_X_genres)

X[:, 1] = new_X_genres

header = ["budget","genres","homepage","id","keywords","original_language","original_title","overview","popularity","production_companies","production_countries","release_date","revenue","runtime","spoken_languages","status","tagline","title","vote_average","vote_count"]

output_csv = input_csv[:-4] + "_top5genre.csv"
print("Output CSV=", output_csv)

pd.DataFrame(X).to_csv(output_csv, header=header, index=None)
