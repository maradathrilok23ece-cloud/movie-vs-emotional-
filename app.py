LogisiticRegression
keyboard_arrow_down Step 1: Import Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
keyboard_arrow_down Step 2: Load Dataset
data = pd.read_csv("Movie vs Emotional Attachment Survey 2024-2026.csv")
keyboard_arrow_down Step 3: View Dataset
Age Gender
Watched
emotional
movie in
last 12
months
Movie/Book
that
affected
you most
recently
How
many
times
watched
When
watched
most
recently
The
storytelling
in this
movie was
emotionally
powerful.
The story
made me
feel
strong
emotions.
The
emotional
moments
felt
genuine
and
realistic.
The
characters'
emotional
experiences
were
clearly
expressed.
The story
focused
strongly on
characters'
feelings and
relationships.
a
cha
0
18–
20 Male Yes La La Land 1 time
1–6
months
ago
3 2 2 4 4
1 Under
18 Male Yes Mirai
More
than 3
times
1–6
months
ago
3 5 4 5 3
2
21–
23 Female Yes Aftersun 1 time
1–6
months
ago
4 4 4 4 4
3
21–
23 Female Yes Room
More
than 3
times
1–6
months
ago
4 5 3 5 4
4
18–
20 Female Yes About Time
More
than 3
times
1–6
months
ago
1 1 1 1 1
data.head()
keyboard_arrow_down Step 4: Prepare Features (X) and Target (y)
print(list(data.columns))
['Age', 'Gender', 'Watched emotional movie in last 12 months', 'Movie/Book that affected you most recently', 'How many times wat
data.columns = data.columns.str.strip() \
.str.lower() \
.str.replace(" ", "_") \
.str.replace("'", "") \
.str.replace("?", "") \
.str.replace("__", "_")
3/31/26, 5:56 PM Movie vs Emotional Attachment Survey 2024-2026.ipynb - Colab
https://colab.research.google.com/drive/1w7xYX7416poF_5aRuDQMjlSoRR1WFd0t?usp=sharing#scrollTo=cBonycQq7xHG&printMode=true 1/6
print(list(data.columns))
['age', 'gender', 'watched_emotional_movie_in_last_12_months', 'movie/book_that_affected_you_most_recently', 'how_many_times_wat
---------------------------------------------------------------------------
KeyError Traceback (most recent call last)
/tmp/ipykernel_151/3224099195.py in <cell line: 0>()
----> 1 X = data[['age',
 2 'gender',
 3 'watched_emotional_movie_last_12_months',
 4 'times_watched',
 5 'recent_watch_time',
2 frames
/usr/local/lib/python3.12/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
 6250
 6251 not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252 raise KeyError(f"{not_found} not in index")
 6253
 6254 @overload
KeyError: "['watched_emotional_movie_last_12_months', 'times_watched', 'recent_watch_time', 'emotional_power_storytelling',
'felt_strong_emotions', 'emotional_moments_realistic', 'characters_emotions_clearly_expressed', 'story_focus_on_relationships']
not in index"
X = data[['age',
'gender',
'watched_emotional_movie_last_12_months',
'times_watched',
'recent_watch_time',
'emotional_power_storytelling',
'felt_strong_emotions',
'emotional_moments_realistic',
'characters_emotions_clearly_expressed',
'story_focus_on_relationships']]
X = pd.get_dummies(X, drop_first=True)
y = data['audience_attachment_score']
---------------------------------------------------------------------------
NameError Traceback (most recent call last)
/tmp/ipykernel_151/2016476978.py in <cell line: 0>()
----> 1 X.shape
NameError: name 'X' is not defined
X.shape
---------------------------------------------------------------------------
KeyError Traceback (most recent call last)
/tmp/ipykernel_151/3475757778.py in <cell line: 0>()
----> 1 y=data[['inactive_3_months_flag']]
2 frames
/usr/local/lib/python3.12/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
 6247 if nmissing:
 6248 if nmissing == len(indexer):
-> 6249 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
 6250
 6251 not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
KeyError: "None of [Index(['inactive_3_months_flag'], dtype='object')] are in the [columns]"
y=data[['inactive_3_months_flag']]
y.shape
(50000, 1)
keyboard_arrow_down Step 5: Split Dataset
3/31/26, 5:56 PM Movie vs Emotional Attachment Survey 2024-2026.ipynb - Colab
https://colab.research.google.com/drive/1w7xYX7416poF_5aRuDQMjlSoRR1WFd0t?usp=sharing#scrollTo=cBonycQq7xHG&printMode=true 2/6
---------------------------------------------------------------------------
NameError Traceback (most recent call last)
/tmp/ipykernel_151/3961619198.py in <cell line: 0>()
----> 1 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
NameError: name 'X' is not defined
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
---------------------------------------------------------------------------
NameError Traceback (most recent call last)
/tmp/ipykernel_151/4225672638.py in <cell line: 0>()
----> 1 X_train.shape
NameError: name 'X_train' is not defined
X_train.shape
---------------------------------------------------------------------------
NameError Traceback (most recent call last)
/tmp/ipykernel_151/3624294392.py in <cell line: 0>()
----> 1 X_test.shape
NameError: name 'X_test' is not defined
X_test.shape
---------------------------------------------------------------------------
NameError Traceback (most recent call last)
/tmp/ipykernel_151/3798806461.py in <cell line: 0>()
----> 1 y_train.shape
NameError: name 'y_train' is not defined
y_train.shape
y_test.shape
(20000, 1)
keyboard_arrow_down Step 6: Train Model
/usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed wh
 y = column_or_1d(y, warn=True)
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (sta
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.
Increase the number of iterations (max_iter) or scale the data as shown in:
 https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
 https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
 n_iter_i = _check_optimize_result(
▾ LogisticRegression i ?
LogisticRegression()
LR = LogisticRegression(max_iter=1000)
LR.fit(X_train, y_train)
keyboard_arrow_down Step 7: Make Predictions
y_pre = LR.predict(X_test)
3/31/26, 5:56 PM Movie vs Emotional Attachment Survey 2024-2026.ipynb - Colab
https://colab.research.google.com/drive/1w7xYX7416poF_5aRuDQMjlSoRR1WFd0t?usp=sharing#scrollTo=cBonycQq7xHG&printMode=true 3/6
accuracy = accuracy_score(y_test, y_pre)
print("Accuracy:", accuracy)
Accuracy: 0.7792
Decision Tree
keyboard_arrow_down Step 1: Import Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
keyboard_arrow_down Step 2: Load Dataset
data = pd.read_csv("Movie vs Emotional Attachment Survey 2024-2026.csv")
keyboard_arrow_down Step 3: View Dataset
Age Gender
Watched
emotional
movie in
last 12
months
Movie/Book
that
affected
you most
recently
How
many
times
watched
When
watched
most
recently
The
storytelling
in this
movie was
emotionally
powerful.
The story
made me
feel
strong
emotions.
The
emotional
moments
felt
genuine
and
realistic.
The
characters'
emotional
experiences
were
clearly
expressed.
The story
focused
strongly on
characters'
feelings and
relationships.
a
cha
0
18–
20 Male Yes La La Land 1 time
1–6
months
ago
3 2 2 4 4
1 Under
18 Male Yes Mirai
More
than 3
times
1–6
months
ago
3 5 4 5 3
2
21–
23 Female Yes Aftersun 1 time
1–6
months
ago
4 4 4 4 4
3
21–
23 Female Yes Room
More
than 3
times
1–6
months
ago
4 5 3 5 4
4
18–
20 Female Yes About Time
More
than 3
times
1–6
months
ago
1 1 1 1 1
data.head()
keyboard_arrow_down Step 4: Prepare Features (X) and Target (y)
X = data[['Age',
'Gender',
'Watched Emotional Movie Last 12 Months',
'Times Watched',
'When Watched Most Recently',
'The storytelling in this movie was emotionally powerful',
'The story made me feel strong emotions',
'The emotional moments felt genuine and realistic',
"The characters' emotional experiences were clearly expressed",
3/31/26, 5:56 PM Movie vs Emotional Attachment Survey 2024-2026.ipynb - Colab
https://colab.research.google.com/drive/1w7xYX7416poF_5aRuDQMjlSoRR1WFd0t?usp=sharing#scrollTo=cBonycQq7xHG&printMode=true 4/6
"The story focused strongly on characters' feelings and relationships"]]
# Convert categorical data
X = pd.get_dummies(X, drop_first=True)
# Target
y = data['Audience Attachment Score']
X.shape
(7050, 9)
y=data[['status_type']]
y.shape
(7050, 1)
keyboard_arrow_down Step 5: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape
(4935, 9)
X_test.shape
(2115, 9)
y_train.shape
(4935, 1)
y_test.shape
(2115, 1)
keyboard_arrow_down Step 6: Train Model
DT=DecisionTreeClassifier()
DT.fit(X_train, y_train)
y_pre = DT.predict(X_test)
accuracy = accuracy_score(y_test, y_pre)
print("Accuracy:", accuracy)
Accuracy: 0.775886524822695
3/31/26, 5:56 PM Movie vs Emotional Attachment Survey 2024-2026.ipynb - Colab
https://colab.research.google.com/drive/1w7xYX7416poF_5aRuDQMjlSoRR1WFd0t?usp=sharing#scrollTo=cBonycQq7xHG&printMode=true 5/6
3/31/26, 5:56 PM Movie vs Emotional Attachment Survey 2024-2026.ipynb - Colab
https://colab.research.google.com/drive/1w7xYX7416poF_5aRuDQMjlSoRR1WFd0t?usp=sharing#scrollTo=cBonycQq7xHG&printMode=true 6/6
