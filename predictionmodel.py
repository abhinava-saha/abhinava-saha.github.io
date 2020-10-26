
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Encode-the-Categorical-columns:-City,-Toss_decision-and-Venue." data-toc-modified-id="Encode-the-Categorical-columns:-City,-Toss_decision-and-Venue.-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Encode the Categorical columns: City, Toss_decision and Venue.</a></span></li><li><span><a href="#Split-dataset-X-into-Train-and-Test" data-toc-modified-id="Split-dataset-X-into-Train-and-Test-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Split dataset X into Train and Test</a></span></li><li><span><a href="#Model-Performance-Metrics-Function." data-toc-modified-id="Model-Performance-Metrics-Function.-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Model Performance Metrics Function.</a></span></li><li><span><a href="#Building-a-Random-Forest-Classifier-Model-and-fit-it-on-the-Training-Set" data-toc-modified-id="Building-a-Random-Forest-Classifier-Model-and-fit-it-on-the-Training-Set-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Building a Random Forest Classifier Model and fit it on the Training Set</a></span><ul class="toc-item"><li><span><a href="#Generate-Training-data-Model-performance-Metrics" data-toc-modified-id="Generate-Training-data-Model-performance-Metrics-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Generate Training data Model performance Metrics</a></span></li></ul></li><li><span><a href="#Grid-Search-Cross-Validation-of-Random-Forest-Features" data-toc-modified-id="Grid-Search-Cross-Validation-of-Random-Forest-Features-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Grid Search Cross Validation of Random Forest Features</a></span><ul class="toc-item"><li><span><a href="#Generate-Training-data-Model-performance-Metrics" data-toc-modified-id="Generate-Training-data-Model-performance-Metrics-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Generate Training data Model performance Metrics</a></span></li></ul></li><li><span><a href="#Get-Predicted-Winner-for-the-given-input-below." data-toc-modified-id="Get-Predicted-Winner-for-the-given-input-below.-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Get Predicted Winner for the given input below.</a></span></li><li><span><a href="#Import-2020-matches-dataset-for-prediction" data-toc-modified-id="Import-2020-matches-dataset-for-prediction-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Import 2020 matches dataset for prediction</a></span></li></ul></div>

# In[1]:


#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,accuracy_score,roc_auc_score,roc_curve


# In[3]:


matches=pd.read_csv('matches2020.csv')


# In[4]:


matches2020 = matches.copy()
matches.head()
# In[5]:
matches.drop(columns=['umpire1','umpire2','umpire3','id','date','result','dl_applied'], inplace=True)
matches.head()
# In[6]:
matches.info()
# In[7]:
matches.isnull().sum()
# In[8]:
matches.dropna(axis=0, subset=['season', 'winner'], inplace=True)
matches.isnull().sum()
# In[9]:
matches.info()
# In[10]:
matches_2020 = matches[matches.season == 2020]
matches_2020.head()
# In[11]:
sns.catplot(x='city',hue='winner',data=matches_2020, kind='count', height=5, aspect=3)
# In[12]:
sns.catplot(x='winner',hue='city',data=matches_2020, kind='count', height=5, aspect=3)
# In[13]:
sns.catplot(x='city',hue='player_of_match',data=matches_2020, kind='count', height=5, aspect=3)
# In[14]:
matches_2020[['winner', 'city']].pivot_table(index=['winner'],
                                             columns=['city'],
                                             aggfunc=len, 
                                             margins=True).sort_values('All', ascending=False )
# In[15]:
matches_2020[['winner','player_of_match','city']].pivot_table(index=['winner','player_of_match'],
                                                              columns=['city'], 
                                                              aggfunc=len, margins=True)
# In[16]:
matches_2020[['winner','toss_winner','city', 'toss_decision']].pivot_table(index=['toss_winner','winner'],
                                                              columns=['toss_decision','city'], 
                                                              aggfunc=len, margins=True)
# In[17]:
matches_2020.toss_decision.value_counts()
# In[18]:
matches.winner.value_counts()
# In[19]:
#remove any null values, winner has hence fill the null value in winner as draw
#City is also null
matches.describe(include='all')
# In[20]:
matches1 = matches[['season','team1','team2','city','toss_decision','toss_winner','venue','winner']].copy()
matches1.tail(20)
# In[21]:
matches1.toss_decision.value_counts()
# In[22]:
matches1.city.value_counts()
# ## Encode the Categorical columns: City, Toss_decision and Venue.
# In[23]:
pd.Categorical(matches['winner'],ordered=False).categories
# In[24]:
# Create dictionary of Team names and team codes.
teams = {'team_names': {'Chennai Super Kings':'CSK', 'Deccan Chargers':'DCH', 'Delhi Capitals':'DC',
       'Delhi Daredevils':'DD', 'Gujarat Lions':'GL', 'Kings XI Punjab':'KXIP',
       'Kochi Tuskers Kerala':'KTK', 'Kolkata Knight Riders':'KKR', 'Mumbai Indians':'MI',
       'Pune Warriors':'PW', 'Rajasthan Royals':'RR', 'Rising Pune Supergiant':'RPS',
       'Rising Pune Supergiants':'RPS', 'Royal Challengers Bangalore':'RCB',
       'Sunrisers Hyderabad':'SRH', 'Tie':'TIE'}, 
         
         'team_codes': {'MI':1, 'CSK':2, 'KKR':3, 'RCB':4, 'KXIP':5, 'RR':6, 'SRH':7, 'DC':8,   
              'DD':9, 'DCH':10, 'GL':11, 'KTK':12, 'PW':13, 'RPS':14, 'TIE':15},
         
         'team_codes_rev': {1:'MI', 2:'CSK', 3:'KKR', 4:'RCB', 5:'KXIP', 6:'RR', 7:'SRH', 8:'DC',   
              9:'DD', 10:'DCH', 11:'GL', 12:'KTK', 13:'PW', 14:'RPS', 15:'TIE'}
        }
# In[25]:
# Encode venue, city and toss_decision columns using Sklearn LabelEncoder
from sklearn.preprocessing import LabelEncoder
le_venue = LabelEncoder()
le_venue = le_venue.fit(matches1['venue'])
le_city = LabelEncoder()
le_city = le_city.fit(matches1['city'])
le_toss = LabelEncoder()
le_toss = le_toss.fit(matches1['toss_decision'])

def to_numerical(df, teams, le_venue, le_city, le_toss):
    df.replace(teams['team_names'], inplace=True)
    df.replace(teams['team_codes'], inplace=True)
    
    df['venue'] = le_venue.transform(df['venue'])
    df['city'] = le_city.transform(df['city'])
    df['toss_decision'] = le_toss.transform(df['toss_decision'])
    return df
# In[26]:
# Define function to get team name passing team code.
def get_team_name(team_no, teams):
    team = team_no
    team_codes = teams['team_codes']
    team_names = teams['team_names']
    teamcode = list(team_codes.keys())[list(team_codes.values()).index(team)]
    teamname = list(team_names.keys())[list(team_names.values()).index(teamcode)]
    #print ("Team {} is {}".format(team, teamname))
    return teamname
# In[27]:
for a in range(1,15):
    print("Team {} is".format(a) , end = ': ') 
    print(get_team_name(a, teams))
# In[28]:
matches1 = to_numerical(matches1, teams, le_venue, le_city, le_toss)
matches1.head()
# In[29]:
le_venue.classes_
# In[30]:
le_venue.transform(['Holkar Cricket Stadium'])
# In[31]:
le_venue.classes_[matches1.venue[2]]
# In[32]:
tosswinner = matches1[matches1.season == 2020].toss_winner.value_counts()
winner = matches1[matches1.season == 2020].winner.value_counts()
# In[33]:
#print("\033[1mTotal Toss won by Teams\n\033[0m")
a, b = [], []
for idx, count in tosswinner.items():
    a.append(get_team_name(idx, teams))
 #   print("{:<30} -> {:2d}".format(get_team_name(idx, teams),count))

#print("\033[1m\nTotal Matches Won by Teams\n\033[0m")
for idx, count in winner.items():
    b.append(get_team_name(idx, teams))
 #   print("{:<30} -> {:2d}".format(get_team_name(idx, teams),count))
# In[34]:
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(121)
tosswinner.plot(kind='bar');
ax1.set_xlabel('Team')
ax1.set_ylabel('Count of toss wins')
ax1.set_title("toss winners")


ax2 = fig.add_subplot(122)
winner.plot(kind = 'bar');
ax2.set_xlabel('Team')
ax2.set_ylabel('count of matches won')
ax2.set_title("Match winners")

#plt.tight_layout()
#plt.show()


# ## Split dataset X into Train and Test 
# 

# In[35]:


X = matches1.copy()
y = X.pop('winner')
X.head()
# In[36]:
# In[37]:
random_state = 100
# ## Model Performance Metrics Function.
# In[38]:
def classification_model(models, X_predictors, y_actual):
    
    for model, models_name in models.items():
        print("Generating Metrics for {} \n\t{}".format(models_name, model))
        y_predicted = model.predict(X_predictors)
        y_predicted_proba = model.predict_proba(X_predictors)
        model_performance_metrics(models_name, y_actual, y_predicted)
# In[39]:
def model_performance_metrics(models_name, y_actual, y_predicted):
    print('\nAccuracy for {} model is'.format(models_name),'\n',accuracy_score(y_actual, y_predicted))
    print('\n')
    print('Classification report for {} model is'.format(models_name),'\n',classification_report(y_actual, y_predicted))
    print('\n')
    print('Confusion Matrix for {} model is'.format(models_name))
    sns.heatmap(confusion_matrix(y_actual, y_predicted),annot=True,fmt='d',cbar=False)
    plt.title('Confusion Matrix for {}'.format(models_name))
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.show()
# ## Building a Random Forest Classifier Model and fit it on the Training Set
# 
# In[40]:
rfc = RandomForestClassifier(n_estimators=101,
                             max_depth=20,
                             max_features=7,
                             min_samples_leaf=3,
                             min_samples_split=3,
                             class_weight='balanced',
                             random_state=random_state,
                             oob_score=True)
# In[41]:
rfc.fit(X, y)
# In[42]:
x=pd.DataFrame(rfc.feature_importances_*100,index=X.columns).sort_values(by=0,ascending=False)
plt.figure(figsize=(12,7))
sns.barplot(x[0],x.index,palette='rainbow')
plt.ylabel('Feature Name')
plt.xlabel('Feature Importance in %')
plt.title('Feature Importance Plot')
#plt.show()
# In[43]:
#models={dtc:'DecisionTree Classifier',rfc:'RandomForest Classifier',mlp:'Artificial Neural Network(ANN)'}
models={rfc:'RandomForest Classifier'}
# ### Generate Training data Model performance Metrics
# In[44]:
print ("\nChecking the Metrics for Predictions on the Training set\n")
classification_model(models, X, y)
# ## Grid Search Cross Validation of Random Forest Features
# In[45]:
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [10,15,20],
    'max_features': [5,6,7],
    'min_samples_leaf': [1,3,5],
    'min_samples_split': [1,3,5],
    'n_estimators': [101,151]
}

rfcl = RandomForestClassifier(random_state=random_state, class_weight='balanced')
grid_search = GridSearchCV(estimator = rfcl, param_grid = param_grid, cv = 3)

# In[46]:

grid_search.fit(X, y)

# In[47]:

grid_search.best_params_

# In[48]:
best_grid = grid_search.best_estimator_
# In[49]:
best_grid
# In[50]:
x=pd.DataFrame(best_grid.feature_importances_*100,index=X.columns).sort_values(by=0,ascending=False)
plt.figure(figsize=(12,7))
sns.barplot(x[0],x.index,palette='rainbow')
plt.ylabel('Feature Name')
plt.xlabel('Feature Importance in %')
plt.title('Feature Importance Plot')
#plt.show()


# In[51]:

model_gscv = {rfc:'RandomForest Classifier', best_grid:'GridSearchCV - RandomForest Classifier'}

# ### Generate Training data Model performance Metrics
# In[52]:

#print ("\nChecking the Metrics for Predictions on the Training set\n")
classification_model(model_gscv, X, y)
# ## Get Predicted Winner for the given input below.
# In[53]:
#'team1', 'team2','city', 'toss_decision', 'toss_winner', 'venue',
teamcode = teams['team_codes']
def predict_winner(model, input):
    output=model.predict(input)
    print("The predicted winner is Team {} - {}".format(output, get_team_name(output, teams)))
# In[54]:
season = 2020
team1= teamcode['KKR'] 
team2= teamcode['RCB']
city = le_city.transform(['Bangalore'])[0] 
toss_decision = le_toss.transform(['field'])[0]  #[0-'bat', 1-'field']
toss_winner= teamcode['RCB']
venue = le_venue.transform(['M Chinnaswamy Stadium'])[0]  

input=[season, team1, team2, city, toss_decision, toss_winner, venue]
input = np.array(input).reshape((1, -1))
print(input)

predict_winner(best_grid, input)
# In[1]:


season = 2020
team1= teamcode['KXIP'] 
team2= teamcode['MI']
city = le_city.transform(['Chandigarh'])[0] 
toss_decision = le_toss.transform(['field'])[0]  #[0-'bat', 1-'field']
toss_winner= teamcode['MI']
venue = le_venue.transform(['Punjab Cricket Association Stadium, Mohali'])[0]  #17

input=[season, team1, team2, city, toss_decision, toss_winner, venue]
input = np.array(input).reshape((1, -1))
print(input)

predict_winner(rfc, input)


# ## Import 2020 matches dataset for prediction

# In[56]:


def predict_result(model, df, teams, le_venue, le_city, le_toss):
    df1 = df[['season','team1','team2','city','toss_decision','toss_winner','venue']].copy()
    if df1.isnull().sum().sum() == 0:
        df1 = to_numerical(df1, teams, le_venue, le_city, le_toss)
        winner = model.predict(df1)
        df['predict_winner'] = winner
        df['predict_winner'] = df['predict_winner'].replace(teams['team_codes_rev'])
        df.replace(teams['team_names'], inplace=True)
        return df
    else:
        print("Remove the null values from data.")
        df.isnull().sum()
        return df


# In[57]:


set1 = pd.read_csv('predict_ipl_2020.csv')
# In[58]:
predict_set1 = set1 #Input dataset for perdiction
model = best_grid  # Model to be used for prediction
predict_set1 = predict_result(model, predict_set1, teams, le_venue, le_city, le_toss)
predict_set1.head(10)
# In[59]:
# Print the output to file
#predict_set1.to_csv('IPL-2020_Predicted_Winner_11-Oct-2020_Run1.csv', index=True)

pickle.dump(model, open('iri.pkl', 'wb'))
