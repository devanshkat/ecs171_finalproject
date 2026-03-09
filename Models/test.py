from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv('instagram_usage_lifestyle.csv')

features = [
    'age',    'daily_active_minutes_instagram',
    'self_reported_happiness'
]

target = ['perceived_stress_score']

scaler = StandardScaler()
minmax = MinMaxScaler(feature_range=(1,10))

df = df[df['country'] == 'United States']

df = df.head(50000)
# df['active_consumption'] = (df['dms_sent_per_week']/7 + df['comments_written_per_day'] + df['posts_created_per_week']) / df['daily_active_minutes_instagram']
# df['passive_consumption'] = (df['reels_watched_per_day'] + df['stories_viewed_per_day']) / df['daily_active_minutes_instagram']
# df['active_ratio'] = df['active_consumption'] / ( df['passive_consumption'] + df['active_consumption'])
data = df[['age',
           'daily_active_minutes_instagram',
'self_reported_happiness', 'perceived_stress_score']]

X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = minmax.fit_transform(y_train)

# Linear regression model
linearModel = LinearRegression()
linearModel.fit(X_train, y_train)

joblib.dump(linearModel, 'linear_regression.joblib')
joblib.dump(scaler, 'scaler_linear.joblib')