
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv(r"C:\Users\raghu\Downloads\datasetb2d9982 (1)\dataset\train.csv")

# Select the features and target
features = ['TITLE', 'DESCRIPTION', 'BULLET_POINTS', 'PRODUCT_TYPE_ID']
target = 'PRODUCT_LENGTH'

for feature in features:
    data[feature] = data[feature].fillna('')
# Split the data into training and validation sets
X = data[features]
y = data[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)

# Define a text preprocessing pipeline
text_preprocessor = Pipeline([
    ('tfidf', TfidfVectorizer())
])

# Define a column transformer to apply the text preprocessing pipeline to the text features
preprocessor = ColumnTransformer([
    ('text', text_preprocessor, 'DESCRIPTION'),
    ('title', text_preprocessor, 'TITLE'),
    ('bullet_points', text_preprocessor, 'BULLET_POINTS'),

])

# Define a pipeline that applies the preprocessor and then trains a random forest regressor
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=0))
])

# Train the model on the training set
model.fit(X_train, y_train)

# Evaluate the model on the validation set
val_score = model.score(X_val, y_val)
print(f'Validation R^2 score: {val_score:.2f}')
nlp = spacy.load("en")