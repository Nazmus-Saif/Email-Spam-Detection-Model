import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
style.use('ggplot')
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split






df = pd.read_csv("Modified Spam or Ham.csv", encoding='latin1')  # extra performance 5537data
print(f"\nFirst 5 Samples of Dataset : \n {df.head()}")
print(f"\nShape of Dataset : {df.shape}")
print("\n", df.info())





# let's create a count plot to see different sentiment
colors = {"positive": "green", "negative": "red"}  # Define a color palette
sns.countplot(x='sentiment', data=df, palette=colors.values())
plt.title("Sentiment Distribution")
plt.show()





# lets create a pie chart
plt.pie(df['sentiment'].value_counts(), labels=['Negative', 'Positive'], autopct="%0.2f%%")
plt.title("Positive & Negative Word Ratio")
plt.show()





# let's show 5 reviews and its corresponding sentiment
for i in range(5):
    print(f"Review : {[i]}")
    print(df['review'].iloc[i])
    print(f"Sentiment : {df['sentiment'].iloc[i]}\n\n")





# let's find the number of words in each review
def no_of_words(text):
    words = text.split()
    word_count = len(words)
    return word_count

df['word count'] = df['review'].apply(no_of_words)
print(f"\nFirst 5 Samples of Dataset with Word Count : \n {df.head()}")





# bar chart for sentence lengths of each review
plt.figure(figsize=(15, 8))
plt.bar(df.index, df['word count'], color=df['sentiment'].map(colors))
plt.title('Number of Words in Each Review')
plt.xlabel('Review Index')
plt.ylabel('Number of Words')
# Set x-axis limits
plt.ylim(0, 1500)
plt.show()





# let's see the word count plot
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].hist(df[df['sentiment'] == 'positive']['word count'], label='Positive', color='blue', rwidth=0.8)
ax[0].legend(loc='upper right')
ax[1].hist(df[df['sentiment'] == 'negative']['word count'], label='Negative', color='orange', rwidth=0.8)
ax[1].legend(loc='upper right')
fig.suptitle("Number of Words in Review")
plt.show()





# let's replace the positive word into 1 & negative word into 2 in sentiment column
df.sentiment.replace("positive", 1, inplace=True)
df.sentiment.replace("negative", 0, inplace=True)
print(f"\nFirst 5 Samples of Dataset after replace Positive with 1 & Negative with 0 : \n{df.head()}\n")





# data processing to convert the text into useful formate
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
nltk.download('wordnet')
def data_processing(review_text):
    clean_text1 = review_text.lower()
    clean_text2 = re.sub('<br />', '', clean_text1)
    clean_text3 = re.sub('\[[^]]*]', '', clean_text2)
    clean_text4 = re.sub(r"https\S+|www\S+|http\S+", '', clean_text3, flags=re.MULTILINE)
    clean_text5 = re.sub(r'@\w+|#', '', clean_text4)
    clean_text6 = re.sub(r'[^\w\s]', '', clean_text5)
    clean_text7 = re.sub(r'\d+', '', clean_text6)  # remove digits
    text_tokens = word_tokenize(clean_text7)

    return " ".join(text_tokens)

df.review = df['review'].apply(data_processing)





# # Save the preprocessed DataFrame to a new CSV file
# df.to_csv('Processed Data of IMDB.csv', index=False)
# print("Preprocessed data saved to 'Processed Data of IMDB.csv'")





# let's see the word count to see the change in review
df['word count'] = df['review'].apply(no_of_words)
print(f"\nFirst 5 Samples of Dataset with Word Count after Preprocessing : \n {df.head()}")





# let's see the word count plot after preprocessing
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].hist(df[df['sentiment'] == 1]['word count'], label='Positive', color='blue', rwidth=0.8)
ax[0].legend(loc='upper right')
ax[1].hist(df[df['sentiment'] == 0]['word count'], label='Negative', color='orange', rwidth=0.8)
ax[1].legend(loc='upper right')
fig.suptitle("Number of Words in Review after Preprocessing")
plt.show()





# let's visualize the positive reviews in wordcloud
pos_reviews = df[df.sentiment == 1]
print(f"\n\nList of only Positive Reviews showing top 5 : \n{pos_reviews.head()}")
# wordcloud
text = ' '.join([word for word in pos_reviews['review']])
plt.figure(figsize=(10, 5), facecolor='None')
wordcloud = WordCloud(max_words=500, width=800, height=400).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words in Positive Reviews', fontdict={'fontsize': 15})
plt.show()

# let's visualize the most frequent word in positive reviews
from collections import Counter  # this is needed for counting positive and negative reviews using Counter

count = Counter()
for text in pos_reviews['review'].values:
    for word in text.split():
        count[word] += 1
print("\nMost Common Words in Positive Reviews:\n", count.most_common(10), "\n\n")

# let's create a data frame
pos_words = pd.DataFrame(count.most_common(10))
pos_words.columns = ['word', 'count']
print("\nDataFrame for Visualization:\n", pos_words.head(), "\n\n")
# Visualize the data in a bar chart
plt.figure(figsize=(10, 5))
sns.barplot(x='count', y='word', data=pos_words, palette="Blues_d")
plt.title('Common Words in Positive Reviews')
plt.show()




# let's visualize the negative reviews in wordcloud
neg_reviews = df[df.sentiment == 0]
print(f"List of only Negative Reviews showing top 5 : \n{neg_reviews.head()}")
#wordcloud
text = ' '.join([word for word in neg_reviews['review']])
plt.figure(figsize=(10, 5), facecolor='None')
wordcloud = WordCloud(max_words=500, width=800, height=400).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words in Negative Reviews', fontdict={'fontsize': 15})
plt.show()

# let's visualize the most frequent word in negative reviews
count = Counter()
for text in neg_reviews['review'].values:
    for word in text.split():
        count[word] += 1
print("\nMost Common Words in Negative Reviews:\n", count.most_common(10), "\n\n")

# let's create a data frame
neg_words = pd.DataFrame(count.most_common(10))
neg_words.columns = ['word', 'count']
print("\nDataFrame for Visualization:\n", neg_words.head(), "\n\n")
# Visualize the data in a bar chart
plt.figure(figsize=(10, 5))
sns.barplot(x='count', y='word', data=neg_words, palette="Oranges_d")
plt.title('Common Words in Negative Reviews')
plt.show()




# data splitting & vectorize
X = df['review']
Y = df['sentiment']
# let's vectorize the data
# vect = CountVectorizer(ngram_range=(1, 2))
vect = TfidfVectorizer(ngram_range=(1, 2))
X_Vect = vect.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X_Vect, Y, test_size=0.2, random_state=42)
print(f"\nSize of x_train : {x_train.shape}")
print(f"Size of y_train : {y_train.shape}")
print(f"\nSize of x_test : {x_test.shape}")
print(f"Size of y_test : {y_test.shape}")





# let's create a machine learning model
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings  # for ignore any kind of warning
warnings.filterwarnings('ignore')
from sklearn.tree import ExtraTreeClassifier
from sklearn.linear_model import SGDClassifier






def ModeTrainTestBeforeTuning(model_name):
    model = model_name()
    model.fit(x_train, y_train)
    # Predict on the training set
    model_train_predictions = model.predict(x_train)
    model_train_accuracy = accuracy_score(model_train_predictions, y_train)
    print(f"Training Accuracy: {model_train_accuracy * 100:.2f}%")
    # Predict on the test set
    model_test_predictions = model.predict(x_test)
    model_test_accuracy = accuracy_score(model_test_predictions, y_test)
    print(f"Test Accuracy: {model_test_accuracy * 100:.2f}%")
    # Accuracy difference of the model
    accuracy_Difference = model_train_accuracy - model_test_accuracy
    print(f"Accuracy Difference: {accuracy_Difference * 100:.2f}%\n")
    # Plot the bar chart for Training Accuracy, Testing Accuracy, and Accuracy Difference
    labels = ['Training Accuracy', 'Testing Accuracy', 'Accuracy Difference']
    values = [model_train_accuracy * 100, model_test_accuracy * 100, accuracy_Difference * 100]
    plt.bar(labels, values, color=['blue', 'orange', 'red'])
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model} Model Accuracy Comparison')
    plt.ylim(0, 100)  # Set the y-axis limits between 0 and 1
    plt.show()
    # print the confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, model_test_predictions))
    # print the classification report
    print("\nClassification Report:")
    print(classification_report(y_test, model_test_predictions))
    # plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, model_test_predictions), annot=True, fmt="d", cmap="Blues", cbar=False)  # cbar show the color bar in right side
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model} Confusion Matrix")
    plt.show()
    return model





print("\n\nSGDClassifier Model Result:")
ModeTrainTestBeforeTuning(SGDClassifier)

print("\nLogistic Regression Model Result:")
ModeTrainTestBeforeTuning(LogisticRegression)

print("\nMultinomialNB Model Result:")
ModeTrainTestBeforeTuning(MultinomialNB)

print("\nBernoulliNB Model Result:")
ModeTrainTestBeforeTuning(BernoulliNB)

print("\nLinearSVC Model Result:")
ModeTrainTestBeforeTuning(LinearSVC)

print("\nRidgeClassifier Model Result:")
ModeTrainTestBeforeTuning(RidgeClassifier)

print("\n\nExtraTreeClassifier Model Result:")
ModeTrainTestBeforeTuning(ExtraTreeClassifier)






from sklearn.model_selection import StratifiedKFold, cross_val_score
def perform_cross_validation(model, x, y):
    # Initialize StratifiedKFold
    stratified_kFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # Perform cross-validation
    cross_val_results = cross_val_score(model, x, y, cv=stratified_kFold, scoring='accuracy')
    # Print cross-validation results
    print(f'Cross-Validation Results Accuracy: {cross_val_results}')
    print(f'Mean Accuracy: {cross_val_results.mean() * 100:.2f}%')
    print(f'Standard Deviation: {cross_val_results.std():.4f}\n')

print("kFold Result for SGDClassifier Before Parameter Tuning:")
perform_cross_validation(SGDClassifier(), X_Vect, Y)
print("kFold Result for SGDClassifier After Parameter Tuning:")
perform_cross_validation(SGDClassifier(alpha=0.0001, loss='modified_huber', penalty='l2'), X_Vect, Y)

print("kFold Result for LogisticRegression Before Parameter Tuning:")
perform_cross_validation(LogisticRegression(), X_Vect, Y)
print("kFold Result for LogisticRegression After Parameter Tuning:")
perform_cross_validation(LogisticRegression(C=10, penalty='l2'), X_Vect, Y)

print("kFold Result for MultinomialNB Before Parameter Tuning:")
perform_cross_validation(MultinomialNB(), X_Vect, Y)
print("kFold Result for MultinomialNB After Parameter Tuning:")
perform_cross_validation(MultinomialNB(alpha=1.0), X_Vect, Y)

print("kFold Result for BernoulliNB Before Parameter Tuning:")
perform_cross_validation(BernoulliNB(), X_Vect, Y)
print("kFold Result for BernoulliNB After Parameter Tuning:")
perform_cross_validation(BernoulliNB(alpha=0.5, binarize=0.0), X_Vect, Y)

print("kFold Result for RidgeClassifier Before Parameter Tuning:")
perform_cross_validation(RidgeClassifier(), X_Vect, Y)
print("kFold Result for RidgeClassifier After Parameter Tuning:")
perform_cross_validation(RidgeClassifier(alpha=1.0, solver='sag'), X_Vect, Y)

print("kFold Result for LinearSVC Before Parameter Tuning:")
perform_cross_validation(LinearSVC(), X_Vect, Y)
print("kFold Result for LinearSVC After Parameter Tuning:")
perform_cross_validation(LinearSVC(C=2, loss='hinge'), X_Vect, Y)

print("kFold Result for ExtraTree Classifier Before Parameter Tuning:")
perform_cross_validation(ExtraTreeClassifier(), X_Vect, Y)
print("kFold Result for ExtraTree Classifier After Parameter Tuning:")
perform_cross_validation(ExtraTreeClassifier(), X_Vect, Y)





# let's tuning hyperparameter of our selected model for finding best fit
# so I use grid search to look through different parameters
from sklearn.model_selection import GridSearchCV
def ModelTrainTestAfterTuned(model_name, hyperparameters):
    model = model_name(**hyperparameters)  # ** is used to unpack the hyperparameters which is passed using dictionary
    model.fit(x_train, y_train)
    # predict on the training set
    model_train_Prediction = model.predict(x_train)
    # calculate and print the training accuracy
    model_train_Accuracy = accuracy_score(model_train_Prediction, y_train)
    print(f"Training Accuracy: {model_train_Accuracy*100:.2f}%")
    # let's predict the sentiment from the model
    model_test_Predictions = model.predict(x_test)
    model_test_Accuracy = accuracy_score(model_test_Predictions, y_test)
    # print the test accuracy of the model
    print(f"Test Accuracy: {model_test_Accuracy*100:.2f}%")
    # Accuracy difference of the model
    accuracy_Difference = model_train_Accuracy - model_test_Accuracy
    print(f"Accuracy Difference: {accuracy_Difference * 100:.2f}%\n")
    # Plot the bar chart for Training Accuracy, Testing Accuracy, and Accuracy Difference
    labels = ['Training Accuracy', 'Testing Accuracy', 'Accuracy Difference']
    values = [model_train_Accuracy * 100, model_test_Accuracy * 100, accuracy_Difference * 100]
    plt.bar(labels, values, color=['blue', 'orange', 'red'])
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model} Model Accuracy Comparison')
    plt.ylim(0, 100)  # Set the y-axis limits between 0 and 1
    plt.show()
    # print the confusion matrix of the model
    print("\nConfusion Matrix: ")
    print(confusion_matrix(y_test, model_test_Predictions))
    print("\n")
    print("\nClassification Report: ")
    print(classification_report(y_test, model_test_Predictions))
    # plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, model_test_Predictions), annot=True, fmt="d", cmap="Blues", cbar=False)  # cbar show the color bar in right side
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model} Confusion Matrix")
    plt.show()
    return model





# Define the parameter grid for SGDClassifier
parameter_grid_sgd = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1], 'penalty': ['l1', 'l2', 'elasticnet'], 'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']}
# Create the GridSearchCV object for SGDClassifier
grid_sgd = GridSearchCV(SGDClassifier(), parameter_grid_sgd, refit=True, verbose=3)
grid_sgd.fit(x_train, y_train)
best_parameter_sgd = grid_sgd.best_params_
print(f"\nBest Cross Validation Score for SGDClassifier: {grid_sgd.best_score_ * 100:.2f}")
print(f"Best Parameters for SGDClassifier: {best_parameter_sgd}")
print("\n\nSGDClassifier model Result after Parameter Tuning:")
sgd = ModelTrainTestAfterTuned(SGDClassifier, {'alpha': 0.0001, 'loss': 'modified_huber', 'penalty': 'l2'})


# let's define parameter range
parameter_grid_lr = {'C': [0.01, 0.1, 2, 10, 100], 'penalty': ['l1', 'l2']}
grid_lr = GridSearchCV(LogisticRegression(), parameter_grid_lr, refit=True, verbose=3)
grid_lr.fit(x_train, y_train)
best_parameter_lr = grid_lr.best_params_
# the above 3 lines of code find the best parameter now lets print that parameter to fit that into the svc model
print(f"\nBest Cross Validation Score for LogisticRegression: {grid_lr.best_score_*100:.2f}")
print(f"Best Parameter : {best_parameter_lr}")
# let's train the Logistic Regression after tuning the hyperparameter
print("\n\nLogistic Regression model Result after Parameter Tuning:")
lr = ModelTrainTestAfterTuned(LogisticRegression, {'C': 10, 'penalty': 'l2'})


# let's define parameter range
parameter_grid_mnb = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
grid_mnb = GridSearchCV(MultinomialNB(), parameter_grid_mnb, refit=True, verbose=3)
grid_mnb.fit(x_train, y_train)
best_parameter_mnb = grid_mnb.best_params_
# the above 4 lines of code find the best parameter now lets print that parameter to fit that into the svc model
print(f"\nBest Cross Validation Score for MultinomialNB: {grid_mnb.best_score_*100:.2f}")
print(f"Best Parameter : {best_parameter_mnb}")
# let's train the data on MultinomialNB model after tuning the hyperparameter
print("\n\nMultinomialNB model Result after Parameter Tuning:")
mnb = ModelTrainTestAfterTuned(MultinomialNB, {'alpha': 1.0})


# let's define parameter range
parameter_grid_lsvc = {'C': [0.01, 0.1, 2, 10, 100], 'loss': ['hinge', 'squared_hinge']}
grid_lsvc = GridSearchCV(LinearSVC(), parameter_grid_lsvc, refit=True, verbose=3)
grid_lsvc.fit(x_train, y_train)
best_parameter_lsvc = grid_lsvc.best_params_
# the above 4 lines of code find the best parameter now lets print that parameter to fit that into the svc model
print(f"\nBest Cross Validation Score for LinearSVC: {grid_lsvc.best_score_*100:.2f}")
print(f"Best Parameter : {best_parameter_lsvc}")
# now let's fit the new parameter to the LinearSVC model and check the new accuracy
print("\n\nLinear SVC model Result after Parameter Tuning:")
lsvc = ModelTrainTestAfterTuned(LinearSVC, {'C': 2, 'loss': 'hinge'})


# let's define parameter range
parameter_grid_rdg = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0], 'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
grid_rdg = GridSearchCV(RidgeClassifier(), parameter_grid_rdg, refit=True, verbose=3)
grid_rdg.fit(x_train, y_train)
best_parameter_rdg = grid_rdg.best_params_
# the above 4 lines of code find the best parameter now lets print that parameter to fit that into the svc model
print(f"\nBest Cross Validation Score for RidgeClassifier: {grid_rdg.best_score_*100:.2f}")
print(f"Best Parameter : {best_parameter_rdg}")
# now let's fit the new parameter to the LinearSVC model and check the new accuracy
print("\n\nRidgeClassifier model Result after Parameter Tuning:")
rdg = ModelTrainTestAfterTuned(RidgeClassifier, {'alpha': 1.0, 'solver': 'sag'})


# let's define parameter range
parameter_grid_bnb = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0], 'binarize': [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]}
grid_bnb = GridSearchCV(BernoulliNB(), parameter_grid_bnb, refit=True, verbose=3)
grid_bnb.fit(x_train, y_train)
best_parameter_bnb= grid_bnb.best_params_
# the above 4 lines of code find the best parameter now lets print that parameter to fit that into the svc model
print(f"\nBest Cross Validation Score for BernoulliNB: {grid_bnb.best_score_*100:.2f}")
print(f"Best Parameter : {best_parameter_bnb}")
# now let's fit the new parameter to the LinearSVC model and check the new accuracy
print("\n\nBernoulliNB model Result after Parameter Tuning:")
bnb = ModelTrainTestAfterTuned(BernoulliNB, {'alpha': 0.5, 'binarize': 0.0})



# Define the parameter grid for ExtraTreeClassifier
parameter_grid_extra_trees = {'n_estimators': [50, 100, 200], 'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20]}
grid_extra_trees = GridSearchCV(ExtraTreeClassifier(), parameter_grid_extra_trees, refit=True, verbose=3)
grid_extra_trees.fit(x_train, y_train)
best_parameter_extra_trees = grid_extra_trees.best_params_
print(f"\nBest Cross Validation Score for ExtraTreesClassifier: {grid_extra_trees.best_score_ * 100:.2f}")
print(f"Best Parameters: {best_parameter_extra_trees}")
# print(f"\n\nExtraTree model Result after Parameter Tuning:")
# trees = ModelTrainTestAfterTuned(ExtraTreeClassifier, {best_parameter_extra_trees})
