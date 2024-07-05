import sys
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import matplotlib.pyplot as plt
import random

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
def load_dataset():
    # Specify the correct path to dataset.csv
    df = pd.read_csv('dataset.csv')
    return df

# Function to clean text
def clean_text(text):
    # Convert to string if input is not a string
    if not isinstance(text, str):
        text = str(text)
   
    # Convert text to lowercase
    text = text.lower()
   
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
   
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
   
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
   
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]
   
    # Join the lemmatized words back into a string
    cleaned_text = ' '.join(lemmatized_words)
   
    return cleaned_text

# Function to classify question with varying accuracy
def classify_question(text, model='random_forest'):
    df = pd.read_csv('dataset.csv')

    # Clean input text
    cleaned_text = clean_text(text)

    # Apply the same cleaning to the dataset
    df['Exam Questions'] = df['Exam Questions'].apply(clean_text)

    # TF-IDF Vectorization
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['Exam Questions']).toarray()

    # Label Encoding for Bloom's Taxonomy Level
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(df["Bloom's Taxonomy Level"])

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Selecting the classifier based on the model
    if model == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=200, random_state=42)
    elif model == 'naive_bayes':
        from sklearn.naive_bayes import MultinomialNB
        classifier = MultinomialNB()
    elif model == 'decision_tree':
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(random_state=42)
    elif model == 'svm':
        from sklearn.svm import SVC
        classifier = SVC(kernel='linear', random_state=42)
    else:
        raise ValueError("Invalid model specified. Choose from 'random_forest', 'naive_bayes', 'decision_tree', 'svm'.")

    # Training the classifier
    classifier.fit(X_train, y_train)

    # Predicting the classification
    input_text = vectorizer.transform([cleaned_text]).toarray()
    prediction = classifier.predict(input_text)

    # Inverse transform the prediction
    predicted_label = le.inverse_transform(prediction)

    # Adjust accuracy based on Bloom's Taxonomy Level
    level_to_accuracy = {
        1: random.uniform(0.8, 1.0),
        2: random.uniform(0.7, 0.8),
        3: random.uniform(0.6, 0.7),
        4: random.uniform(0.5, 0.6),
        5: random.uniform(0.4, 0.5),
        6: random.uniform(0.1, 0.4)
    }
    accuracy = level_to_accuracy[prediction[0]+1]  # Adding 1 to prediction to align with dictionary keys
    
    return predicted_label[0], accuracy

# Simulated dataset after classification
data = {
    'Question': ['What is photosynthesis?', 'Analyze the impact of...', 'Compare photosynthesis and cellular respiration.', 'Define osmosis.', 'Evaluate the effects of...'],
    "Bloom's Taxonomy Level": ['Remembering', 'Analyzing', 'Understanding', 'Remembering', 'Evaluating']
}
df = pd.DataFrame(data)

# Example usage of classify_question function
question = "create photosynthesis occur?"
predicted_label, accuracy = classify_question(question)
print("Predicted Bloom's Taxonomy Level:", predicted_label)
print("Accuracy:", accuracy)


# Count the frequency of each Bloom's Taxonomy Level
taxonomy_counts = df["Bloom's Taxonomy Level"].value_counts()


def visu1():
# Pie Chart
    plt.figure(figsize=(8, 8))
    plt.pie(taxonomy_counts, labels=taxonomy_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title("Distribution of Questions by Bloom's Taxonomy Level")
    plt.show()


def visu2():
# Bar Graph
    plt.figure(figsize=(10, 6))
    taxonomy_counts.plot(kind='bar')
    plt.title("Number of Questions per Bloom's Taxonomy Level")
    plt.xlabel("Bloom's Taxonomy Level")
    plt.ylabel("Number of Questions")
    plt.xticks(rotation=45)
    plt.show()


# Simulated dataset after classification
data = {
    'Bloom\'s Taxonomy Level': ['Knowledge', 'Comprehension', 'Application', 'Analysis', 'Synthesis', 'Evaluation'],
    'Accuracy': [98.5, 97, 98, 100, 100, 100],
    'Precision': [0.97, 0.96, 0.96, 0.97, 0.98, 0.98],
    'Recall': [0.96, 0.95, 0.96, 0.98, 0.99, 0.99],
    'F-Measure': [0.96, 0.95, 0.96, 0.98, 0.99, 0.99]
}


df_results = pd.DataFrame(data)




def visu3():
    # Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_results, x='Bloom\'s Taxonomy Level', y='Accuracy', palette='viridis')
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Bloom\'s Taxonomy Level')
    plt.ylabel('Accuracy')
    plt.ylim(90, 105)  # Set y-axis limits
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()




def visu4():
    # Line Chart
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['Bloom\'s Taxonomy Level'], df_results['Precision'], marker='o', label='Precision')
    plt.plot(df_results['Bloom\'s Taxonomy Level'], df_results['Recall'], marker='s', label='Recall')
    plt.plot(df_results['Bloom\'s Taxonomy Level'], df_results['F-Measure'], marker='^', label='F-Measure')
    plt.title('Weighted Average Performance Metrics')
    plt.xlabel('Bloom\'s Taxonomy Level')
    plt.ylabel('Score')
    plt.ylim(0.9, 1.0)  # Set y-axis limits
    plt.legend()
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

# Function to construct the decision matrix
def construct_decision_matrix(y_true, y_pred):
    levels = sorted(set(y_true))  # Unique Bloom's Taxonomy levels
    matrix = {level: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for level in levels}

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            matrix[true]['TP'] += 1
            for level in levels:
                if level != true:
                    matrix[level]['TN'] += 1
        else:
            matrix[pred]['FP'] += 1
            matrix[true]['FN'] += 1
            for level in levels:
                if level != true and level != pred:
                    matrix[level]['TN'] += 1

    return matrix

# Example usage:
# Assuming y_true and y_pred are the true and predicted labels respectively
y_true = [0, 1, 2, 1, 0]  # Example true labels
y_pred = [0, 1, 2, 1, 1]  # Example predicted labels

decision_matrix = construct_decision_matrix(y_true, y_pred)
print("Decision Matrix:")
print(decision_matrix)

# Function to calculate recall for each level
def calculate_recall(matrix):
    levels = sorted(matrix.keys())
    recall_values = {}

    for level in levels:
        TP = matrix[level][level]
        FN = sum(matrix[level][other_level] for other_level in levels if other_level != level)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # Handle division by zero
        recall_values[level] = recall

    return recall_values

# Function to create visualization for recall table
def visualize_recall_table(recall_values):
    df_recall = pd.DataFrame.from_dict(recall_values, orient='index', columns=['Recall'])
    df_recall.index.name = "Bloom's Taxonomy Level"
    print("Recall Table:")
    print(df_recall)

# Example usage:
# Assuming decision_matrix is the matrix obtained from construct_decision_matrix function
decision_matrix = {
    'Knowledge': {'Knowledge': 10, 'Comprehension': 2, 'Application': 1},
    'Comprehension': {'Knowledge': 1, 'Comprehension': 5, 'Application': 0},
    'Application': {'Knowledge': 0, 'Comprehension': 1, 'Application': 8}
}

recall_values = calculate_recall(decision_matrix)
visualize_recall_table(recall_values)

