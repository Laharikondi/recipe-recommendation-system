from flask import Flask, request, render_template_string
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from gensim.models import Word2Vec

# Initialize Flask app
app = Flask(__name__)

# Load the dataset
df = pd.read_csv('cuisines.csv')

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

def extract_ingredients(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if token.pos_ in ["NOUN", "ADJ"]]

# Train Word2Vec model on ingredients
ingredient_sentences = [row.split(', ') for row in df['ingredients']]
word2vec_model = Word2Vec(sentences=ingredient_sentences, vector_size=100, window=5, min_count=1, workers=4)

def get_similar_ingredients(ingredients):
    similar_words = set()
    for ing in ingredients:
        if ing in word2vec_model.wv:
            similar_words.update([word for word, _ in word2vec_model.wv.most_similar(ing, topn=3)])
    return list(similar_words) + ingredients

# Create a TF-IDF vectorizer for ingredients
vectorizer = TfidfVectorizer(stop_words='english')
ingredients_vector = vectorizer.fit_transform(df['ingredients'])

# Initialize k-NN model
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(ingredients_vector)

def get_recommendations(input_ingredients):
    extracted_ingredients = extract_ingredients(input_ingredients)
    expanded_ingredients = get_similar_ingredients(extracted_ingredients)
    input_text = ', '.join(expanded_ingredients)
    input_vector = vectorizer.transform([input_text])
    distances, indices = knn.kneighbors(input_vector)
    return df.iloc[indices[0]][['name', 'image_url', 'description', 'prep_time', 'ingredients', 'instructions']]

# HTML Template
html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recipe Recommendation System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }
        .container {
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            color: #333;
        }
        form {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 20px;
        }
        input[type="text"] {
            padding: 10px;
            width: 60%;
            font-size: 16px;
        }
        input[type="submit"] {
            padding: 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            font-size: 16px;
        }
        .recommendations {
            margin-top: 20px;
        }
        .recipe {
            padding: 15px;
            border-bottom: 1px solid #ddd;
        }
        .recipe h3 {
            margin: 0 0 10px;
            color: #333;
            font-size: 20px;
            font-weight: bold;
        }
        img {
            width: 200px;
            display: block;
            margin: 10px auto;
        }
        ul {
            padding-left: 20px;
        }
        .instructions {
            font-family: 'Courier New', monospace;
        }
        .no-recommendations {
            color: #999;
            text-align: center;
        }
    </style>
</head>
<body>

<div class="container">
    <h1>Recipe Recommendation System</h1>

    <form action="/" method="POST">
        <input type="text" name="ingredients" placeholder="Enter ingredients..." value="{{ input_ingredients or '' }}">
        <input type="submit" value="Get Recommendations">
    </form>

    {% if recommendations %}
        <div class="recommendations">
            {% for rec in recommendations %}
                <div class="recipe">
                    <h3>{{ rec['name'] }}</h3>
                    <img src="{{ rec['image_url'] }}" alt="{{ rec['name'] }}">
                    <p><strong>Description:</strong> {{ rec['description'] }}</p>
                    <p><strong>Prep Time:</strong> {{ rec['prep_time'] }}</p>
                    <p><strong>Ingredients:</strong></p>
                    <ul>
                        {% for ingredient in rec['ingredients'].split(', ') %}
                            <li>{{ ingredient }}</li>
                        {% endfor %}
                    </ul>
                    <p><strong>Instructions:</strong></p>
                    <ul class="instructions">
                        {% for step in rec['instructions'].split('. ') %}
                            <li>{{ step }}</li>
                        {% endfor %}
                    </ul>
                </div>
            {% endfor %}
        </div>
    {% elif input_ingredients %}
        <p class="no-recommendations">No recommendations found for the given ingredients. Try again with different ingredients.</p>
    {% endif %}
</div>

</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the input ingredients from the form
        input_ingredients = request.form['ingredients']
        
        # Get the recipe recommendations
        recommendations = get_recommendations(input_ingredients)
        
        # Pass recommendations to the HTML template
        return render_template_string(html_template, recommendations=recommendations.to_dict('records'), input_ingredients=input_ingredients)
    
    # Default route behavior, display the form
    return render_template_string(html_template, recommendations=None)

if __name__ == '__main__':
    app.run(debug=True)   