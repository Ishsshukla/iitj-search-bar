import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from fastapi import FastAPI
import requests

app = FastAPI()
# origins = [
#     "http://127.0.0.1:8000",
#     "http://localhost:5173",
#     "https://workshala-navy.vercel.app",
#     # "https://intrship.onrender.com",
#     "http://localhost:5000",
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
response = requests.get('https://trendify-ui65.onrender.com/products')
data = response.json()  
df = pd.DataFrame(data)
df.isnull().sum()
df.head()
df.loc[df.duplicated(subset='productName')] 
df_copy1 =df.drop(columns=[ 'wishingUsers','_id']).copy()

# Extract details and product names
details = [item['details'] for item in data]
product_names = [item['productName'] for item in data]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(details)

# Compute cosine similarity
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend_products(user_input):
    # Find the index of the product in the data
    index = product_names.index(user_input)

    # Get the cosine similarity scores for the given product
    sim_scores = list(enumerate(cosine_similarities[index]))

    # Sort the products based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 3 similar products
    top_products = sim_scores[1:4]

    # Get the indices of the top products
    top_indices = [i[0] for i in top_products]

    # Return the recommended products
    return [(data[i]['productName'], data[i]['price'], data[i]['productImage'],data[i]['details']) for i in top_indices]
user_input = "DENIM SNEAKERS"
recommendations = recommend_products(user_input)

# Display recommendations
# Display recommendations
for recommendation in recommendations:
    print(f"Product Name: {recommendation[0]}\nPrice: {recommendation[1]}\ndetails: {recommendation[2]}\nProduct Image: {recommendation[3]}\n")
    
    
@app.get("/recommendation/{product_name}")
def recommendation_func(product_name: str):
    recommendations = recommend_products(product_name)
    return recommendations