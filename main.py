from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import requests
from typing import List, Tuple


app = FastAPI()
response = requests.get('https://trendify-ui65.onrender.com/products')
data = response.json() 
df = pd.DataFrame(data)  #converting the JSON data into a pandas DataFrame.
# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Assuming 'data' is your product data
# data = [
#     {'_id': '65aa9fea11b750a2db1ad9ab', 'price': '3,990.00', 'details': 'Sneakers with a combination of materials, piec...', 'category': 'shoes', 'productName': 'DENIM SNEAKERS', 'productImage': 'https://static.zara.net/photos///2023/I/1/2/p/'},
#     {'_id': '65aa9fea11b750a2db1ad9ac', 'price': '2200', 'details': 'Monochrome trainers. Seven-eyelet facing. The ...', 'category': 'shoes', 'productName': 'CHUNKY TRAINERS', 'productImage': 'https://static.zara.net/photos///2023/V/1/2/p/'},
#     {'_id': '65aa9fea11b750a2db1ad9ad', 'price': '3,990.00', 'details': 'Sneakers. Plain upper. Seven-eyelet facing. Ch...', 'category': 'shoes', 'productName': 'CHUNKY TRAINERS', 'productImage': 'https://static.zara.net/photos///2023/I/1/2/p/'},
#     {'_id': '65aa9fea11b750a2db1ad9ae', 'price': '2,990.00', 'details': 'Trainers. Combination of materials, pieces and...', 'category': 'shoes', 'productName': 'MULTIPIECE TRAINERS', 'productImage': 'https://static.zara.net/photos///2023/I/1/2/p/'},
#     # Add more product entries as needed
# ]

# Extract details, categories, and product names
details = [item['details'] for item in data]
categories = [item['category'] for item in data]
product_names = [item['productName'] for item in data]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(details)

# Compute cosine similarity
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

def products_by_category(category_input: str) -> List[Tuple[str, str, str, str]]:
    # Find the indices of products in the data based on the category
    category_indices = [i for i, cat in enumerate(categories) if cat.lower() == category_input.lower()]

    if category_indices:
        # Get products with the specified category
        matching_products = [(data[i]['productName'], data[i]['price'], data[i]['productImage'], data[i]['details']) for i in category_indices]

        # Return the products with the same category
        return matching_products
    else:
        return [("No products found for the given category.", "", "", "")]

@app.get("/products/{category}")
def products_in_category(category: str):
    products = products_by_category(category)
    return products

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
