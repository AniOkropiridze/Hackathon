import pyodbc
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import google.generativeai as genai
import logging
from typing import List
from dataclasses import dataclass
from datetime import datetime
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    dish_id: int
    restaurant_name: str
    product_name: str
    score: float
    text: str

@dataclass
class ConversationTurn:
    timestamp: datetime
    user_query: str
    assistant_response: str

class ConversationMemory:
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.conversation_history: List[ConversationTurn] = []
        
    def add_turn(self, user_query: str, assistant_response: str):
        """Add a conversation turn to memory"""
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_query=user_query,
            assistant_response=assistant_response
        )
        
        self.conversation_history.append(turn)
        
        # Keep only recent turns
        if len(self.conversation_history) > self.max_turns:
            self.conversation_history.pop(0)
    
    def get_conversation_context(self) -> str:
        """Get formatted conversation context"""
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for i, turn in enumerate(self.conversation_history, 1):
            context_parts.append(f"წინა კითხვა {i}: {turn.user_query}")
            context_parts.append(f"წინა პასუხი {i}: {turn.assistant_response}")
        
        return "\n".join(context_parts)

class GeorgianRAGSystem:
    def __init__(self):
        # Configuration
        self.api_key = "Api_key"
        self.server = "Server_name"
        self.database = "Full_data"
        self.embedding_dim = 768
        
        # Initialize components
        self.df = None
        self.documents = []
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.faiss_index = None
        
        # Add memory component
        self.memory = ConversationMemory(max_turns=10)
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        logger.info("✅ Gemini API configured")
        
    def connect_database(self):
        """Connect to Georgian database"""
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            f"Trusted_Connection=yes;"
            f"Connection Timeout=30;"
        )
        
        conn = pyodbc.connect(conn_str)
        logger.info(f"✅ Connected to {self.database}")
        return conn
    
    def load_data(self):
        """Load Georgian restaurant data"""
        conn = self.connect_database()
        
        query = """
        SELECT 
            cd.DishID,
            r.Name AS RestaurantName,
            r.Location,
            p.Name AS ProductName,
            cd.Portion,
            cd.Price,
            STRING_AGG(i.Name, ', ') AS Ingredients,
            STRING_AGG(rev.Text, ' || ') AS Reviews
        FROM ConcreteDish cd
        JOIN Restaurant r ON cd.RestaurantID = r.RestaurantID
        JOIN Product p ON cd.ProductID = p.ProductID
        LEFT JOIN DishIngredient di ON cd.DishID = di.DishID
        LEFT JOIN Ingredient i ON di.IngredientID = i.IngredientID
        LEFT JOIN Review rev ON cd.DishID = rev.DishID
        GROUP BY cd.DishID, r.Name, r.Location, p.Name, cd.Portion, cd.Price
        ORDER BY cd.DishID
        """
        
        self.df = pd.read_sql(query, conn)
        conn.close()
        
        # Prepare text documents
        self.df['Ingredients'] = self.df['Ingredients'].fillna('')
        self.df['Reviews'] = self.df['Reviews'].fillna('')
        self.df['text'] = self.df.apply(
            lambda row: f"{row['ProductName']} ({row['Portion']}): {row['Ingredients']}. Reviews: {row['Reviews']}", 
            axis=1
        )
        
        self.documents = self.df['text'].tolist()
        logger.info(f"✅ Loaded {len(self.documents)} Georgian dishes")
    
    def setup_search(self):
        """Setup hybrid search components"""
        # BM25
        tokenized_corpus = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            token_pattern=r'(?u)\b\w+\b'
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)
        
        # FAISS embeddings
        embeddings = []
        for i, doc in enumerate(self.documents):
            if i % 5 == 0:
                logger.info(f"Generating embeddings {i}/{len(self.documents)}")
            
            result = genai.embed_content(
                model="models/embedding-001",
                content=doc,
                task_type="retrieval_document"
            )
            embeddings.append(result["embedding"])
        
        embeddings = np.array(embeddings).astype("float32")
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        self.faiss_index.add(embeddings)
        
        logger.info("✅ Hybrid search ready")
    
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Hybrid search for Georgian queries"""
        # Semantic search
        query_embedding = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )["embedding"]
        query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)
        
        distances, _ = self.faiss_index.search(query_embedding, len(self.documents))
        semantic_scores = 1 / (1 + distances[0])
        
        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # TF-IDF search
        tfidf_query = self.tfidf_vectorizer.transform([query])
        tfidf_scores = cosine_similarity(self.tfidf_matrix, tfidf_query).flatten()
        
        # Normalize and combine scores
        def normalize(scores):
            if scores.max() - scores.min() == 0:
                return scores
            return (scores - scores.min()) / (scores.max() - scores.min())
        
        bm25_scores = normalize(bm25_scores)
        tfidf_scores = normalize(tfidf_scores)
        semantic_scores = normalize(semantic_scores)
        
        hybrid_scores = 0.4 * bm25_scores + 0.3 * tfidf_scores + 0.3 * semantic_scores
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            result = SearchResult(
                dish_id=row['DishID'],
                restaurant_name=row['RestaurantName'],
                product_name=row['ProductName'],
                score=hybrid_scores[idx],
                text=self.documents[idx]
            )
            results.append(result)
        
        return results
    
    def generate_answer(self, question: str, results: List[SearchResult]) -> str:
        """Generate Georgian answer using Gemini"""
        context = "\n".join([
            f"{i}. {r.restaurant_name}: {r.text} (ქულა: {r.score:.2f})"
            for i, r in enumerate(results, 1)
        ])
        
        # Add conversation history to prompt
        conversation_context = self.memory.get_conversation_context()
        
        prompt = f"""შეკითხვა: {question}

რელევანტური ინფორმაცია:
{context}

{f"წინა საუბრის კონტექსტი:{chr(10)}{conversation_context}{chr(10)}" if conversation_context else ""}

გამეცი პასუხი ქართულად მხოლოდ ზემოთ მოცემული ინფორმაციის საფუძველზე:"""

        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "429" in str(e):
                return f"🔍 ძებნის შედეგები:\n\n" + "\n".join([
                    f"{i}. {r.restaurant_name} - {r.product_name} (ქულა: {r.score:.2f})"
                    for i, r in enumerate(results, 1)
                ])
            return f"შეცდომა: {e}"
    
    def query(self, question: str) -> str:
        """Main query function"""
        results = self.search(question)
        response = self.generate_answer(question, results)
        
        # Store in memory
        self.memory.add_turn(question, response)
        
        return response

def main():
    """Main function"""
    # Fix Windows Georgian text display
    import sys, codecs
    if sys.platform.startswith('win'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    
    # Initialize system
    rag = GeorgianRAGSystem()
    rag.load_data()
    rag.setup_search()
    
    logger.info("🎯 Georgian RAG ready!")
    
    # Interactive mode
    print("\nGeorgian Restaurant Search")
    print("შეიყვანეთ შეკითხვა ('exit' გასვლა):")
    
    while True:
        try:
            query = input("\n🔍 შეკითხვა: ")
            if query.lower() in ['exit', 'quit', 'გამოსვლა']:
                break
            
            answer = rag.query(query)
            print(f"\n📝 პასუხი: {answer}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"შეცდომა: {e}")

if __name__ == "__main__":
    main()
