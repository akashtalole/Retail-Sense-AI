import gradio as gr
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.memory import ConversationBufferMemory
from opensearchpy import RequestsHttpConnection, AWSV4SignerAuth
import boto3
import os
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

# ==================== Environment Configuration ====================
LLM_URL = os.environ.get("LLM_URL", "http://localhost:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "nvidia/llama-3.1-nemotron-nano-8b-v1")
EMBEDDINGS_URL = os.environ.get("EMBEDDINGS_URL", "http://localhost:8001/v1")
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "nvidia/llama-3.2-nv-embedqa-1b-v2")
AWS_DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION")
OPENSEARCH_COLLECTION_ID = os.environ.get("OPENSEARCH_COLLECTION_ID")
OPENSEARCH_INDEX = os.environ.get("OPENSEARCH_INDEX")

host = "https://" + str(OPENSEARCH_COLLECTION_ID) + "." + str(AWS_DEFAULT_REGION) + ".aoss.amazonaws.com"

# ==================== Data Models ====================

class QueryIntent(Enum):
    """User query intent classification"""
    SEARCH = "search"
    COMPARE = "compare"
    RECOMMEND = "recommend"
    QUESTION = "question"
    GENERAL = "general"

@dataclass
class Product:
    """Product data model"""
    product_id: str
    name: str
    description: str
    price: float
    category: str
    brand: str = "Unknown"
    rating: float = 0.0
    specifications: Dict[str, Any] = None
    image_url: Optional[str] = None

    def __post_init__(self):
        if self.specifications is None:
            self.specifications = {}

    def to_dict(self):
        return asdict(self)

    def to_display_text(self):
        """Format product for display"""
        specs_text = ""
        if self.specifications:
            specs_text = "\n- Specifications: " + ", ".join([f"{k}: {v}" for k, v in self.specifications.items()])

        return f"""**{self.name}**
- Brand: {self.brand}
- Category: {self.category}
- Price: ${self.price:.2f}
- Rating: {self.rating}/5.0
{specs_text}
- Description: {self.description}
"""

# ==================== Initialization ====================

embedder = NVIDIAEmbeddings(base_url=EMBEDDINGS_URL, model=EMBEDDINGS_MODEL)

text_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=100,
)

llm = ChatNVIDIA(
    base_url=LLM_URL,
    model=LLM_MODEL,
)

# ==================== Agent System ====================

class QueryUnderstandingAgent:
    """Agent responsible for understanding user queries and extracting intent"""

    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query understanding agent for an e-commerce platform.
Analyze the user's query and determine:
1. Intent: search, compare, recommend, question, or general
2. Key entities: product types, brands, price ranges, features
3. Constraints: budget, specifications, preferences

Respond in JSON format:
{{
    "intent": "search|compare|recommend|question|general",
    "reasoning": "explanation of why you chose this intent",
    "entities": {{"product_type": "...", "brand": "...", "price_max": ..., "features": [...]}},
    "query_simplified": "simplified search query"
}}"""),
            ("user", "{query}")
        ])

    def analyze(self, query: str) -> Dict[str, Any]:
        """Analyze user query and extract intent"""
        try:
            chain = self.prompt | self.llm | StrOutputParser()
            response = chain.invoke({"query": query})

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                # Fallback to simple intent detection
                query_lower = query.lower()
                if any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
                    intent = "compare"
                elif any(word in query_lower for word in ["recommend", "suggest", "best"]):
                    intent = "recommend"
                elif any(word in query_lower for word in ["?", "what", "how", "why", "when"]):
                    intent = "question"
                else:
                    intent = "search"

                return {
                    "intent": intent,
                    "reasoning": "Fallback classification based on keywords",
                    "entities": {},
                    "query_simplified": query
                }
        except Exception as e:
            print(f"Error in query understanding: {e}")
            return {
                "intent": "search",
                "reasoning": f"Error occurred: {str(e)}",
                "entities": {},
                "query_simplified": query
            }

class ProductSearchAgent:
    """Agent responsible for searching products using semantic search"""

    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm

    def search(self, query: str, filters: Dict[str, Any] = None, k: int = 5) -> List[Dict[str, Any]]:
        """Search for products using semantic search"""
        try:
            # Perform semantic search
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
            documents = retriever.get_relevant_documents(query)

            # Extract product information from documents
            products = []
            for i, doc in enumerate(documents):
                # Try to parse metadata as product info
                metadata = doc.metadata
                product_info = {
                    "product_id": metadata.get("product_id", f"prod_{i}"),
                    "name": metadata.get("name", "Unknown Product"),
                    "description": doc.page_content[:200],
                    "price": metadata.get("price", 0.0),
                    "category": metadata.get("category", "General"),
                    "brand": metadata.get("brand", "Unknown"),
                    "rating": metadata.get("rating", 0.0),
                    "score": metadata.get("score", 0.0)
                }
                products.append(product_info)

            return products
        except Exception as e:
            print(f"Error in product search: {e}")
            return []

class ComparisonAgent:
    """Agent responsible for comparing multiple products"""

    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a product comparison expert for an e-commerce platform.
Compare the given products across key dimensions:
- Price and value for money
- Features and specifications
- Quality and ratings
- Best use cases

Provide a clear, structured comparison with reasoning for each point.
Be objective and highlight both strengths and weaknesses."""),
            ("user", "Compare these products:\n\n{products}\n\nUser's question: {query}")
        ])

    def compare(self, products: List[Dict[str, Any]], query: str) -> str:
        """Compare multiple products and provide analysis"""
        try:
            products_text = "\n\n".join([
                f"Product {i+1}: {p['name']}\n"
                f"- Price: ${p['price']:.2f}\n"
                f"- Category: {p['category']}\n"
                f"- Brand: {p['brand']}\n"
                f"- Rating: {p['rating']}/5.0\n"
                f"- Description: {p['description']}"
                for i, p in enumerate(products)
            ])

            chain = self.prompt | self.llm | StrOutputParser()
            response = chain.invoke({"products": products_text, "query": query})
            return response
        except Exception as e:
            print(f"Error in comparison: {e}")
            return f"Unable to compare products: {str(e)}"

class RecommendationAgent:
    """Agent responsible for providing personalized recommendations"""

    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a recommendation expert for an e-commerce platform.
Based on the user's query and available products, provide personalized recommendations.

For each recommendation:
1. Explain WHY it's a good match
2. Highlight key features that align with user needs
3. Mention any trade-offs or considerations
4. Suggest the best use case

Be transparent about your reasoning and provide value beyond just listing products."""),
            ("user", "User query: {query}\n\nAvailable products:\n{products}\n\nProvide recommendations with detailed reasoning.")
        ])

    def recommend(self, query: str, products: List[Dict[str, Any]], context: str = "") -> str:
        """Generate personalized recommendations"""
        try:
            products_text = "\n\n".join([
                f"{p['name']} (${p['price']:.2f})\n"
                f"Category: {p['category']}, Brand: {p['brand']}, Rating: {p['rating']}/5.0\n"
                f"Description: {p['description']}"
                for p in products
            ])

            chain = self.prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "query": query,
                "products": products_text
            })
            return response
        except Exception as e:
            print(f"Error in recommendation: {e}")
            return f"Unable to generate recommendations: {str(e)}"

class AgentOrchestrator:
    """Central orchestrator that coordinates all agents"""

    def __init__(self, vectorstore, llm):
        self.query_agent = QueryUnderstandingAgent(llm)
        self.search_agent = ProductSearchAgent(vectorstore, llm)
        self.comparison_agent = ComparisonAgent(llm)
        self.recommendation_agent = RecommendationAgent(vectorstore, llm)
        self.llm = llm
        self.conversation_history = []

    def process_query(self, query: str) -> str:
        """Main orchestration logic using ReAct pattern"""
        try:
            # Step 1: Understand the query
            understanding = self.query_agent.analyze(query)
            intent = understanding.get("intent", "search")
            reasoning = understanding.get("reasoning", "")
            simplified_query = understanding.get("query_simplified", query)

            response_parts = []
            response_parts.append(f"🤔 **Understanding**: {reasoning}\n")

            # Step 2: Route to appropriate agent based on intent
            if intent == "compare":
                # Search for products to compare
                products = self.search_agent.search(simplified_query, k=5)
                if products:
                    response_parts.append(f"🔍 **Found {len(products)} products for comparison**\n")
                    comparison = self.comparison_agent.compare(products[:3], query)
                    response_parts.append(f"\n📊 **Comparison Analysis**:\n{comparison}\n")

                    # Display products
                    response_parts.append("\n**Products:**\n")
                    for p in products[:3]:
                        response_parts.append(f"- **{p['name']}** (${p['price']:.2f}) - {p['brand']}\n")
                else:
                    response_parts.append("❌ No products found for comparison.\n")

            elif intent == "recommend":
                # Search and recommend
                products = self.search_agent.search(simplified_query, k=5)
                if products:
                    response_parts.append(f"🔍 **Found {len(products)} potential products**\n")
                    recommendations = self.recommendation_agent.recommend(query, products)
                    response_parts.append(f"\n💡 **Recommendations**:\n{recommendations}\n")

                    # Display top products
                    response_parts.append("\n**Top Products:**\n")
                    for p in products[:3]:
                        response_parts.append(f"- **{p['name']}** (${p['price']:.2f}) - Rating: {p['rating']}/5.0\n")
                else:
                    response_parts.append("❌ No products found to recommend.\n")

            elif intent == "search":
                # Perform semantic search
                products = self.search_agent.search(simplified_query, k=5)
                if products:
                    response_parts.append(f"🔍 **Found {len(products)} matching products**\n\n")

                    # Generate natural response with product details
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", """You are a helpful shopping assistant. Present the search results in a natural, conversational way.
Highlight key features that match the user's query. Be concise but informative."""),
                        ("user", "User searched for: {query}\n\nResults:\n{products}")
                    ])

                    products_text = "\n".join([
                        f"{i+1}. {p['name']} - ${p['price']:.2f} ({p['brand']})\n   {p['description'][:100]}..."
                        for i, p in enumerate(products)
                    ])

                    chain = prompt | self.llm | StrOutputParser()
                    natural_response = chain.invoke({"query": query, "products": products_text})
                    response_parts.append(natural_response)
                else:
                    response_parts.append("❌ No products found matching your search.\n")

            else:  # question or general
                # Use RAG to answer questions
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a knowledgeable shopping assistant. Answer questions about products, shopping, and e-commerce.
If you don't have enough information, say so honestly. Base your answers on the available product context."""),
                    ("user", "Question: {query}\n\nContext: {context}")
                ])

                # Get relevant context
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
                docs = retriever.get_relevant_documents(query)
                context = "\n\n".join([doc.page_content for doc in docs])

                chain = prompt | self.llm | StrOutputParser()
                answer = chain.invoke({"query": query, "context": context if context else "No specific product information available."})
                response_parts.append(f"\n💬 **Answer**:\n{answer}\n")

            # Save to conversation history
            self.conversation_history.append({
                "query": query,
                "intent": intent,
                "response": "".join(response_parts)
            })

            return "".join(response_parts)

        except Exception as e:
            print(f"Error in orchestration: {e}")
            return f"❌ I encountered an error processing your request: {str(e)}\n\nPlease try rephrasing your question."

# ==================== Vector Store Setup ====================

def create_vectorstore():
    """Create and initialize OpenSearch vector store"""
    credentials = boto3.Session().get_credentials()
    awsauth = AWSV4SignerAuth(credentials, AWS_DEFAULT_REGION, "aoss")

    vectorstore = OpenSearchVectorSearch(
        host,
        OPENSEARCH_INDEX,
        embedder,
        http_auth=awsauth,
        timeout=300,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        vector_field="vector_field"
    )

    index_mapping = {
        "settings": {"index": {"knn": True, "knn.algo_param.ef_search": 512}},
        "mappings": {
            "properties": {
                "vector_field": {
                    "type": "knn_vector",
                    "dimension": 2048,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "nmslib",
                        "parameters": {"ef_construction": 512, "m": 16},
                    }
                },
                "name": {"type": "text"},
                "description": {"type": "text"},
                "price": {"type": "float"},
                "category": {"type": "keyword"},
                "brand": {"type": "keyword"},
                "rating": {"type": "float"},
                "product_id": {"type": "keyword"}
            }
        }
    }

    if vectorstore.index_exists(OPENSEARCH_INDEX):
        vectorstore.delete_index(OPENSEARCH_INDEX)
    vectorstore.client.indices.create(index=OPENSEARCH_INDEX, body=index_mapping)

    return vectorstore

# ==================== Document Processing ====================

def upload_documents(files):
    """Upload and process PDF documents as product catalogs"""
    if not files:
        return "No files uploaded"

    try:
        for file_path in files:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            split_documents = text_splitter.split_documents(documents)

            # Enhance documents with product metadata
            for doc in split_documents:
                # Try to extract product info from content
                content = doc.page_content
                doc.metadata.update({
                    "product_id": doc.metadata.get("product_id", f"prod_{hash(content) % 10000}"),
                    "category": doc.metadata.get("category", "General"),
                    "price": doc.metadata.get("price", 0.0),
                    "brand": doc.metadata.get("brand", "Unknown"),
                    "rating": doc.metadata.get("rating", 0.0)
                })

            vectorstore.add_documents(split_documents)

        return f"✅ Successfully uploaded {len(files)} document(s) to the product catalog!"
    except Exception as e:
        return f"❌ Error uploading documents: {str(e)}"

def upload_products_json(json_file):
    """Upload products from JSON file"""
    try:
        with open(json_file.name, 'r') as f:
            products_data = json.load(f)

        documents = []
        for product in products_data:
            # Create document from product
            content = f"{product.get('name', 'Unknown Product')}\n"
            content += f"Category: {product.get('category', 'General')}\n"
            content += f"Brand: {product.get('brand', 'Unknown')}\n"
            content += f"Price: ${product.get('price', 0.0)}\n"
            content += f"Description: {product.get('description', '')}\n"

            if 'specifications' in product:
                content += "Specifications:\n"
                for key, value in product['specifications'].items():
                    content += f"- {key}: {value}\n"

            from langchain.schema import Document
            doc = Document(
                page_content=content,
                metadata={
                    "product_id": product.get('product_id', f"prod_{hash(content) % 10000}"),
                    "name": product.get('name', 'Unknown'),
                    "category": product.get('category', 'General'),
                    "brand": product.get('brand', 'Unknown'),
                    "price": float(product.get('price', 0.0)),
                    "rating": float(product.get('rating', 0.0))
                }
            )
            documents.append(doc)

        vectorstore.add_documents(documents)
        return f"✅ Successfully uploaded {len(products_data)} products from JSON!"
    except Exception as e:
        return f"❌ Error uploading JSON: {str(e)}"

# ==================== Initialize System ====================

print("Initializing SmartRetail AI...")
vectorstore = create_vectorstore()
orchestrator = AgentOrchestrator(vectorstore, llm)
print("✅ SmartRetail AI initialized successfully!")

# ==================== Gradio Interface ====================

def chat_function(message, history):
    """Main chat function for Gradio interface"""
    try:
        response = orchestrator.process_query(message)
        return response
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="SmartRetail AI - Intelligent Shopping Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🛍️ SmartRetail AI - Intelligent Shopping Assistant

    **Powered by NVIDIA Llama-3.1-Nemotron & NV-Embed-v2**

    ### Features:
    - 🔍 **Smart Search**: Natural language product search with semantic understanding
    - 📊 **Product Comparison**: Compare multiple products with detailed analysis
    - 💡 **Recommendations**: Get personalized product recommendations with reasoning
    - 💬 **Q&A**: Ask questions about products and shopping

    ---
    """)

    with gr.Tab("💬 Chat with AI Assistant"):
        gr.Markdown("### Ask me anything about products!")
        gr.Markdown("**Example queries:**")
        gr.Markdown("""
        - "Show me running shoes under $150"
        - "Compare the top 3 laptops for programming"
        - "Recommend a good camera for beginners"
        - "What are the best features of wireless headphones?"
        """)

        chatbot = gr.ChatInterface(
            fn=chat_function,
            type="messages",
            examples=[
                "Show me running shoes for flat feet under $150",
                "Compare gaming laptops under $1000",
                "Recommend the best smartphone for photography",
                "What should I look for when buying headphones?",
            ],
            cache_examples=False,
        )

    with gr.Tab("📄 Upload Product Catalog (PDF)"):
        gr.Markdown("### Upload PDF documents containing product information")
        gr.Markdown("The AI will extract product details and add them to the searchable catalog.")

        with gr.Row():
            file_output = gr.Textbox(label="Upload Status", lines=3)
        with gr.Row():
            upload_button = gr.UploadButton(
                "📁 Click to upload PDF documents",
                file_types=[".pdf"],
                file_count="multiple"
            )
            upload_button.upload(
                fn=upload_documents,
                inputs=upload_button,
                outputs=file_output
            )

    with gr.Tab("📦 Upload Products (JSON)"):
        gr.Markdown("### Upload products from JSON file")
        gr.Markdown("""
        **JSON Format:**
        ```json
        [
            {
                "product_id": "prod_001",
                "name": "Product Name",
                "description": "Product description",
                "price": 99.99,
                "category": "Electronics",
                "brand": "BrandName",
                "rating": 4.5,
                "specifications": {"key": "value"}
            }
        ]
        ```
        """)

        with gr.Row():
            json_output = gr.Textbox(label="Upload Status", lines=3)
        with gr.Row():
            json_upload = gr.UploadButton(
                "📁 Click to upload JSON file",
                file_types=[".json"],
                file_count="single"
            )
            json_upload.upload(
                fn=upload_products_json,
                inputs=json_upload,
                outputs=json_output
            )

    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## About SmartRetail AI

        SmartRetail AI is an intelligent shopping assistant built with:

        ### 🏗️ Architecture
        - **Multi-Agent System**: Specialized agents for search, comparison, and recommendations
        - **NVIDIA NIM**: Llama-3.1-Nemotron-Nano-8B for reasoning and NV-Embed-v2 for embeddings
        - **Vector Search**: OpenSearch for semantic product search
        - **RAG Pipeline**: Retrieval-Augmented Generation for accurate responses

        ### 🤖 Agents
        1. **Query Understanding Agent**: Analyzes user intent and extracts entities
        2. **Product Search Agent**: Performs semantic search across product catalog
        3. **Comparison Agent**: Compares products with detailed analysis
        4. **Recommendation Agent**: Provides personalized recommendations with reasoning
        5. **Agent Orchestrator**: Coordinates all agents using ReAct pattern

        ### 🎯 Key Features
        - Natural language understanding
        - Transparent reasoning
        - Semantic product search
        - Multi-turn conversations
        - Context-aware responses

        ---

        Built for the **NVIDIA NIM + AWS EKS Hackathon**
        """)

# Launch the application
if __name__ == "__main__":
    demo.launch(share=False)
