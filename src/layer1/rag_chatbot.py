"""
Layer 1: UX Automation - RAG-Enhanced Chatbot
Production-ready conversational AI with retrieval-augmented generation

Features:
- Domain-specific knowledge retrieval
- Context-aware conversation management  
- Multi-source document integration
- Intelligent response generation with citations
"""

from typing import List, Dict, Any, Optional
import json
import asyncio
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

try:
    from openai import AsyncOpenAI
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings.openai import OpenAIEmbeddings
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("Optional dependencies not installed. Run: pip install openai langchain chromadb")

@dataclass
class ChatMessage:
    """Structured chat message with metadata"""
    content: str
    role: str  # 'user', 'assistant', 'system'
    timestamp: datetime
    sources: Optional[List[str]] = None
    confidence_score: Optional[float] = None

@dataclass
class DocumentChunk:
    """Document chunk for RAG processing"""
    content: str
    source: str
    chunk_id: str
    metadata: Dict[str, Any]

class EnterpriseRAGChatbot:
    """
    Enterprise-grade RAG chatbot with domain-specific knowledge integration
    Designed for production deployment with comprehensive error handling
    """
    
    def __init__(self, 
                 knowledge_base_path: str = "./knowledge_base",
                 model_name: str = "gpt-4",
                 max_context_length: int = 4000,
                 similarity_threshold: float = 0.75):
        
        self.knowledge_base_path = Path(knowledge_base_path)
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.similarity_threshold = similarity_threshold
        
        # Initialize components
        self._initialize_vector_store()
        self._initialize_llm_client()
        self._initialize_text_splitter()
        
        # Conversation state management
        self.conversation_history: List[ChatMessage] = []
        self.system_prompt = self._create_system_prompt()
    
    def _initialize_vector_store(self):
        """Initialize ChromaDB vector store for document retrieval"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.knowledge_base_path / "chroma_db"),
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name="enterprise_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"Vector store initialization error: {e}")
            self.chroma_client = None
            self.collection = None
    
    def _initialize_llm_client(self):
        """Initialize OpenAI client for response generation"""
        try:
            self.llm_client = AsyncOpenAI()
        except Exception as e:
            print(f"LLM client initialization error: {e}")
            self.llm_client = None
    
    def _initialize_text_splitter(self):
        """Initialize text splitter for document processing"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _create_system_prompt(self) -> str:
        """Create domain-specific system prompt"""
        return """
        You are an expert AI assistant specializing in enterprise AI solutions and three-layer architecture implementations.
        
        Your expertise covers:
        - Layer 1: UX Automation (Microsoft Copilot, RAG chatbots, workflow builders)
        - Layer 2: Data Intelligence (Knowledge graphs, process mining, data pipelines)  
        - Layer 3: Strategic Intelligence (Azure AI Foundry, forecasting, scenario planning)
        
        Guidelines:
        - Provide accurate, actionable information based on retrieved context
        - Always cite sources when referencing specific documents or data
        - Focus on practical implementation advice with real-world examples
        - Maintain professional tone suitable for enterprise stakeholders
        - If uncertain about specific details, acknowledge limitations clearly
        """
    
    async def ingest_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Ingest documents into the knowledge base
        
        Args:
            documents: List of documents with 'content', 'source', and 'metadata'
        
        Returns:
            bool: Success status
        """
        if not self.collection:
            return False
        
        try:
            all_chunks = []
            for doc in documents:
                chunks = self._process_document(doc)
                all_chunks.extend(chunks)
            
            # Add to vector store
            chunk_contents = [chunk.content for chunk in all_chunks]
            chunk_metadatas = [
                {
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                    **chunk.metadata
                }
                for chunk in all_chunks
            ]
            chunk_ids = [chunk.chunk_id for chunk in all_chunks]
            
            self.collection.add(
                documents=chunk_contents,
                metadatas=chunk_metadatas,
                ids=chunk_ids
            )
            
            return True
            
        except Exception as e:
            print(f"Document ingestion error: {e}")
            return False
    
    def _process_document(self, document: Dict[str, Any]) -> List[DocumentChunk]:
        """Process document into chunks for vector storage"""
        content = document['content']
        source = document['source']
        metadata = document.get('metadata', {})
        
        # Split document into chunks
        text_chunks = self.text_splitter.split_text(content)
        
        # Create DocumentChunk objects
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = DocumentChunk(
                content=chunk_text,
                source=source,
                chunk_id=f"{source}_{i}",
                metadata={
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    async def retrieve_relevant_context(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for the user query
        
        Args:
            query: User query text
            max_results: Maximum number of results to return
        
        Returns:
            List of relevant document chunks with metadata
        """
        if not self.collection:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=max_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Filter by similarity threshold
            relevant_chunks = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0], 
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    # ChromaDB uses cosine distance (lower is more similar)
                    similarity = 1 - distance
                    if similarity >= self.similarity_threshold:
                        relevant_chunks.append({
                            'content': doc,
                            'metadata': metadata,
                            'similarity_score': similarity
                        })
            
            return relevant_chunks
            
        except Exception as e:
            print(f"Context retrieval error: {e}")
            return []
    
    async def generate_response(self, user_message: str) -> ChatMessage:
        """
        Generate AI response using RAG approach
        
        Args:
            user_message: User's input message
        
        Returns:
            ChatMessage with AI response and sources
        """
        if not self.llm_client:
            return ChatMessage(
                content="AI service unavailable. Please check configuration.",
                role="assistant",
                timestamp=datetime.now()
            )
        
        try:
            # 1. Retrieve relevant context
            relevant_chunks = await self.retrieve_relevant_context(user_message)
            
            # 2. Build context-enhanced prompt
            context_text = self._build_context_prompt(relevant_chunks)
            sources = [chunk['metadata']['source'] for chunk in relevant_chunks]
            
            # 3. Prepare conversation messages
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add recent conversation history (last 5 exchanges)
            for msg in self.conversation_history[-10:]:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Add current query with context
            user_prompt = f"""
            Context from knowledge base:
            {context_text}
            
            User question: {user_message}
            
            Please provide a comprehensive answer based on the context above. 
            If you reference specific information, cite the source.
            """
            
            messages.append({"role": "user", "content": user_prompt})
            
            # 4. Generate response
            response = await self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            # 5. Create response message
            response_message = ChatMessage(
                content=ai_response,
                role="assistant",
                timestamp=datetime.now(),
                sources=list(set(sources)) if sources else None,
                confidence_score=self._calculate_confidence_score(relevant_chunks)
            )
            
            # 6. Update conversation history
            user_msg = ChatMessage(
                content=user_message,
                role="user", 
                timestamp=datetime.now()
            )
            
            self.conversation_history.extend([user_msg, response_message])
            
            return response_message
            
        except Exception as e:
            print(f"Response generation error: {e}")
            return ChatMessage(
                content=f"I encountered an error processing your request: {str(e)}",
                role="assistant",
                timestamp=datetime.now()
            )
    
    def _build_context_prompt(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Build context prompt from relevant chunks"""
        if not relevant_chunks:
            return "No specific context found in knowledge base."
        
        context_parts = []
        for chunk in relevant_chunks:
            source = chunk['metadata']['source']
            content = chunk['content']
            context_parts.append(f"Source: {source}\n{content}\n")
        
        return "\n---\n".join(context_parts)
    
    def _calculate_confidence_score(self, relevant_chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieval quality"""
        if not relevant_chunks:
            return 0.0
        
        avg_similarity = sum(chunk['similarity_score'] for chunk in relevant_chunks) / len(relevant_chunks)
        chunk_count_factor = min(len(relevant_chunks) / 3.0, 1.0)  # Normalize to max of 3 chunks
        
        return (avg_similarity * 0.7) + (chunk_count_factor * 0.3)
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get conversation summary with statistics"""
        total_messages = len(self.conversation_history)
        user_messages = sum(1 for msg in self.conversation_history if msg.role == "user")
        assistant_messages = sum(1 for msg in self.conversation_history if msg.role == "assistant")
        
        avg_confidence = 0.0
        confidence_scores = [msg.confidence_score for msg in self.conversation_history 
                           if msg.confidence_score is not None]
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        return {
            "total_messages": total_messages,
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "average_confidence": round(avg_confidence, 2),
            "sources_used": len(set([
                source for msg in self.conversation_history 
                if msg.sources for source in msg.sources
            ]))
        }
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []

# Example usage and demo functions
async def demo_rag_chatbot():
    """Demonstrate RAG chatbot capabilities"""
    print("🤖 Enterprise RAG Chatbot Demo")
    print("=" * 40)
    
    # Initialize chatbot
    chatbot = EnterpriseRAGChatbot()
    
    # Sample documents (in production, these would come from your knowledge base)
    sample_docs = [
        {
            "content": """
            The Three-Layer AI Architecture provides a structured approach to enterprise AI implementation:
            
            Layer 1 (UX Automation): Focuses on user-facing AI applications including Microsoft Copilot plugins, 
            RAG chatbots, and visual workflow builders. This layer achieves 85% user adoption rates.
            
            Layer 2 (Data Intelligence): Handles knowledge graphs, process mining, and real-time data pipelines. 
            Provides 90% faster insights through automated data processing.
            
            Layer 3 (Strategic Intelligence): Implements Azure AI Foundry for forecasting and scenario planning.
            Delivers 300% ROI through strategic decision support systems.
            """,
            "source": "three_layer_architecture_guide.md",
            "metadata": {"type": "architecture_guide", "version": "2.1"}
        },
        {
            "content": """
            Microsoft Copilot Studio enables custom plugin development for enterprise workflow automation.
            Key features include:
            - Custom action development with semantic triggers
            - Integration with Microsoft 365 ecosystem
            - Advanced prompt engineering capabilities
            - Enterprise security and compliance features
            
            Best practices for Copilot plugin development:
            1. Define clear use cases and user personas
            2. Implement robust error handling and fallback mechanisms
            3. Use semantic kernel for multi-step reasoning
            4. Integrate with existing business systems via APIs
            """,
            "source": "copilot_plugin_development.md",
            "metadata": {"type": "technical_guide", "layer": 1}
        }
    ]
    
    # Ingest documents
    print("📚 Ingesting knowledge base...")
    success = await chatbot.ingest_documents(sample_docs)
    
    if success:
        print("✅ Knowledge base ready")
    else:
        print("❌ Knowledge base ingestion failed")
        return
    
    # Demo conversation
    demo_queries = [
        "What is the Three-Layer AI Architecture?",
        "How do I develop Microsoft Copilot plugins?",
        "What are the business benefits of each layer?",
        "What best practices should I follow for enterprise AI?"
    ]
    
    for query in demo_queries:
        print(f"\n👤 User: {query}")
        response = await chatbot.generate_response(query)
        
        print(f"🤖 Assistant: {response.content}")
        
        if response.sources:
            print(f"📖 Sources: {', '.join(response.sources)}")
        
        if response.confidence_score:
            print(f"🎯 Confidence: {response.confidence_score:.2f}")
    
    # Show conversation summary
    print(f"\n📊 Conversation Summary:")
    summary = chatbot.get_conversation_summary()
    for key, value in summary.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    asyncio.run(demo_rag_chatbot())