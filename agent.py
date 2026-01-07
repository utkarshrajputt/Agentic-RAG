# agent.py
"""
Agent Logic
Intent classification, agent reasoning, and confidence checking.
"""

import os
import time
import hashlib
import json
from typing import Dict, Any, Optional
from functools import lru_cache, wraps
from collections import OrderedDict
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from tools import AgentTools


# Load environment variables
load_dotenv()


def create_llm(provider: str = None, model_name: str = None):
    """
    Factory function to create LLM based on provider.
    
    Args:
        provider: 'arliai', 'gemini', 'openai', or 'ollama'. If None, reads from env.
        model_name: Model name. If None, uses default for provider.
    
    Returns:
        LLM instance
    """
    # Get provider from env if not specified
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "arliai").lower()
    
    print(f"🤖 Initializing LLM: {provider}")
    
    if provider == "arliai":
        # Arli AI using OpenAI-compatible API
        api_key = os.getenv("ARLIAI_API_KEY")
        if not api_key:
            raise ValueError(
                "ARLIAI_API_KEY not found. Add to .env file:\n"
                "ARLIAI_API_KEY=your_key_here"
            )
        
        # Try Gemma model (might have separate trial limit)
        model = model_name or "Gemma-3-27B-it"
        
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://api.arliai.com/v1",
            temperature=0.1
        )
    
    elif provider == "gemini":
        # Google Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Add to .env file:\n"
                "GOOGLE_API_KEY=your_key_here"
            )
        
        model = model_name or "gemini-2.0-flash-lite"
        
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0.1,
            google_api_key=api_key
        )
    
    elif provider == "openai":
        # OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Add to .env file:\n"
                "OPENAI_API_KEY=your_key_here"
            )
        
        model = model_name or "gpt-3.5-turbo"
        
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0.1
        )
    
    elif provider == "ollama":
        # Local Ollama (unlimited, free)
        model = model_name or "llama3.2"
        
        return ChatOpenAI(
            model=model,
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # Dummy key, not needed
            temperature=0.1
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'arliai', 'gemini', 'openai', or 'ollama'")


class QueryCache:
    """Simple LRU cache for query responses."""
    
    def __init__(self, max_size: int = 100):
        """Initialize cache with max size."""
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _hash_query(self, query: str, intent: str) -> str:
        """Create hash key from query and intent."""
        key = f"{query.lower().strip()}:{intent}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, query: str, intent: str) -> Optional[Dict[str, Any]]:
        """Get cached response if exists."""
        key = self._hash_query(query, intent)
        if key in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, query: str, intent: str, response: Dict[str, Any]):
        """Cache a response."""
        key = self._hash_query(query, intent)
        self.cache[key] = response
        self.cache.move_to_end(key)
        
        # Remove oldest if over max size
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def clear(self):
        """Clear all cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%"
        }


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retrying with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e)
                    
                    # Check for rate limit errors
                    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                        retries += 1
                        if retries >= max_retries:
                            print(f"\n⚠️ Rate limit exceeded after {max_retries} retries.")
                            raise Exception(
                                "API rate limit exceeded. Please wait a moment and try again."
                            )
                        
                        delay = base_delay * (2 ** (retries - 1))
                        print(f"\n⏳ Rate limit hit. Retrying in {delay:.1f}s... (attempt {retries}/{max_retries})")
                        time.sleep(delay)
                    
                    # Check for other transient errors
                    elif "503" in error_msg or "500" in error_msg or "UNAVAILABLE" in error_msg:
                        retries += 1
                        if retries >= max_retries:
                            raise
                        delay = base_delay * (2 ** (retries - 1))
                        print(f"\n⏳ Service unavailable. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    
                    else:
                        # Non-retryable error
                        raise
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


class IntentClassifier:
    """Classify user intent before agent execution."""
    
    def __init__(self, llm):
        self.llm = llm
        self.intent_prompt = """Classify the user's intent into one of these categories:
- general: General knowledge questions that don't require documents or calculations
- document_search: Questions about specific information that requires searching documents
- calculation: Mathematical calculations or numerical operations

User query: {query}

Respond with only one word: general, document_search, or calculation"""
    
    def classify(self, query: str) -> str:
        """Classify query intent."""
        try:
            prompt = self.intent_prompt.format(query=query)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            intent = response.content.strip().lower()
            
            # Validate intent
            valid_intents = ["general", "document_search", "calculation"]
            if intent in valid_intents:
                return intent
            else:
                return "general"  # Default fallback
        except Exception as e:
            print(f"Intent classification error: {e}")
            return "general"


class AgenticRAG:
    """Main agent with intent classification and confidence checking."""
    
    def __init__(self, index_dir: str = "./index", provider: str = None, model_name: str = None):
        """
        Initialize agentic RAG system.
        
        Args:
            index_dir: Directory containing FAISS index
            provider: LLM provider ('arliai', 'gemini', 'openai'). Reads from env if None.
            model_name: Model name. Uses provider default if None.
        """
        # Initialize LLM using factory
        self.llm = create_llm(provider=provider, model_name=model_name)
        
        # Initialize tools
        self.agent_tools = AgentTools(index_dir=index_dir)
        
        # Initialize intent classifier
        self.intent_classifier = IntentClassifier(self.llm)
        
        # Initialize cache
        self.cache = QueryCache(max_size=100)
        
        print(f"✅ Agent initialized with caching and retry logic")
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _execute_with_tools(self, query: str, intent: str, max_iterations: int = 3) -> str:
        """
        Simple tool execution loop based on intent.
        
        Args:
            query: User query
            intent: Classified intent
            max_iterations: Max tool calls
        
        Returns:
            Final answer
        """
        system_prompt = """You are a helpful AI assistant.

IMPORTANT RULES:
1. For document-related questions, use the retrieved context to answer.
2. For calculations, use the calculation result provided.
3. For general questions, answer directly from your knowledge.
4. If you don't have enough information, clearly state "I don't know" or "I don't have enough information."
5. NEVER make up information. Be honest about limitations."""

        # Execute based on intent
        if intent == "calculation":
            # Extract calculation expression and use calculator
            calc_prompt = f"Extract just the mathematical expression from this query (e.g., '2+2', '145*37'): {query}\nReturn only the expression, nothing else."
            calc_msg = self.llm.invoke([HumanMessage(content=calc_prompt)])
            expression = calc_msg.content.strip()
            
            calc_result = self.agent_tools.calculator(expression)
            
            # Generate natural language response
            final_prompt = f"{system_prompt}\n\nUser asked: {query}\nCalculation result: {calc_result}\n\nProvide a natural language answer."
            response = self.llm.invoke([HumanMessage(content=final_prompt)])
            return response.content
        
        elif intent == "document_search":
            # Search documents
            context = self.agent_tools.document_search(query)
            
            # Generate answer with context
            final_prompt = f"{system_prompt}\n\nUser question: {query}\n\nRetrieved context:\n{context}\n\nAnswer the question using the context. If the context doesn't contain relevant information, say so."
            response = self.llm.invoke([HumanMessage(content=final_prompt)])
            return response.content
        
        else:  # general
            # Answer directly
            final_prompt = f"{system_prompt}\n\nUser question: {query}\n\nProvide a helpful answer."
            response = self.llm.invoke([HumanMessage(content=final_prompt)])
            return response.content
    
    def _check_confidence(self, query: str, answer: str, intent: str) -> Dict[str, Any]:
        """
        Check confidence in the answer and trigger fallback if needed.
        
        Args:
            query: Original query
            answer: Generated answer
            intent: Classified intent
        
        Returns:
            Dict with final_answer and metadata
        """
        # Trigger fallback conditions
        fallback_phrases = [
            "i don't know",
            "i'm not sure",
            "i don't have",
            "no information",
            "cannot find",
            "unable to",
            "not enough information",
            "no relevant documents found"
        ]
        
        answer_lower = answer.lower()
        
        # Check if answer indicates uncertainty
        has_uncertainty = any(phrase in answer_lower for phrase in fallback_phrases)
        
        # Check if answer is too short (likely uncertain)
        is_too_short = len(answer.strip()) < 20
        
        if has_uncertainty or is_too_short:
            return {
                "final_answer": "I don't have enough reliable information to answer that.",
                "confidence": "low",
                "intent": intent,
                "original_answer": answer
            }
        
        return {
            "final_answer": answer,
            "confidence": "high",
            "intent": intent
        }
    
    def query(self, user_query: str) -> Dict[str, Any]:
        """
        Process user query through the complete pipeline.
        
        Args:
            user_query: User's question
        
        Returns:
            Dict with answer and metadata
        """
        print(f"\n{'='*60}")
        print(f"Query: {user_query}")
        print(f"{'='*60}\n")
        
        # Step 1: Classify intent
        try:
            intent = self.intent_classifier.classify(user_query)
            print(f"Classified Intent: {intent}\n")
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                return {
                    "final_answer": "⚠️ API rate limit reached. Please wait a moment and try again.",
                    "confidence": "error",
                    "intent": "error",
                    "error": "rate_limit"
                }
            else:
                print(f"Intent classification error: {e}")
                intent = "general"
        
        # Step 2: Check cache
        cached = self.cache.get(user_query, intent)
        if cached:
            print(f"💾 Cache HIT! Returning cached response.\n")
            return cached
        
        print(f"🔍 Cache MISS. Processing query...\n")
        
        # Step 3: Execute agent with tools
        try:
            answer = self._execute_with_tools(user_query, intent)
        except Exception as e:
            error_msg = str(e)
            print(f"Agent execution error: {e}")
            
            if "rate limit" in error_msg.lower() or "429" in error_msg:
                answer = "I'm currently experiencing high demand. Please try again in a moment."
            elif "503" in error_msg or "unavailable" in error_msg.lower():
                answer = "The service is temporarily unavailable. Please try again shortly."
            else:
                answer = "I encountered an error processing your request."
        
        # Step 4: Confidence check
        final_result = self._check_confidence(user_query, answer, intent)
        
        # Step 5: Cache the result
        self.cache.set(user_query, intent, final_result)
        
        print(f"\n{'='*60}")
        print(f"Final Answer: {final_result['final_answer']}")
        print(f"Confidence: {final_result['confidence']}")
        print(f"{'='*60}\n")
        
        return final_result


if __name__ == "__main__":
    # Example usage
    agent = AgenticRAG(index_dir="./index")
    
    # Test queries
    queries = [
        "What is Artificial Intelligence?",
        "What is 145 * 37?",
        "What does the document say about machine learning?",
        "What is the CEO's private phone number?"
    ]
    
    for query in queries:
        result = agent.query(query)
        print(f"Result: {result}\n")
    
    # Show cache statistics
    print(f"\n📊 Cache Statistics:")
    print(json.dumps(agent.cache.stats(), indent=2))