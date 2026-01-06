# agent.py
"""
Agent Logic
Intent classification, agent reasoning, and confidence checking.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from tools import AgentTools


# Load environment variables
load_dotenv()


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
    
    def __init__(self, index_dir: str = "./index", model_name: str = "gemini-2.0-flash-lite"):
        """
        Initialize agentic RAG system.
        
        Args:
            index_dir: Directory containing FAISS index
            model_name: Gemini model name
        """
        # Initialize LLM
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Please set it as an environment variable:\n"
                "Windows PowerShell: $env:GOOGLE_API_KEY='your_key_here'\n"
                "Or create a .env file with: GOOGLE_API_KEY=your_key_here"
            )
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.1,
            google_api_key=api_key
        )
        
        # Initialize tools
        self.agent_tools = AgentTools(index_dir=index_dir)
        
        # Initialize intent classifier
        self.intent_classifier = IntentClassifier(self.llm)
    
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
        intent = self.intent_classifier.classify(user_query)
        print(f"Classified Intent: {intent}\n")
        
        # Step 2: Execute agent with tools
        try:
            answer = self._execute_with_tools(user_query, intent)
        except Exception as e:
            print(f"Agent execution error: {e}")
            answer = "I encountered an error processing your request."
        
        # Step 3: Confidence check
        final_result = self._check_confidence(user_query, answer, intent)
        
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