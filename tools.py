# tools.py
"""
Agent Tools
Document search and calculator tools for the agent.
"""

from typing import Optional
from retriever import DocumentRetriever


class AgentTools:
    def __init__(self, index_dir: str = "./index"):
        """Initialize agent tools."""
        self.retriever = DocumentRetriever(index_dir=index_dir)
    
    def document_search(self, query: str) -> str:
        """
        Search documents for relevant information.
        
        Args:
            query: Search query
        
        Returns:
            Retrieved context or message if no results
        """
        try:
            context = self.retriever.retrieve_context(query, top_k=5)
            
            if not context or context.strip() == "":
                return "No relevant documents found for this query."
            
            return context
        except Exception as e:
            return f"Error searching documents: {str(e)}"
    
    @staticmethod
    def calculator(expression: str) -> str:
        """
        Evaluate mathematical expressions.
        
        Args:
            expression: Math expression as string (e.g., "2 + 2", "145 * 37")
        
        Returns:
            Calculation result or error message
        """
        try:
            # Security: Only allow safe mathematical operations
            allowed_chars = set("0123456789+-*/().% ")
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression. Only numbers and basic operators allowed."
            
            result = eval(expression)
            return f"The result is: {result}"
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"


if __name__ == "__main__":
    # Example usage
    tools = AgentTools(index_dir="./index")
    
    # Test document search
    print("Testing document search:")
    result = tools.document_search("machine learning")
    print(result[:200], "...\n")
    
    # Test calculator
    print("Testing calculator:")
    print(tools.calculator("145 * 37"))
    print(tools.calculator("100 / 5 + 20"))