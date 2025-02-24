import streamlit as st
import os
import json
import requests
import sys
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

# Check if anthropic is installed
try:
    import anthropic
except ImportError:
    st.error("The anthropic package is not installed. Please install it with 'pip install anthropic'")
    st.info("If you're running this on Streamlit Cloud, make sure you have a requirements.txt file with 'anthropic' listed in it.")
    st.stop()

# Load environment variables if available
load_dotenv()

# Set title and description
st.title("🧠 Research Assistant")
st.write(
    "This research assistant uses Claude 3.7 and Brave Search to perform step-by-step research on any topic. "
    "Enter your API keys below to get started."
)

class ResearchAssistant:
    def __init__(self, anthropic_api_key, brave_api_key):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.brave_api_key = brave_api_key
        self.conversation_history = []
        self.search_results_cache = {}
        
    def web_search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Perform a web search using Brave Search API."""
        # Check cache first
        cache_key = f"{query}_{num_results}"
        if cache_key in self.search_results_cache:
            return self.search_results_cache[cache_key]
            
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.brave_api_key
        }
        params = {
            "q": query,
            "count": num_results,
            "search_lang": "en"
        }
        
        try:
            with st.status(f"🔎 Searching: {query}", expanded=False) as status:
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                results = response.json()
                
                processed_results = []
                if "web" in results and "results" in results["web"]:
                    for item in results["web"]["results"][:num_results]:
                        processed_results.append({
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "description": item.get("description", "")
                        })
                    status.update(label=f"✅ Found {len(processed_results)} results for: {query}", state="complete")
                else:
                    status.update(label=f"⚠️ No results found for: {query}", state="error")
                
                # Cache the results
                self.search_results_cache[cache_key] = processed_results
                return processed_results
                
        except Exception as e:
            st.error(f"Error performing web search: {e}")
            return []
    
    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into a readable string."""
        if not results:
            return "No search results found."
            
        formatted = "### Search Results:\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"{i}. **{result['title']}**\n"
            formatted += f"   URL: {result['url']}\n"
            formatted += f"   {result['description']}\n\n"
            
        return formatted
    
    def create_tool_specs(self) -> List[Dict[str, Any]]:
        """Create tool specifications for Claude API."""
        return [
            {
                "name": "web_search",
                "description": "Search the web for information on a specific topic or query.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to look up"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of search results to return (default: 5)"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
    
    def process_research_query(self, user_query: str) -> str:
        """Process a research query using Claude and web search."""
        # Add the user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_query})
        
        system_prompt = """You are an expert research assistant powered by Claude 3.7 Sonnet.
Your goal is to perform thorough research on any topic requested by the user.

When researching a topic:
1. Break down complex questions into smaller, manageable parts
2. Use web_search to find relevant and up-to-date information
3. Critically evaluate sources for credibility and relevance
4. Synthesize information from multiple sources
5. Provide detailed, step-by-step reasoning
6. Cite your sources clearly with URLs
7. Present information in a structured, easy-to-understand format
8. Acknowledge uncertainties or limitations in available information
9. Avoid making assumptions or presenting opinions as facts

Always begin by understanding the research question, then plan your approach before diving into searches.
"""

        # Create a function to process tool use
        def handle_tool_use(response):
            for content_item in response.content:
                if content_item.type == "tool_use":
                    tool_name = content_item.name
                    tool_input = content_item.input
                    tool_id = content_item.id
                    
                    if tool_name == "web_search":
                        query = tool_input.get("query", "")
                        num_results = tool_input.get("num_results", 5)
                        
                        # Perform the search
                        search_results = self.web_search(query, num_results)
                        formatted_results = self.format_search_results(search_results)
                        
                        # Add tool use to history
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": [
                                {"type": "tool_use", "id": tool_id, "name": tool_name, "input": tool_input}
                            ]
                        })
                        
                        # Add tool result to history
                        self.conversation_history.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": formatted_results
                                }
                            ]
                        })
                        
                        return True  # Indicate tool was used
            return False  # No tool was used
        
        try:
            research_status = st.status("🤔 Thinking about your research question...", expanded=True)
            
            max_iterations = 5  # Prevent infinite loops
            iterations = 0
            
            while iterations < max_iterations:
                iterations += 1
                
                # Send message to Claude with tool specs
                response = self.client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=4096,
                    temperature=0.2,
                    system=system_prompt,
                    messages=self.conversation_history,
                    tools=self.create_tool_specs()
                )
                
                # Check for tool use
                if response.stop_reason == "tool_use":
                    # Show thinking progress
                    for content_item in response.content:
                        if content_item.type == "text":
                            thinking = content_item.text.split('\n')[0]
                            research_status.update(label=f"💭 {thinking}")
                    
                    # Process tool use
                    tool_used = handle_tool_use(response)
                    if not tool_used:
                        break
                else:
                    # We got a final response
                    research_status.update(label="✅ Research complete!", state="complete")
                    break
            
            # Extract final text from the last response
            final_text = ""
            for content_item in response.content:
                if content_item.type == "text":
                    final_text += content_item.text
            
            # Add final response to conversation history
            if final_text:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_text
                })
            
            return final_text
            
        except anthropic.APIError as e:
            error_msg = f"Anthropic API Error: {str(e)}"
            st.error(error_msg)
            return error_msg
        except requests.RequestException as e:
            error_msg = f"Network Error: {str(e)}"
            st.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected Error: {str(e)}"
            st.error(error_msg)
            return error_msg
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        return "Conversation has been reset."

# API key inputs
with st.sidebar:
    st.header("API Keys")
    # Get API keys from environment variables or user input
    default_anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    default_brave_key = os.getenv("BRAVE_API_KEY", "")
    
    anthropic_api_key = st.text_input("Anthropic API Key", 
                                      value=default_anthropic_key,
                                      type="password", 
                                      placeholder="Enter your Anthropic API key")
    
    brave_api_key = st.text_input("Brave Search API Key", 
                                 value=default_brave_key,
                                 type="password", 
                                 placeholder="Enter your Brave Search API key")
    
    if st.button("Reset Conversation"):
        if "messages" in st.session_state:
            st.session_state.messages = []
        if "assistant" in st.session_state:
            st.session_state.assistant.reset_conversation()
        st.success("Conversation has been reset")

# Check for API keys
if not anthropic_api_key or not brave_api_key:
    st.info("Please add your API keys in the sidebar to continue.", icon="🔑")
    
    # Add links for getting API keys
    st.markdown("""
    ### How to get API keys:
    - [Anthropic API Key](https://console.anthropic.com/)
    - [Brave Search API Key](https://brave.com/search/api/)
    """)
else:
    # Initialize the assistant
    if "assistant" not in st.session_state:
        st.session_state.assistant = ResearchAssistant(anthropic_api_key, brave_api_key)
        
    # Initialize messages if not in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if prompt := st.chat_input("What would you like to research?"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process research query
        with st.chat_message("assistant"):
            response = st.session_state.assistant.process_research_query(prompt)
            st.markdown(response)
            
        # Store assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})