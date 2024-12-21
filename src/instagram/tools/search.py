import json
import os
from typing import List, Dict, Optional
import requests
from openai import OpenAI
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader

class SearchTools:
    """
    A collection of search tools using OpenAI for web search and content analysis.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize SearchTools with OpenAI configuration.
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided, will try to get from environment
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Base URL for search API
        self.search_url = "https://google.serper.dev/search"
        self.search_api_key = os.getenv("SERPER_API_KEY")

    def _get_ai_response(self, prompt: str) -> str:
        """
        Get response from OpenAI API.
        
        Args:
            prompt (str): The prompt to send to OpenAI
            
        Returns:
            str: The formatted response
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",  # Using GPT-4 for best results
                messages=[
                    {"role": "system", "content": "You are a helpful search assistant that provides accurate and relevant information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error querying OpenAI: {str(e)}"

    def _perform_web_search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Perform a web search using Serper API.
        
        Args:
            query (str): Search query
            limit (int): Number of results to return
            
        Returns:
            List[Dict]: List of search results
        """
        try:
            headers = {
                "X-API-KEY": self.search_api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "q": query,
                "num": limit
            }
            
            response = requests.post(self.search_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()['organic']
        except Exception as e:
            return [{"title": f"Error performing search: {str(e)}"}]

    @tool('search internet')
    def search_internet(self, query: str) -> str:
        """
        Search the internet and enhance results with OpenAI analysis.
        
        Args:
            query (str): The search query
            
        Returns:
            str: Enhanced search results
        """
        # Get raw search results
        raw_results = self._perform_web_search(query)
        
        # Create prompt for OpenAI to analyze and enhance results
        prompt = f"""
        Analyze and enhance these search results for the query: "{query}"
        
        Raw results:
        {json.dumps(raw_results, indent=2)}
        
        Please provide:
        1. A brief summary of the search results
        2. The most relevant points from each result
        3. Any additional context or insights
        
        Format the response in a clear, readable way.
        """
        
        enhanced_results = self._get_ai_response(prompt)
        return enhanced_results

    @tool('search instagram')
    def search_instagram(self, query: str) -> str:
        """
        Search Instagram-related content and analyze with OpenAI.
        
        Args:
            query (str): The search query
            
        Returns:
            str: Enhanced Instagram search results
        """
        # Get raw search results with Instagram filter
        raw_results = self._perform_web_search(f"site:instagram.com {query}", limit=5)
        
        # Create prompt for OpenAI to analyze Instagram content
        prompt = f"""
        Analyze these Instagram-related search results for: "{query}"
        
        Raw results:
        {json.dumps(raw_results, indent=2)}
        
        Please provide:
        1. A summary of trending content related to the query
        2. Relevant hashtags and themes
        3. Notable Instagram accounts or posts
        4. Content patterns and engagement insights
        
        Format the response in a clear, readable way.
        """
        
        enhanced_results = self._get_ai_response(prompt)
        return enhanced_results

    @tool('open page')
    def open_page(self, url: str) -> str:
        """
        Open a webpage, get its content, and analyze with OpenAI.
        
        Args:
            url (str): The URL to load
            
        Returns:
            str: Analyzed webpage content
        """
        try:
            # Load raw content
            loader = WebBaseLoader(url)
            raw_content = loader.load()
            
            # Create prompt for OpenAI to analyze the content
            prompt = f"""
            Analyze this webpage content and provide:
            1. A comprehensive summary
            2. Key points and insights
            3. Relevant contextual information
            
            Content:
            {raw_content}
            """
            
            analyzed_content = self._get_ai_response(prompt)
            return analyzed_content
            
        except Exception as e:
            return f"Error processing webpage: {str(e)}"

