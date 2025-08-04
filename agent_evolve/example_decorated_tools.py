"""
Example file with decorated tools for testing the optimization system.
"""
import logging
from typing import List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

# Import the optimization decorators
from evolve_decorator import evolve, evolve_nlg, evolve_research, evolve_classification

logger = logging.getLogger(__name__)

# Example 1: Simple function with basic decorator
@evolve(description="Generate a compelling essay on any topic")
def generate_essay(topic: str, length: int = 500) -> str:
    """Generate an essay on the given topic"""
    model = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    prompt = f"""Write a compelling {length}-word essay about {topic}.
    Make it engaging, well-structured, and informative."""
    
    response = model.invoke([HumanMessage(content=prompt)])
    return response.content


# Example 2: NLG tool with specific metrics and metadata
@evolve_nlg(
    name="linkedin_post_creator",
    description="Create professional LinkedIn posts with optimal engagement",
    metrics=["engagement", "professionalism", "clarity", "call_to_action"],
    metadata={"platform": "linkedin", "max_length": 300, "target_audience": "professionals"}
)
def create_linkedin_post(topic: str, tone: str = "professional", include_hashtags: bool = True) -> str:
    """Create a LinkedIn post optimized for engagement"""
    model = ChatOpenAI(model="gpt-4o", temperature=0.8)
    
    hashtag_instruction = "Include 3-5 relevant hashtags at the end." if include_hashtags else ""
    
    prompt = f"""Create a compelling LinkedIn post about {topic}.
    
    Requirements:
    - Tone: {tone}
    - Length: 150-300 words
    - Professional and engaging
    - Include a clear call to action
    {hashtag_instruction}
    
    Make it valuable for a professional audience."""
    
    response = model.invoke([HumanMessage(content=prompt)])
    return response.content


# Example 3: Research tool
@evolve_research(
    description="Research and summarize information about any topic",
    metrics=["accuracy", "completeness", "relevance", "timeliness"]
)
def research_topic(query: str, max_sources: int = 5) -> dict:
    """Research a topic and return summarized findings"""
    # This would normally use Tavily or another search API
    # For this example, we'll simulate with ChatGPT
    model = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    prompt = f"""Research the topic: {query}
    
    Provide a comprehensive summary including:
    1. Key facts and current information
    2. Recent developments or trends
    3. Important statistics or data points
    4. Expert opinions or insights
    
    Format as a structured summary."""
    
    response = model.invoke([HumanMessage(content=prompt)])
    
    return {
        "query": query,
        "summary": response.content,
        "sources_used": max_sources,
        "research_date": "2024-07-30"
    }


# Example 4: Classification tool
@evolve_classification(
    description="Classify text sentiment with confidence scores",
    metrics=["accuracy", "precision", "recall", "confidence_calibration"]
)
def classify_sentiment(text: str) -> dict:
    """Classify the sentiment of text"""
    model = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    prompt = f"""Analyze the sentiment of this text: "{text}"
    
    Respond with ONLY a JSON object:
    {{
        "sentiment": "positive|negative|neutral",
        "confidence": 0.95,
        "reasoning": "brief explanation"
    }}"""
    
    response = model.invoke([HumanMessage(content=prompt)])
    
    try:
        import json
        return json.loads(response.content)
    except:
        return {
            "sentiment": "neutral",
            "confidence": 0.5,
            "reasoning": "Failed to parse response"
        }


# Example 5: Class with decorated methods
class ContentGenerator:
    """Example class with decorated methods"""
    
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    @evolve_nlg(
        name="tweet_generator",
        description="Generate engaging tweets with hashtags",
        metrics=["engagement", "virality", "clarity", "hashtag_relevance"]
    )
    def generate_tweet(self, topic: str, style: str = "casual") -> str:
        """Generate a tweet about the topic"""
        prompt = f"""Create an engaging tweet about {topic}.
        
        Style: {style}
        Requirements:
        - Under 280 characters
        - Include 2-3 relevant hashtags
        - Make it shareable and conversation-starting
        
        Be creative and engaging!"""
        
        response = self.model.invoke([HumanMessage(content=prompt)])
        return response.content
    
    @evolve(
        category="natural_language_generation",
        description="Create email subject lines that improve open rates",
        metrics=["open_rate_prediction", "clarity", "urgency", "personalization"]
    )
    def create_email_subject(self, email_content: str, audience: str = "general") -> str:
        """Create compelling email subject lines"""
        prompt = f"""Based on this email content, create 3 compelling subject lines:
        
        Email content: {email_content}
        Target audience: {audience}
        
        Requirements:
        - Under 60 characters
        - Clear and compelling
        - Avoid spam words
        - Create urgency or curiosity
        
        Return just the 3 subject lines, one per line."""
        
        response = self.model.invoke([HumanMessage(content=prompt)])
        return response.content


# Regular function without decorator (should be ignored)
def helper_function(text: str) -> str:
    """This function should not be extracted"""
    return text.upper()


if __name__ == "__main__":
    # Test the decorated functions
    print("Testing decorated functions:")
    
    # Test simple function
    essay = generate_essay("artificial intelligence", 200)
    print(f"Essay: {essay[:100]}...")
    
    # Test LinkedIn post
    post = create_linkedin_post("remote work benefits", "professional")
    print(f"LinkedIn post: {post[:100]}...")
    
    # Test class method
    generator = ContentGenerator()
    tweet = generator.generate_tweet("climate change", "urgent")
    print(f"Tweet: {tweet}")
    
    # Show registered functions
    from evolve_decorator import EvolveCandidate
    print(f"\nRegistered functions: {len(EvolveCandidate.get_registered_functions())}")
    for func_info in EvolveCandidate.get_registered_functions():
        print(f"- {func_info['name']}: {func_info['category']}")