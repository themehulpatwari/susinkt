"""
LLM-based decision system for YouTube Shorts Stay/Scroll prediction.
Uses Gemini API with dynamic weighting based on user preferences and behavioral data.
"""

from typing import Dict, Tuple, Optional
import os
from dotenv import load_dotenv
import google.generativeai as genai
import re
import json

# Load environment variables
load_dotenv()


class VideoDecisionSystem:
    """
    Makes Stay/Scroll decisions using LLM with user preferences and behavioral embeddings.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the decision system.
        
        Args:
            api_key: Gemini API key. If not provided, will load from GEMINI_API_KEY or API_KEY env variable.
        """
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY or API_KEY not found in environment variables")
        
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')

    def format_metadata_for_prompt(
        self, 
        metadata: Dict,
        stay_similarity: Optional[float] = None,
        scroll_similarity: Optional[float] = None,
        stay_count: int = 0,
        scroll_count: int = 0
    ) -> str:
        """
        Format video metadata into a clean, organized string for the LLM prompt.
        
        Args:
            metadata: Video metadata dictionary
            stay_similarity: Similarity score to Stay collection (0-1)
            scroll_similarity: Similarity score to Scroll collection (0-1)
            stay_count: Number of videos in Stay collection
            scroll_count: Number of videos in Scroll collection
            
        Returns:
            Formatted string with all relevant information
        """
        
        # Build the formatted output
        lines = []
        
        lines.append("VIDEO INFORMATION:")
        lines.append("")
        
        # Basic video details
        lines.append("CONTENT DETAILS:")
        if metadata.get('title'):
            lines.append(f"  Title: {metadata['title']}")
        if metadata.get('channelName'):
            lines.append(f"  Channel: {metadata['channelName']}")
        if metadata.get('description'):
            lines.append(f"  Description: {metadata['description']}")
        if metadata.get('categoryName'):
            lines.append(f"  Category: {metadata['categoryName']}")
        
        lines.append("")
        
        # Tags and keywords
        if metadata.get('tags'):
            tags_str = ', '.join(metadata['tags'])
            lines.append(f"TAGS: {tags_str}")
            lines.append("")
        
        # Visual content
        if metadata.get('thumbnail_context'):
            lines.append(f"VISUAL CONTENT: {metadata['thumbnail_context']}")
            lines.append("")
        
        # Engagement metrics (formatted nicely)
        lines.append("ENGAGEMENT METRICS:")
        if metadata.get('viewCount'):
            views = int(metadata['viewCount'])
            lines.append(f"  Views: {views:,}")
        if metadata.get('likeCount'):
            likes = int(metadata['likeCount'])
            lines.append(f"  Likes: {likes:,}")
        if metadata.get('commentCount'):
            comments = int(metadata['commentCount'])
            lines.append(f"  Comments: {comments:,}")
        if metadata.get('duration'):
            lines.append(f"  Duration: {metadata['duration']} seconds")
        
        # Calculate engagement rate if possible
        if metadata.get('viewCount') and metadata.get('likeCount'):
            views = int(metadata['viewCount'])
            likes = int(metadata['likeCount'])
            if views > 0:
                engagement_rate = (likes / views) * 100
                lines.append(f"  Engagement Rate: {engagement_rate:.2f}%")
        
        lines.append("")
        
        # Behavioral similarity scores (if available)
        if stay_similarity is not None or scroll_similarity is not None:
            lines.append("BEHAVIORAL ANALYSIS:")
            lines.append("")
            lines.append(f"Collection Sizes:")
            lines.append(f"  Videos stayed on: {stay_count}")
            lines.append(f"  Videos scrolled past: {scroll_count}")
            lines.append("")
            
            if stay_similarity is not None:
                lines.append(f"Similarity to STAY collection: {stay_similarity:.4f}")
            if scroll_similarity is not None:
                lines.append(f"Similarity to SCROLL collection: {scroll_similarity:.4f}")
            
            # Calculate confidence based on collection size
            total_samples = stay_count + scroll_count
            if total_samples > 0:
                confidence = min(100, (total_samples / 30) * 100)  # Max confidence at 30+ videos
                lines.append("")
                lines.append(f"Behavioral Data Confidence: {confidence:.0f}%")
                if confidence < 30:
                    lines.append("  Note: Low confidence - prioritize user preferences")
                elif confidence < 70:
                    lines.append("  Note: Moderate confidence - balance preferences and behavior")
                else:
                    lines.append("  Note: High confidence - behavior is reliable")
        
        return '\n'.join(lines)
    
    def build_prompt(
        self,
        formatted_metadata: str,
        user_preferences: str,
        stay_count: int = 0,
        scroll_count: int = 0
    ) -> str:
        """
        Build the complete prompt for the LLM with dynamic weighting instructions.
        
        Args:
            formatted_metadata: Pre-formatted metadata string
            user_preferences: User's explicit preferences
            stay_count: Number of videos in Stay collection
            scroll_count: Number of videos in Scroll collection
            
        Returns:
            Complete prompt string for the LLM
        """
        
        # Calculate confidence and determine weighting strategy
        total_samples = stay_count + scroll_count
        confidence = min(100, (total_samples / 30) * 100)
        
        # Dynamic weighting based on collection size
        if total_samples == 0:
            # Cold start - rely entirely on user preferences
            weight_instruction = (
                "IMPORTANT: This is a cold start with NO behavioral data. "
                "Make your decision ENTIRELY based on the user's preferences. "
                "Ignore the behavioral similarity scores as they are meaningless."
            )
        elif confidence < 30:
            # Low confidence - heavy preference weighting
            weight_instruction = (
                "WEIGHTING STRATEGY: Low behavioral confidence (few samples). "
                "Prioritize user preferences (~95%) with minimal consideration for behavioral patterns (~5%). "
                "The behavioral data may not be representative yet."
            )
        elif confidence < 70:
            # Moderate confidence - balanced approach
            weight_instruction = (
                "WEIGHTING STRATEGY: Moderate behavioral confidence. "
                "Balance user preferences (~80%) with behavioral patterns (~20%). "
                "Use behavioral data to refine your understanding of user preferences."
            )
        else:
            # High confidence - significant behavioral weight
            weight_instruction = (
                "WEIGHTING STRATEGY: High behavioral confidence (many samples). "
                "Balance user preferences (~60%) with strong behavioral patterns (~40%). "
                "Behavioral data is reliable and can inform preference refinement. "
                "If behavior contradicts stated preferences, consider that user's taste may have evolved."
            )
        
        prompt = f"""You are an AI assistant that helps decide whether a user should STAY and watch a YouTube Short or SCROLL to the next one.

USER PREFERENCES:
{user_preferences}

{weight_instruction}

{formatted_metadata}

TASK:
Analyze the video and decide whether the user should STAY or SCROLL.

Your response MUST be a valid JSON object with this EXACT format:
{{
  "decision": "STAY or SCROLL",
  "reasoning": "Brief one-line explanation"
}}

Remember:
- User preferences are ALWAYS the primary factor
- Behavioral patterns help refine understanding but should not override explicit preferences
- Consider content type, topic, engagement quality, and visual elements
- Be decisive - choose either STAY or SCROLL, not both
- Keep reasoning to a single concise sentence"""

        return prompt
    
    def parse_llm_response(self, response_text: str) -> Tuple[str, str]:
        """
        Parse the LLM response to extract decision and reasoning from JSON.
        
        Args:
            response_text: Raw response from LLM (should be JSON)
            
        Returns:
            Tuple of (decision, reasoning)
        """
        try:
            # Try to extract JSON from the response
            # Sometimes LLM might add extra text, so look for JSON object
            json_match = re.search(r'\{[^{}]*"decision"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                decision = data.get('decision', '').upper()
                reasoning = data.get('reasoning', '')
                
                # Validate decision
                if decision not in ['STAY', 'SCROLL']:
                    decision = 'STAY'  # Default
                
                return decision, reasoning
            else:
                # Fallback to old parsing method
                decision_match = re.search(r'DECISION[:\s]*(STAY|SCROLL)', response_text, re.IGNORECASE)
                reasoning_match = re.search(r'REASONING[:\s]*(.+)', response_text, re.DOTALL)
                
                if decision_match:
                    decision = decision_match.group(1).upper()
                else:
                    decision = "STAY" if 'STAY' in response_text.upper() else "SCROLL" if 'SCROLL' in response_text.upper() else "STAY"
                
                reasoning = reasoning_match.group(1).strip() if reasoning_match else response_text.strip()
                
                return decision, reasoning
                
        except json.JSONDecodeError:
            # If JSON parsing fails, use fallback
            if 'STAY' in response_text.upper():
                decision = "STAY"
            elif 'SCROLL' in response_text.upper():
                decision = "SCROLL"
            else:
                decision = "STAY"
            
            reasoning = response_text.strip()
            return decision, reasoning
    
    def make_decision(
        self,
        metadata: Dict,
        user_preferences: str,
        stay_similarity: Optional[float] = None,
        scroll_similarity: Optional[float] = None,
        stay_count: int = 0,
        scroll_count: int = 0
    ) -> Tuple[str, str]:
        """
        Make a Stay/Scroll decision using LLM.
        
        Args:
            metadata: Video metadata dictionary
            user_preferences: User's explicit preferences (e.g., "I want funny cat videos")
            stay_similarity: Similarity to Stay collection
            scroll_similarity: Similarity to Scroll collection
            stay_count: Number of videos in Stay collection
            scroll_count: Number of videos in Scroll collection
            
        Returns:
            Tuple of (decision, reasoning) where decision is "STAY" or "SCROLL"
        """
        # Format the metadata
        formatted_metadata = self.format_metadata_for_prompt(
            metadata, stay_similarity, scroll_similarity, stay_count, scroll_count
        )
        
        # Build the prompt
        prompt = self.build_prompt(formatted_metadata, user_preferences, stay_count, scroll_count)
        
        # Send to Gemini API
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Parse the response
            decision, reasoning = self.parse_llm_response(response_text)
            
            return decision, reasoning
            
        except Exception as e:
            # Fallback in case of API error
            print(f"Error calling Gemini API: {e}")
            return "STAY", f"Error occurred: {str(e)}"


if __name__ == "__main__":
    # Simple test
    sample_metadata = {
        'title': 'Be kind to our delivery drivers! ❤️ #goodnews #kindness #ringcam #goodman',
        'channelName': 'Cosondra',
        'description': 'Feel Good News!',
        'tags': ['freegood', 'goodnews', 'bekind', 'delivery drivers'],
        'viewCount': '49089759',
        'likeCount': '1719398',
        'commentCount': '25314',
        'duration': 20,
        'thumbnail_context': 'a person is walking down a sidewalk with a camera',
        'categoryName': 'Music'
    }
    
    try:
        # Try to initialize with API key from env
        system = VideoDecisionSystem()
        
        user_prefs = "I want to watch wholesome, feel-good content and funny animal videos"
        
        print("Testing LLM decision system...")
        print("="*80)
        
        decision, reasoning = system.make_decision(
            metadata=sample_metadata,
            user_preferences=user_prefs,
            stay_similarity=0.4401,
            scroll_similarity=0.3028,
            stay_count=5,
            scroll_count=5
        )
        
        print(f"DECISION: {decision}")
        print(f"REASONING: {reasoning}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure GEMINI_API_KEY is set in your .env file")
