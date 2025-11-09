
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from datetime import datetime
import isodate
import requests
from typing import Optional
from .category_id import CATEGORY_MAP
from .process_image import FastImageProcessor

load_dotenv()


class YouTubeMetadataExtractor:
    """
    Extracts and processes metadata from YouTube videos.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the metadata extractor.
        
        Args:
            api_key: YouTube Data API key. If not provided, will load from API_KEY env variable.
        """
        if api_key is None:
            api_key = os.getenv("API_KEY")
            if not api_key:
                raise ValueError("API_KEY not found in environment variables")
        
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        self._image_processor = None
    
    def get_image_processor(self):
        """Get or initialize the image processor lazily."""
        if self._image_processor is None:
            self._image_processor = FastImageProcessor()
        return self._image_processor
    
    def convert_link_to_id(self, youtube_link: str) -> Optional[str]:
        """
        Converts a YouTube link to its corresponding video ID.

        Args:
            youtube_link: The full YouTube video link.

        Returns:
            The extracted video ID or None if invalid.
        """
        if "v=" in youtube_link:
            return youtube_link.split("v=")[1].split("&")[0]
        elif "youtu.be/" in youtube_link:
            return youtube_link.split("youtu.be/")[1].split("?")[0]
        elif "shorts/" in youtube_link:
            return youtube_link.split("shorts/")[1].split("?")[0]
        else:
            return None
    
    def get_youtube_metadata(self, video_link: str) -> Optional[dict]:
        """
        Extracts metadata from a YouTube video link through the YouTube API.

        Args:
            video_link: The full YouTube video link.

        Returns:
            A dictionary containing the video's metadata or None if not found.
        """
        video_id = self.convert_link_to_id(video_link)
        
        if not video_id:
            return None

        request = self.youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=video_id
        )
        response = request.execute()

        if 'items' in response and len(response['items']) > 0:
            video_data = response['items'][0]
            snippet = video_data.get('snippet', {})
            statistics = video_data.get('statistics', {})
            content_details = video_data.get('contentDetails', {})
            
            # Store thumbnail URL temporarily for processing
            thumbnail_url = snippet.get('thumbnails', {}).get('maxres', {}).get('url')
            if not thumbnail_url:
                thumbnail_url = snippet.get('thumbnails', {}).get('high', {}).get('url')
            
            metadata = {
                'title': snippet.get('title'),
                'channelName': snippet.get('channelTitle'),
                'description': snippet.get('description'),
                'publishedAt': snippet.get('publishedAt'),
                '_thumbnail_url': thumbnail_url,  # Temporary, will be removed after processing
                'tags': snippet.get('tags'),
                'viewCount': statistics.get('viewCount'),
                'likeCount': statistics.get('likeCount'),
                'commentCount': statistics.get('commentCount'),
                'duration': content_details.get('duration'),
                'categoryId': snippet.get('categoryId')
            }
            return metadata
        else:
            return None
    
    def process_metadata(self, metadata: dict, process_image: bool = True) -> Optional[dict]:
        """
        Processes the extracted metadata for further use.

        Args:
            metadata: The metadata dictionary extracted from YouTube.
            process_image: Whether to process the thumbnail image with AI model. Default is True.

        Returns:
            Processed metadata or None if input is invalid.
        """
        if not metadata:
            return None
        
        # Convert publishedAt to a more readable format
        if metadata.get('publishedAt'):
            published_at = metadata['publishedAt']
            metadata['publishedAt'] = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ").strftime("%B %d, %Y")
        
        # Convert duration from ISO 8601 to total seconds
        if metadata.get('duration'):
            try:
                duration_obj = isodate.parse_duration(metadata['duration'])
                metadata['duration'] = int(duration_obj.total_seconds())
            except:
                metadata['duration'] = None
        
        # Get image content in bytes
        if metadata.get('_thumbnail_url'):
            try:
                response = requests.get(metadata['_thumbnail_url'])
                if response.status_code == 200:
                    thumbnail_bytes = response.content
                    
                    # Process image to get context (only if flag is True)
                    if process_image:
                        try:
                            processor = self.get_image_processor()
                            image_context = processor.get_image_context(thumbnail_bytes)
                            metadata['thumbnail_context'] = image_context.get('description')
                        except Exception as e:
                            print(f"Error processing image: {e}")
                            metadata['thumbnail_context'] = None
                    else:
                        metadata['thumbnail_context'] = None
            except:
                metadata['thumbnail_context'] = None
            
            # Remove the temporary thumbnail URL
            del metadata['_thumbnail_url']
        
        # Convert categoryId to category name using dictionary lookup
        if metadata.get('categoryId'):
            metadata['categoryName'] = CATEGORY_MAP.get(metadata['categoryId'], 'Unknown')
            # Remove categoryId since we have categoryName
            del metadata['categoryId']
        
        return metadata
    
    def get_processed_metadata(self, video_link: str, process_image: bool = True) -> Optional[dict]:
        """
        Convenience method to extract and process metadata in one call.
        
        Args:
            video_link: The full YouTube video link.
            process_image: Whether to process the thumbnail image. Default is True.
            
        Returns:
            Processed metadata dictionary or None if video not found.
        """
        metadata = self.get_youtube_metadata(video_link)
        return self.process_metadata(metadata, process_image)


if __name__ == "__main__":
    import time
    
    # Example usage
    extractor = YouTubeMetadataExtractor()
    
    # Start timing
    start_time = time.time()
    
    # Get and process metadata
    processed_metadata = extractor.get_processed_metadata(
        "https://www.youtube.com/shorts/fzp__hWRC1k",
        process_image=True
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print results
    if processed_metadata:
        print(f"\nProcessing time: {elapsed_time:.2f}s")
        print(processed_metadata)