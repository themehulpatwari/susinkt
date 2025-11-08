"""
Comprehensive test for the complete YouTube Shorts decision system.
Tests metadata extraction, embedding creation, and LLM decision-making.
"""

import time
from metadata import YouTubeMetadataExtractor
from embedding import VideoEmbeddingStore
from llm import VideoDecisionSystem


def test_complete_system():
    """
    Test the entire pipeline: metadata extraction -> embeddings -> LLM decision
    """
    print("="*80)
    print("COMPREHENSIVE SYSTEM TEST")
    print("="*80)
    
    # Initialize all components
    print("\n1. Initializing components...")
    metadata_extractor = YouTubeMetadataExtractor()
    embedding_store = VideoEmbeddingStore()
    decision_system = VideoDecisionSystem()
    
    # Test video links
    test_videos = [
        "https://www.youtube.com/shorts/NEa-QQzWYNI",  # Kindness video
        "https://www.youtube.com/shorts/fzp__hWRC1k",  # Cooking video
    ]
    
    # Build up some behavioral data first
    print("\n2. Building behavioral data...")
    print("-"*80)
    
    # Simulate user watching kindness videos (stay)
    for video_link in test_videos[:1]:
        print(f"\nProcessing video for STAY collection: {video_link}")
        start = time.time()
        
        metadata = metadata_extractor.get_processed_metadata(video_link, process_image=False)
        
        if metadata:
            print(f"  Title: {metadata.get('title', 'N/A')[:60]}...")
            embedding_store.add_to_stay(metadata)
            elapsed = time.time() - start
            print(f"  Processing time: {elapsed:.2f}s")
    
    # Simulate user scrolling past cooking videos
    for video_link in test_videos[1:]:
        print(f"\nProcessing video for SCROLL collection: {video_link}")
        start = time.time()
        
        metadata = metadata_extractor.get_processed_metadata(video_link, process_image=False)
        
        if metadata:
            print(f"  Title: {metadata.get('title', 'N/A')[:60]}...")
            embedding_store.add_to_scroll(metadata)
            elapsed = time.time() - start
            print(f"  Processing time: {elapsed:.2f}s")
    
    stay_count, scroll_count = embedding_store.get_collection_sizes()
    print(f"\n📊 Collection sizes: Stay={stay_count}, Scroll={scroll_count}")
    
    # Now test with a new video
    print("\n\n3. Testing decision on NEW video...")
    print("="*80)
    
    test_video = "https://www.youtube.com/shorts/dQw4w9WgXcQ"  # Replace with actual video
    user_preferences = "I want to watch wholesome, feel-good content and heartwarming stories"
    
    print(f"\nVideo: {test_video}")
    print(f"User preferences: {user_preferences}")
    print("\n" + "-"*80)
    
    # Start full pipeline timer
    total_start = time.time()
    
    # Step 1: Extract metadata
    print("\n📥 Step 1: Extracting metadata...")
    meta_start = time.time()
    metadata = metadata_extractor.get_processed_metadata(test_video, process_image=False)
    meta_time = time.time() - meta_start
    
    if not metadata:
        print("❌ Failed to extract metadata")
        return
    
    print(f"   ✓ Metadata extracted in {meta_time:.2f}s")
    print(f"   Title: {metadata.get('title', 'N/A')}")
    print(f"   Channel: {metadata.get('channelName', 'N/A')}")
    print(f"   Duration: {metadata.get('duration', 'N/A')}s")
    
    # Step 2: Calculate similarity scores
    print("\n🔍 Step 2: Calculating similarity scores...")
    embed_start = time.time()
    stay_sim, scroll_sim = embedding_store.get_similarity_scores(metadata)
    embed_time = time.time() - embed_start
    
    print(f"   ✓ Similarities calculated in {embed_time:.2f}s")
    print(f"   Stay similarity: {stay_sim:.4f}")
    print(f"   Scroll similarity: {scroll_sim:.4f}")
    
    # Step 3: Get LLM decision
    print("\n🤖 Step 3: Getting LLM decision...")
    llm_start = time.time()
    decision, reasoning = decision_system.make_decision(
        metadata=metadata,
        user_preferences=user_preferences,
        stay_similarity=stay_sim,
        scroll_similarity=scroll_sim,
        stay_count=stay_count,
        scroll_count=scroll_count
    )
    llm_time = time.time() - llm_start
    
    print(f"   ✓ Decision made in {llm_time:.2f}s")
    
    # Total time
    total_time = time.time() - total_start
    
    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\n🎯 DECISION: {decision}")
    print(f"💭 REASONING: {reasoning}")
    
    print("\n⏱️  TIMING BREAKDOWN:")
    print(f"   Metadata extraction: {meta_time:.2f}s ({(meta_time/total_time)*100:.1f}%)")
    print(f"   Embedding similarity: {embed_time:.2f}s ({(embed_time/total_time)*100:.1f}%)")
    print(f"   LLM decision: {llm_time:.2f}s ({(llm_time/total_time)*100:.1f}%)")
    print(f"   {'─'*40}")
    print(f"   TOTAL TIME: {total_time:.2f}s")
    
    print("\n" + "="*80)


def test_cold_start():
    """
    Test the system with no behavioral data (cold start scenario)
    """
    print("\n\n")
    print("="*80)
    print("COLD START TEST (No Behavioral Data)")
    print("="*80)
    
    # Initialize components
    metadata_extractor = YouTubeMetadataExtractor()
    decision_system = VideoDecisionSystem()
    
    test_video = "https://www.youtube.com/shorts/NEa-QQzWYNI"
    user_preferences = "I want funny cat videos and comedy skits"
    
    print(f"\nVideo: {test_video}")
    print(f"User preferences: {user_preferences}")
    print("-"*80)
    
    # Start timer
    total_start = time.time()
    
    # Extract metadata
    print("\n📥 Extracting metadata...")
    metadata = metadata_extractor.get_processed_metadata(test_video, process_image=False)
    
    if not metadata:
        print("❌ Failed to extract metadata")
        return
    
    print(f"   Title: {metadata.get('title', 'N/A')[:60]}...")
    
    # Make decision without behavioral data
    print("\n🤖 Making decision (cold start - no behavioral data)...")
    decision, reasoning = decision_system.make_decision(
        metadata=metadata,
        user_preferences=user_preferences,
        stay_similarity=None,
        scroll_similarity=None,
        stay_count=0,
        scroll_count=0
    )
    
    total_time = time.time() - total_start
    
    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\n🎯 DECISION: {decision}")
    print(f"💭 REASONING: {reasoning}")
    print(f"\n⏱️  TOTAL TIME: {total_time:.2f}s")
    print("\n" + "="*80)


def main():
    """Run all tests"""
    try:
        # Test 1: Cold start (no behavioral data)
        test_cold_start()
        
        # Test 2: Complete system with behavioral data
        test_complete_system()
        
        print("\n\n✅ ALL TESTS COMPLETED!")
        
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
