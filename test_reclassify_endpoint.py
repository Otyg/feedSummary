#!/usr/bin/env python3
"""
Test script for the new reclassification endpoint.

This endpoint suggests tags from existing database tags only.
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"
API_BASE = f"{BASE_URL}/api/v1"

def get_first_article():
    """Get the first available article."""
    url = f"{API_BASE}/articles?limit=1"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get("items", [])
        if articles:
            return articles[0]
    return None

def test_reclassify():
    """Test the reclassification endpoint."""
    print("=" * 70)
    print("RECLASSIFICATION ENDPOINT TEST")
    print("=" * 70)
    
    # Get an article
    print("\n[1] Getting an article...")
    article = get_first_article()
    if not article:
        print("❌ No articles found")
        return False
    
    article_id = article.get("id")
    print(f"✓ Article: {article_id[:20]}...")
    print(f"  Title: {article.get('title')[:60]}...")
    
    # Call reclassification endpoint
    print(f"\n[2] Calling reclassification endpoint...")
    url = f"{API_BASE}/article/{article_id}/reclassify"
    payload = {"max_tags": 5}
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ Request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        result = response.json()
        
        print(f"✓ Request successful")
        print(f"  Suggestion count: {result.get('suggestion_count')}")
        print(f"  Current tags: {len(result.get('current_tags', []))}")
        
        # Display suggested tags
        suggested = result.get("suggested_tags", [])
        print(f"\n[3] Suggested tags ({len(suggested)} total):")
        
        if not suggested:
            print("   (No tags suggested - this is OK!)")
        else:
            for tag in suggested:
                print(f"   • {tag.get('name')} [{tag.get('category')}]")
                if tag.get('reasoning'):
                    print(f"     ↳ {tag.get('reasoning')}")
        
        print("\n" + "=" * 70)
        print("✓ TEST PASSED - Reclassification works correctly")
        print("=" * 70)
        return True
        
    except requests.exceptions.Timeout:
        print("❌ Request timed out (LLM might be slow)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_reclassify()
