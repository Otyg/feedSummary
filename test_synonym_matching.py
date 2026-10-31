#!/usr/bin/env python3
"""
Test script for synonym tag matching functionality.
"""

import sys
import os

# Add paths
sys.path.insert(0, '/home/otyg/projects/feedSummary')
sys.path.insert(0, '/home/otyg/projects/feedSummaryBase/src')

from feedsummary_core.persistence import create_store
from feedsummary_core.summarizer.tagging import TagManager

def test_synonym_matching():
    """Test that synonym tags are matched to main tags."""
    
    # Initialize store and tag manager (without LLM client for this test)
    cfg = {
        'type': 'tinydb',
        'path': '/home/otyg/projects/feedSummary/data/news_docs.json'
    }
    store = create_store(cfg)
    tag_manager = TagManager(store=store, llm_client=None)
    
    # Get the test tag we created
    all_tags = store.get_all_tags()
    test_tag = next((t for t in all_tags if t.get('name') == 'sql_injection_test'), None)
    
    if not test_tag:
        print("ERROR: Test tag 'sql_injection_test' not found!")
        return False
    
    print(f"✓ Found test tag: {test_tag['name']}")
    print(f"  Synonyms: {test_tag.get('synonyms', [])}")
    
    # Test 1: Match a synonym to the main tag
    print("\nTest 1: Matching synonym 'sqli' to main tag...")
    matches = tag_manager._find_similar_existing_tags('sqli')
    
    if not matches:
        print("  ERROR: No matches found!")
        return False
    
    matched_tag = matches[0]
    if matched_tag.get('name') == 'sql_injection_test':
        print(f"  ✓ SUCCESS: Synonym 'sqli' matched to main tag '{matched_tag['name']}'")
    else:
        print(f"  ERROR: Synonym matched to '{matched_tag['name']}' instead of 'sql_injection_test'")
        return False
    
    # Test 2: Match another synonym
    print("\nTest 2: Matching synonym 'sql_inj' to main tag...")
    matches = tag_manager._find_similar_existing_tags('sql_inj')
    
    if not matches:
        print("  ERROR: No matches found!")
        return False
    
    matched_tag = matches[0]
    if matched_tag.get('name') == 'sql_injection_test':
        print(f"  ✓ SUCCESS: Synonym 'sql_inj' matched to main tag '{matched_tag['name']}'")
    else:
        print(f"  ERROR: Synonym matched to '{matched_tag['name']}' instead of 'sql_injection_test'")
        return False
    
    # Test 3: Verify main tag name still matches
    print("\nTest 3: Matching main tag name directly...")
    matches = tag_manager._find_similar_existing_tags('sql_injection_test')
    
    if not matches:
        print("  ERROR: No matches found!")
        return False
    
    matched_tag = matches[0]
    if matched_tag.get('name') == 'sql_injection_test':
        print(f"  ✓ SUCCESS: Main tag name matched exactly")
    else:
        print(f"  ERROR: Main tag name did not match")
        return False
    
    print("\n✅ All synonym matching tests passed!")
    return True


if __name__ == '__main__':
    success = test_synonym_matching()
    sys.exit(0 if success else 1)
