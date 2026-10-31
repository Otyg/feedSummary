#!/usr/bin/env python3
"""
Test script to verify tag reasoning flows through the system correctly.
"""
import sys
import json
sys.path.insert(0, '/home/otyg/projects/feedSummaryBase/src')

from feedsummary_core.summarizer.tagging import TagManager

# Test 1: Verify extract_tags_from_llm_response includes reasoning
print("Test 1: extract_tags_from_llm_response with per-tag reasoning")
print("=" * 60)

tag_manager = TagManager()

# Simulate LLM response with per-tag reasoning
llm_response = """{
    "tags": [
        {"tag": "apple", "type": "NAMED_ENTITY", "reasoning": "Apple Inc. is mentioned as a company in security context"},
        {"tag": "data breach", "type": "CATEGORY", "reasoning": "Article discusses unauthorized access to customer data"},
        {"tag": "ransomware", "type": "CATEGORY", "reasoning": "Ransomware attack mentioned multiple times"}
    ]
}"""

extracted = tag_manager.extract_tags_from_llm_response(llm_response)
print(f"Extracted {len(extracted)} tags:")
for tag in extracted:
    print(f"  - {tag['name']:<20} (type: {tag['type']:<15}) reasoning: {tag.get('reasoning', 'N/A')}")

# Verify reasoning is present
assert all('reasoning' in tag for tag in extracted), "Not all tags have reasoning field!"
print("✓ All tags have reasoning field\n")

# Test 2: Verify select_tags_for_article preserves reasoning
print("Test 2: select_tags_for_article preserves reasoning")
print("=" * 60)

article_id = "test_article_123"
selected = tag_manager.select_tags_for_article(article_id, extracted, allow_new_tags=True)

print(f"Selected {len(selected)} tags:")
for tag in selected:
    print(f"  - {tag['name']:<20} (id: {tag.get('id', 'N/A'):<5}) reasoning: {tag.get('reasoning', 'N/A')}")

# Verify reasoning is preserved
assert all('reasoning' in tag for tag in selected), "Not all selected tags have reasoning field!"
print("✓ All selected tags have reasoning field\n")

# Test 3: Verify backwards compatibility with string tags
print("Test 3: Backwards compatibility with string tags")
print("=" * 60)

string_tags = ["malware", "exploit", "vulnerability"]
selected_strings = tag_manager.select_tags_for_article(article_id, string_tags, allow_new_tags=True)

print(f"Selected {len(selected_strings)} tags from strings:")
for tag in selected_strings:
    print(f"  - {tag['name']:<20} reasoning: '{tag.get('reasoning', '')}'")

# Verify empty reasoning for string tags
assert all(tag.get('reasoning', '') == '' for tag in selected_strings), "String tags should have empty reasoning!"
print("✓ String tags have empty reasoning field (as expected)\n")

print("=" * 60)
print("✓ All tests passed! Reasoning flow is working correctly.")
