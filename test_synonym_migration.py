#!/usr/bin/env python3
"""
Test script for synonym tag migration feature.

When a tag is set as a synonym of another tag:
1. All articles using the synonym tag get the main tag instead
2. The synonym tag is deleted from the database
"""

import requests
import json
import time
import sys

BASE_URL = "http://127.0.0.1:5000"
API_BASE = f"{BASE_URL}/api/v1"

def log(msg, level="INFO"):
    print(f"[{level}] {msg}", flush=True)

def get_first_article():
    """Get the first available article."""
    url = f"{API_BASE}/articles?limit=1"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get("items", [])
        if articles:
            article = articles[0]
            log(f"Using existing article: {article.get('id')} - {article.get('title')}")
            return article
        else:
            log("No articles available in the system", "ERROR")
            return None
    else:
        log(f"Failed to get articles: {response.status_code}", "ERROR")
        return None

def create_tag(name, category="GENERAL", description="", synonyms=None):
    """Create a tag."""
    url = f"{API_BASE}/tags"
    data = {
        "name": name,
        "category": category,
        "description": description,
        "synonyms": synonyms or []
    }
    response = requests.post(url, json=data)
    if response.status_code == 201:
        tag = response.json().get("tag", {})
        log(f"Created tag: {tag.get('id')} - {name}")
        return tag
    else:
        log(f"Failed to create tag {name}: {response.status_code} - {response.text}", "ERROR")
        return None

def tag_article(article_id, tag_id):
    """Add a tag to an article."""
    url = f"{API_BASE}/article/{article_id}/tags"  # Singular "article"
    data = {"tag_id": tag_id}  # Singular tag_id
    response = requests.post(url, json=data)
    if response.status_code == 200:
        log(f"Tagged article {article_id} with tag {tag_id}")
        return True
    else:
        log(f"Failed to tag article: {response.status_code} - {response.text}", "ERROR")
        return False

def get_article_tags(article_id):
    """Get all tags for an article."""
    url = f"{API_BASE}/article/{article_id}/tags"  # Singular "article"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("tags", [])
    else:
        log(f"Failed to get article tags: {response.status_code}", "ERROR")
        return []

def update_tag_with_synonyms(tag_id, name, category, description, synonyms):
    """Update a tag with new synonyms."""
    url = f"{API_BASE}/tags/{tag_id}"
    data = {
        "name": name,
        "category": category,
        "description": description,
        "synonyms": synonyms
    }
    response = requests.put(url, json=data)
    if response.status_code == 200:
        log(f"Updated tag {tag_id} with synonyms: {synonyms}")
        return response.json().get("tag")
    else:
        log(f"Failed to update tag: {response.status_code} - {response.text}", "ERROR")
        return None

def get_tag(tag_id):
    """Get tag details."""
    url = f"{API_BASE}/tags/{tag_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("tag")
    else:
        return None

def get_all_tags():
    """Get all tags."""
    url = f"{API_BASE}/tags"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("tags", [])
    else:
        return []

def run_test():
    """Run the complete migration test."""
    # Use unique tag names with timestamp
    test_id = int(time.time() * 1000) % 1000000
    
    log("=" * 60)
    log("SYNONYM TAG MIGRATION TEST")
    log("=" * 60)
    
    # Test 1: Create two tags
    log("\n[TEST 1] Creating tags...")
    main_tag_name = f"main_tag_{test_id}"
    synonym_tag_name = f"synonym_tag_{test_id}"
    
    tag_main = create_tag(main_tag_name, "GENERAL", "Main tag for testing")
    tag_synonym = create_tag(synonym_tag_name, "GENERAL", "Will become synonym of main_tag")
    
    if not tag_main or not tag_synonym:
        log("Failed to create tags", "ERROR")
        return False
    
    main_id = tag_main.get("id")
    synonym_id = tag_synonym.get("id")
    log(f"✓ Created main_tag (ID: {main_id}) and synonym_tag (ID: {synonym_id})")
    
    # Test 2: Create an article and tag it with the synonym tag
    log("\n[TEST 2] Getting an article and tagging it...")
    article = get_first_article()
    if not article:
        log("Failed to get article", "ERROR")
        return False
    
    article_id = article.get("id")
    if not tag_article(article_id, synonym_id):
        log("Failed to tag article", "ERROR")
        return False
    
    # Verify article has the synonym tag
    tags = get_article_tags(article_id)
    if not any(t.get("id") == synonym_id for t in tags):
        log(f"Article does not have synonym tag! Tags: {tags}", "ERROR")
        return False
    
    log(f"✓ Article {article_id} tagged with synonym_tag")
    
    # Test 3: Update main_tag to have synonym_tag as a synonym
    log("\n[TEST 3] Setting synonym_tag as synonym of main_tag...")
    updated = update_tag_with_synonyms(
        main_id,
        tag_main.get("name"),
        tag_main.get("category"),
        tag_main.get("description"),
        [synonym_tag_name]  # This should trigger migration
    )
    
    if not updated:
        log("Failed to update tag with synonyms", "ERROR")
        return False
    
    log("✓ Updated main_tag with synonym_tag as synonym")
    
    # Wait a moment for processing
    time.sleep(1)
    
    # Test 4: Verify article now has main_tag instead of synonym_tag
    log("\n[TEST 4] Verifying article tags after migration...")
    tags_after = get_article_tags(article_id)
    tag_ids = [t.get("id") for t in tags_after]
    
    has_main = main_id in tag_ids
    has_synonym = synonym_id in tag_ids
    
    log(f"Article tags after migration: {[t.get('name') for t in tags_after]}")
    
    if not has_main:
        log(f"✗ Article does NOT have main_tag! Tag IDs: {tag_ids}", "ERROR")
        return False
    
    if has_synonym:
        log(f"✗ Article still has synonym_tag! It should have been removed.", "ERROR")
        return False
    
    log(f"✓ Article now has main_tag (ID: {main_id})")
    log(f"✓ Article no longer has synonym_tag")
    
    # Test 5: Verify synonym_tag is deleted from database
    log("\n[TEST 5] Verifying synonym_tag is deleted...")
    all_tags = get_all_tags()
    tag_ids_in_db = [t.get("id") for t in all_tags]
    tag_names_in_db = [t.get("name") for t in all_tags]
    
    if synonym_id in tag_ids_in_db:
        log(f"✗ synonym_tag still exists in database! All tags: {tag_names_in_db}", "ERROR")
        return False
    
    log(f"✓ synonym_tag has been deleted from database")
    
    # Test 6: Verify main_tag has synonym in its synonyms list
    log("\n[TEST 6] Verifying main_tag has synonym_tag as synonym...")
    main_tag_final = get_tag(main_id)
    synonyms = main_tag_final.get("synonyms", [])
    
    if synonym_tag_name not in [s.lower() for s in synonyms]:
        log(f"✗ main_tag does not have synonym_tag as synonym! Synonyms: {synonyms}", "ERROR")
        return False
    
    log(f"✓ main_tag has synonym_tag as synonym: {synonyms}")
    
    log("\n" + "=" * 60)
    log("ALL TESTS PASSED ✓")
    log("=" * 60)
    return True

if __name__ == "__main__":
    try:
        success = run_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        log(f"Test failed with exception: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
