#!/bin/bash
# Test script for synonym feature

set -e

BASE_URL="http://localhost:5000"
TEST_TAG_ID=""

echo "🧪 Testing Synonym Feature"
echo "========================="

# Test 1: Create a tag with synonyms
echo -e "\n1️⃣  Creating tag 'xss_attack' with synonyms..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/tags" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "xss_attack",
    "category": "THREAT",
    "description": "Cross-site scripting attack",
    "synonyms": ["xss", "cross_site_scripting", "cxss"]
  }')

TEST_TAG_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('tag', {}).get('id', ''))" 2>/dev/null)

if [ -z "$TEST_TAG_ID" ]; then
  # Tag already exists, find it
  echo "  (Tag already exists, searching for it...)"
  curl -s "$BASE_URL/api/v1/tags" | python3 -c "import sys, json; tags=json.load(sys.stdin)['tags']; xss=[t for t in tags if t.get('name')=='xss_attack']; print(xss[0]['id'] if xss else '')" > /tmp/tag_id.txt
  TEST_TAG_ID=$(cat /tmp/tag_id.txt)
fi

if [ -z "$TEST_TAG_ID" ]; then
  echo "  ❌ ERROR: Could not create or find tag"
  exit 1
fi

echo "  ✅ Tag created with ID: $TEST_TAG_ID"

# Test 2: Retrieve tag and verify synonyms
echo -e "\n2️⃣  Retrieving tag and verifying synonyms..."
TAG_JSON=$(curl -s "$BASE_URL/api/v1/tags/$TEST_TAG_ID")
SYNONYMS=$(echo "$TAG_JSON" | python3 -c "import sys, json; data=json.load(sys.stdin); print(json.dumps(data.get('tag', {}).get('synonyms', [])))")

echo "  Synonyms: $SYNONYMS"

EXPECTED="[\"xss\", \"cross_site_scripting\", \"cxss\"]"
if [[ "$SYNONYMS" == *"xss"* ]] && [[ "$SYNONYMS" == *"cross_site_scripting"* ]]; then
  echo "  ✅ Synonyms retrieved correctly"
else
  echo "  ❌ ERROR: Synonyms not returned correctly"
  exit 1
fi

# Test 3: Update tag with new synonyms
echo -e "\n3️⃣  Updating tag with additional synonym..."
RESPONSE=$(curl -s -X PUT "$BASE_URL/api/v1/tags/$TEST_TAG_ID" \
  -H 'Content-Type: application/json' \
  -d '{
    "synonyms": ["xss", "cross_site_scripting", "cxss", "xss_vulnerability"]
  }')

SYNONYMS=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(json.dumps(data.get('tag', {}).get('synonyms', [])))")

if [[ "$SYNONYMS" == *"xss_vulnerability"* ]]; then
  echo "  ✅ Tag updated with new synonym"
else
  echo "  ❌ ERROR: Synonym not updated"
  exit 1
fi

# Test 4: Test synonym matching (if tagging is available)
echo -e "\n4️⃣  Testing synonym matching in tagging..."
echo "  (This requires running the Python test script)"
PYTHONPATH=/home/otyg/projects/feedSummary:/home/otyg/projects/feedSummaryBase/src \
  /home/otyg/projects/air-marine/.venv/bin/python3 -c "
import sys
sys.path.insert(0, '/home/otyg/projects/feedSummary')
sys.path.insert(0, '/home/otyg/projects/feedSummaryBase/src')

from feedsummary_core.persistence import create_store
from feedsummary_core.summarizer.tagging import TagManager

cfg = {'type': 'tinydb', 'path': '/home/otyg/projects/feedSummary/data/news_docs.json'}
store = create_store(cfg)
tag_manager = TagManager(store=store, llm_client=None)

# Test matching synonym
matches = tag_manager._find_similar_existing_tags('xss')
if matches and matches[0].get('name') == 'xss_attack':
    print('  ✅ Synonym \"xss\" matched to main tag \"xss_attack\"')
else:
    print('  ❌ Synonym matching failed')
    sys.exit(1)
" || exit 1

echo -e "\n✅ All synonym feature tests passed!"
echo "================================================================"
