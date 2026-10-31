#!/usr/bin/env python3
"""Generic command-line entry point for auditing tag synonyms."""

from audit_location_synonyms import (
    audit_tag_synonyms,
    create_child_from_synonym,
    find_tags_with_synonyms,
    main,
    resolve_category_name,
    retag_parent_articles,
    suggest_category_tags,
)


__all__ = [
    "audit_tag_synonyms",
    "create_child_from_synonym",
    "find_tags_with_synonyms",
    "resolve_category_name",
    "retag_parent_articles",
    "suggest_category_tags",
]


if __name__ == "__main__":
    raise SystemExit(main())
