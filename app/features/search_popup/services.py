import logging
import math
from typing import List, Dict, Optional
import numpy as np

from app.features.search.services.embedding_service import embedding_service
from app.shared.qdrant.service import qdrant_service
from app.features.recommendation.services.recommendation_service import get_mysql_connection
from app.shared.utils.query_validator import is_malicious_or_invalid_query

logger = logging.getLogger(__name__)

# Simple in-memory cache for category name -> category ID mapping
category_cache: Dict[str, int] = {}

def get_category_id_by_name(category_name: str) -> Optional[int]:
    """Looks up category ID by name, caching results to minimize MySQL queries."""
    if not category_name:
        return None
    
    normalized_name = category_name.strip().lower()
    if normalized_name in category_cache:
        return category_cache[normalized_name]
    
    conn = None
    try:
        conn = get_mysql_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT id FROM categories WHERE LOWER(name) = %s AND deleted_at IS NULL LIMIT 1", (normalized_name,))
            row = cursor.fetchone()
            if row:
                cat_id = int(row["id"])
                category_cache[normalized_name] = cat_id
                return cat_id
    except Exception as e:
        logger.error(f"Error querying category ID for '{category_name}': {e}")
    finally:
        if conn:
            conn.close()
    return None


class SearchPopupService:

    def get_semantic_suggestions(self, query: str, user_id: Optional[str] = None, limit: int = 8) -> List[Dict]:
        """
        Retrieves autocomplete suggestions:
        1. Prefix match from MySQL `search_logs`
        2. Semantic similarity match from Qdrant
        3. Merge, deduplicate, and score using the simplified AI formula
        """
        if is_malicious_or_invalid_query(query):
            logger.warning(f"Blocked malicious autocomplete query: '{query}'")
            return []
        suggestions_map: Dict[str, Dict] = {}
        clean_query = query.strip().lower()

        # Source 1: Prefix match from search logs
        conn = None
        db_keywords = []
        try:
            conn = get_mysql_connection()
            with conn.cursor() as cursor:
                # Get most frequent searches starting with our prefix
                cursor.execute(
                    """
                    SELECT query_text, COUNT(*) as cnt 
                    FROM search_logs 
                    WHERE LOWER(query_text) LIKE %s 
                    GROUP BY query_text 
                    ORDER BY cnt DESC 
                    LIMIT 15
                    """,
                    (f"{clean_query}%",)
                )
                db_keywords = cursor.fetchall()
        except Exception as e:
            logger.error(f"Error reading prefix matches from search_logs: {e}")
        finally:
            if conn:
                conn.close()

        # Map prefix keywords as candidates
        for row in db_keywords:
            kw = row["query_text"].strip()
            cnt = int(row["cnt"])
            suggestions_map[kw.lower()] = {
                "keyword": kw,
                "type": "HISTORY" if user_id else "TRENDING",
                "semantic_score": 0.85, # Base similarity for exact prefix matches
                "freq": cnt,
                "category_name": None
            }

        # Source 2: Semantic product matching from Qdrant
        try:
            query_vector = embedding_service.embed_query(query)
            qdrant_results = qdrant_service.search(
                query_vector=query_vector,
                top_k=20
            )
            for res in qdrant_results:
                title = res.get("title", "").strip()
                # Clean up title if it's too long (limit to 40 characters for search suggestions)
                if len(title) > 40:
                    title = title[:37] + "..."
                
                title_lower = title.lower()
                # Deduplicate and prioritize Qdrant's actual semantic score
                if title_lower not in suggestions_map:
                    suggestions_map[title_lower] = {
                        "keyword": title,
                        "type": "AI_SUGGESTION",
                        "semantic_score": float(res.get("score", 0.7)),
                        "freq": 1, # Default count
                        "category_name": res.get("category")
                    }
                else:
                    # If already present via prefix, update semantic score & category
                    suggestions_map[title_lower]["semantic_score"] = float(res.get("score", 0.75))
                    suggestions_map[title_lower]["category_name"] = res.get("category")
        except Exception as e:
            logger.error(f"Error querying semantic suggestions from Qdrant: {e}")

        # AI Ranking step using: Score = 0.6 * SemanticScore + 0.4 * log10(1 + Freq)
        scored_candidates = []
        for key, val in suggestions_map.items():
            sem_score = val["semantic_score"]
            freq = val["freq"]
            
            # Simple AI formula
            ai_score = 0.6 * sem_score + 0.4 * math.log10(1 + freq)
            
            # Category ID lookup
            cat_id = get_category_id_by_name(val["category_name"]) if val["category_name"] else None
            
            scored_candidates.append({
                "keyword": val["keyword"],
                "type": val["type"],
                "categoryId": cat_id,
                "score": ai_score
            })

        # Sort by final score descending and limit results
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        return scored_candidates[:limit]

    def get_related_queries(self, product_uuid: str, limit: int = 5) -> List[str]:
        """
        Finds related search queries by searching Qdrant for similar products
        and returning their title/category details.
        """
        try:
            # 1. Fetch source vector
            source_vector = qdrant_service.get_product_vector(product_uuid)
            if not source_vector:
                return []
            
            # 2. Search similar products in Qdrant
            results = qdrant_service.search(
                query_vector=source_vector,
                top_k=limit + 2,
                exclude_ids=[product_uuid]
            )
            
            # 3. Extract unique category terms or clean short names as related query suggestions
            related_terms = []
            seen = set()
            for r in results:
                cat = r.get("category", "")
                if cat:
                    # Get leaf category name
                    leaf = cat.split(">")[-1].strip().lower()
                    if leaf and leaf not in seen:
                        seen.add(leaf)
                        related_terms.append(leaf.capitalize())
                
                title = r.get("title", "")
                if title:
                    words = title.split()
                    # Grab a short 2-3 word phrase from the title
                    short_phrase = " ".join(words[:3]).strip().lower()
                    if short_phrase and short_phrase not in seen and len(short_phrase) > 5:
                        seen.add(short_phrase)
                        related_terms.append(short_phrase.capitalize())

            return related_terms[:limit]
        except Exception as e:
            logger.error(f"Error fetching related queries for '{product_uuid}': {e}")
            return []


search_popup_service = SearchPopupService()
