### import os
import pymongo
from voyageai import Client
import logging
from pprint import pprint
import time
import decimal
from decimal import Decimal

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def hybrid_mongodb_search(query, vector_limit=15, text_limit=15):
    try:
        # Initialize Voyage AI client
        voyage_api_key = <CODE>
        voyage_client = Client(api_key=voyage_api_key)
        
        # Connect to MongoDB
        mongo_uri = <CODE>
        mongo_client = pymongo.MongoClient(mongo_uri)
        db = mongo_client["amazon_data"]
        collection = db["product"]
        
        # Generate embedding for the query
        logger.info(f"Generating embedding for query: '{query}'")
        query_result = voyage_client.embed(
            texts=[query],
            model="voyage-3",
            input_type="query"  # Using query input type for search queries
        )
        query_embedding = query_result.embeddings[0]
        
        # Build the aggregation pipeline that combines vector and text search
        logger.info("Executing hybrid search in MongoDB using unionWith")
        
        pipeline = [
    # First part: Vector search
    {
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": query_embedding,
            "path": "embeddings",
            "numCandidates": vector_limit * 5,
            "limit": vector_limit
        }
    },
    {
        "$addFields": {
            "search_type": "vector",
            "vector_score": {"$meta": "vectorSearchScore"}
        }
    },
    
    # Union with Atlas Search
    {
        "$unionWith": {
            "coll": "product",  
            "pipeline": [
                {
                    "$search": {
                        "index": "description",  
                        "text": {
                            "query": query,
                            "path": "Description"
                        }
                    }
                },
                {
                    "$addFields": {
                        "search_type": "text",
                        "text_score": {"$meta": "searchScore"}  # Atlas Search uses searchScore
                    }
                },
                {
                    "$sort": {"text_score": -1}
                },
                {
                    "$limit": text_limit
                }
            ]
        }
    },
    
    
    {
        "$project": {
            "_id": 1,
            "Product Name": 1,
            "Description": 1,
            "price": 1,
            "search_type": 1,
            "vector_score": 1,
            "text_score": 1
        }
    }
]
        
        raw_results = list(collection.aggregate(pipeline))
        print(raw_results)
        
        results_by_id = {}
        
        for result in raw_results:
            doc_id = str(result["_id"])
            
            if doc_id not in results_by_id:
                results_by_id[doc_id] = result
            else:
                existing = results_by_id[doc_id]
                
                if result["search_type"] != existing["search_type"]:
                    existing["search_type"] = "both"
                
                if "vector_score" in result and result["vector_score"] > 0:
                    existing["vector_score"] = result["vector_score"]
                    
                if "text_score" in result and result["text_score"] > 0:
                    existing["text_score"] = result["text_score"]
        
        # Convert the dictionary back to a list
        merged_results = list(results_by_id.values())
        
        logger.info(f"Found {len(merged_results)} total results in hybrid search")
        
        # Extract vector and text results for separate display
        vector_results = [r.copy() for r in raw_results if r["search_type"] == "vector"]
        text_results = [r.copy() for r in raw_results if r["search_type"] == "text"]
        
        # Sort vector results by vector_score in descending order
        vector_results.sort(key=lambda x: x.get("vector_score", 0), reverse=True)
        
        # Sort text results by text_score in descending order
        text_results.sort(key=lambda x: x.get("text_score", 0), reverse=True)
        
        logger.info(f"Vector results: {len(vector_results)}, Text results: {len(text_results)}")
        
        return merged_results, vector_results[:vector_limit], text_results[:text_limit]
        
    except Exception as e:
        logger.error(f"Error during hybrid MongoDB search: {str(e)}")
        return [], [], []
    finally:
        if 'mongo_client' in locals():
            mongo_client.close()

def analyze_query(query):
    words = query.split()
    length = len(words)

    specific_terms = {"model", "brand", "exact", "size", "only", "type", "color"}
    semantic_terms = {
        "like", "recommend", "best", "alternative", "similar", "top",
        "related", "suggested", "equivalent", "advanced"
    }

    spec_score = sum(any(term in w.lower() for term in specific_terms) for w in words)
    sem_score = sum(any(term in w.lower() for term in semantic_terms) for w in words)

    if spec_score > sem_score or length <= 2:
        query_type = "keyword"
    else:
        query_type = "semantic"

    return {
       
       
        "probable_type": query_type
    }

def custom_rerank_with_rrf(vector_results, text_results, top_k=10, k=60, vector_weight=1.0, text_weight=1.0):
   
    # Track scores and document details
    doc_scores = {}
    all_docs = {}
    calculation_details = {}
    
    # Track which documents appear in each result set
    vector_ids = set()
    text_ids = set()
    
    # Process both result sets in the same loop structure
    for source, results, weight, id_set in [
        ("vector", vector_results, vector_weight, vector_ids),
        ("text", text_results, text_weight, text_ids)
    ]:
        for rank, result in enumerate(results, 1):
            doc_id = str(result["_id"])
            id_set.add(doc_id)
            
            # Calculate RRF contribution for this item: weight * 1/(rank + k)
            rrf_base = 1 / (rank + k)
            rrf_contribution = weight * rrf_base
            
            if doc_id in doc_scores:
                # Document already exists in the other result set
                prev_score = doc_scores[doc_id]
                doc_scores[doc_id] += rrf_contribution
                all_docs[doc_id]["search_type"] = "both"
                
                # Update source-specific details
                if source == "vector":
                    all_docs[doc_id]["vector_score"] = result.get("vector_score", 0)
                    calculation_details[doc_id]["vector_rank"] = rank
                    calculation_details[doc_id]["vector_score"] = result.get("vector_score", 0)
                    calculation_details[doc_id]["vector_base_contribution"] = rrf_base
                    calculation_details[doc_id]["vector_weighted_contribution"] = rrf_contribution
                    calculation_details[doc_id]["final_rrf"] = prev_score + rrf_contribution
                else:  # source == "text"
                    all_docs[doc_id]["text_score"] = result.get("text_score", 0)
                    calculation_details[doc_id]["text_rank"] = rank
                    calculation_details[doc_id]["text_score"] = result.get("text_score", 0)
                    calculation_details[doc_id]["text_base_contribution"] = rrf_base
                    calculation_details[doc_id]["text_weighted_contribution"] = rrf_contribution
                    calculation_details[doc_id]["final_rrf"] = prev_score + rrf_contribution
            else:
                # New document - first appearance in either result set
                doc_scores[doc_id] = rrf_contribution
                all_docs[doc_id] = result.copy()
                all_docs[doc_id]["search_type"] = source
                
                # Initialize calculation details with appropriate defaults
                calculation_details[doc_id] = {
                    "title": result.get("Product Name", "Unknown"),
                    "vector_rank": rank if source == "vector" else None,
                    "vector_score": result.get("vector_score", 0) if source == "vector" else None,
                    "vector_base_contribution": rrf_base if source == "vector" else 0,
                    "vector_weighted_contribution": rrf_contribution if source == "vector" else 0,
                    "text_rank": rank if source == "text" else None,
                    "text_score": result.get("text_score", 0) if source == "text" else None,
                    "text_base_contribution": rrf_base if source == "text" else 0,
                    "text_weighted_contribution": rrf_contribution if source == "text" else 0,
                    "final_rrf": rrf_contribution,
                    "vector_weight": vector_weight,
                    "text_weight": text_weight
                }
    
    # Add RRF scores and mark docs in both sets
    for doc_id, doc in all_docs.items():
        doc["rrf_score"] = doc_scores[doc_id]
        doc["rrf_calculation"] = calculation_details[doc_id]
        doc["in_both_sets"] = doc_id in vector_ids and doc_id in text_ids
    
    # Sort by RRF score (descending) and return top_k results
    combined_results = sorted(all_docs.values(), key=lambda x: x.get("rrf_score", 0), reverse=True)
    
    return combined_results[:top_k]
def display_search_results(results, title):
    print("\n" + "=" * 80)
    print(f"{title} ({len(results)} results)")
    print("=" * 80)
    
    if results and "vector_weight" in results[0] and "text_weight" in results[0]:
        vector_weight = results[0]["vector_weight"]
        text_weight = results[0]["text_weight"]
        print(f"Using weights: Vector = {vector_weight}, Text = {text_weight}")
    
    if title == "VoyageAI Reranked Results" and results:
        if "enhanced_query_used" in results[0]:
            print(f"Enhanced query used: '{results[0]['enhanced_query_used']}'")
        if "descriptions_only" in results[0] and results[0]["descriptions_only"]:
            print("Reranking used descriptions only (no product names or additional context)")
    
    search_types = {"vector": 0, "text": 0, "both": 0}
    for r in results:
        search_type = r.get("search_type", "unknown")
        search_types[search_type] = search_types.get(search_type, 0) + 1
    
    # Display result type breakdown
    if title in ["RRF Combined Results", "VoyageAI Reranked Results"]:
        print(f"Breakdown: {search_types.get('vector', 0)} vector only, {search_types.get('text', 0)} text only, {search_types.get('both', 0)} in both")
    
    
    if results:
        for i, product in enumerate(results, 1):
            print(f"Result #{i}")
            
            # Get search type with clear labeling
            search_type = product.get("search_type", "unknown")
            search_type_display = {
                "vector": "Vector only", 
                "text": "Text only", 
                "both": "Both vector and text"
            }.get(search_type, search_type)
            
            if title == "Vector Search Results":
                # Only show vector score for vector results
                print(f"Vector Score: {product.get('vector_score', 'N/A'):.4f}")
                print(f"Title: {product.get('Product Name', 'No title')}")
                
            elif title == "Full Text Search Results":
                print(f"Text Score: {product.get('text_score', 'N/A'):.4f}")
                print(f"Title: {product.get('Product Name', 'No title')}")
                
            elif title == "VoyageAI Reranked Results":
                # For VoyageAI reranked results - show all elements
                print(f"Search Type: {search_type_display}")
                print(f"Rerank Score: {product.get('rerank_score', 'N/A'):.4f}")
                print(f"Query Type: {product.get('query_type', 'unknown')}")
                
                if "keyword_matches" in product:
                    matches = product["keyword_matches"]
                    if matches.get("total_matches", 0) > 0:
                        print(f"Keyword Matches: {matches.get('total_matches', 0)} total")
                        if matches.get("name_match_count", 0) > 0:
                            print(f"  - Name matches: {matches.get('name_match_count', 0)} ({', '.join(matches.get('name_matches', []))})")
                        if matches.get("desc_match_count", 0) > 0:
                            print(f"  - Description matches: {matches.get('desc_match_count', 0)} ({', '.join(matches.get('desc_matches', []))})")
                
                if "rrf_score" in product:
                    print(f"RRF Score: {product.get('rrf_score', 'N/A'):.4f}")
                    
                    # Show RRF calculation if available
                    if "rrf_calculation" in product:
                        details = product["rrf_calculation"]
                        vector_weight = details.get("vector_weight", 1.0)
                        text_weight = details.get("text_weight", 1.0)
                        k = 60  # Default k value
                        
                        if search_type == "both":
                            v_base = details.get("vector_base_contribution", 0)
                            v_contrib = details.get("vector_weighted_contribution", 0)
                            t_base = details.get("text_base_contribution", 0)
                            t_contrib = details.get("text_weighted_contribution", 0)
                            print(f"RRF Calculation: Vector({vector_weight} * 1/({details.get('vector_rank', 'N/A')}+{k})={vector_weight} * {v_base:.4f}={v_contrib:.4f}) + Text({text_weight} * 1/({details.get('text_rank', 'N/A')}+{k})={text_weight} * {t_base:.4f}={t_contrib:.4f})")
                        elif search_type == "vector":
                            v_base = details.get("vector_base_contribution", 0)
                            v_contrib = details.get("vector_weighted_contribution", 0)
                            print(f"RRF Calculation: Vector only - {vector_weight} * 1/({details.get('vector_rank', 'N/A')}+{k})={vector_weight} * {v_base:.4f}={v_contrib:.4f}")
                        else:
                            t_base = details.get("text_base_contribution", 0)
                            t_contrib = details.get("text_weighted_contribution", 0)
                            print(f"RRF Calculation: Text only - {text_weight} * 1/({details.get('text_rank', 'N/A')}+{k})={text_weight} * {t_base:.4f}={t_contrib:.4f}")
                
                if search_type in ["vector", "both"]:
                    print(f"Vector Score: {product.get('vector_score', 'N/A'):.4f}")
                
                if search_type in ["text", "both"]:
                    print(f"Text Score: {product.get('text_score', 'N/A'):.4f}")
                
                print(f"Title: {product.get('Product Name', 'No title')}")
                print(f"Price: {product.get('price', 'N/A')}")
                
            elif title == "RRF Combined Results":
                # For RRF combined results - show all elements with calculations
                print(f"Search Type: {search_type_display}")
                print(f"RRF Score: {product.get('rrf_score', 'N/A'):.4f}")
                
                # Show RRF calculation if available
                if "rrf_calculation" in product:
                    details = product["rrf_calculation"]
                    vector_weight = details.get("vector_weight", 1.0)
                    text_weight = details.get("text_weight", 1.0)
                    k = 60  # Default k value
                    
                    if search_type == "both":
                        v_base = details.get("vector_base_contribution", 0)
                        v_contrib = details.get("vector_weighted_contribution", 0)
                        t_base = details.get("text_base_contribution", 0)
                        t_contrib = details.get("text_weighted_contribution", 0)
                        print(f"RRF Calculation: Vector({vector_weight} * 1/({details.get('vector_rank', 'N/A')}+{k})={vector_weight} * {v_base:.4f}={v_contrib:.4f}) + Text({text_weight} * 1/({details.get('text_rank', 'N/A')}+{k})={text_weight} * {t_base:.4f}={t_contrib:.4f})")
                    elif search_type == "vector":
                        v_base = details.get("vector_base_contribution", 0)
                        v_contrib = details.get("vector_weighted_contribution", 0)
                        print(f"RRF Calculation: Vector only - {vector_weight} * 1/({details.get('vector_rank', 'N/A')}+{k})={vector_weight} * {v_base:.4f}={v_contrib:.4f}")
                    else:
                        t_base = details.get("text_base_contribution", 0)
                        t_contrib = details.get("text_weighted_contribution", 0)
                        print(f"RRF Calculation: Text only - {text_weight} * 1/({details.get('text_rank', 'N/A')}+{k})={text_weight} * {t_base:.4f}={t_contrib:.4f}")
                
                if search_type in ["vector", "both"]:
                    print(f"Vector Score: {product.get('vector_score', 'N/A'):.4f}")
                    if "rrf_calculation" in product:
                        print(f"Vector Rank: {product['rrf_calculation'].get('vector_rank', 'N/A')}")
                
                if search_type in ["text", "both"]:
                    print(f"Text Score: {product.get('text_score', 'N/A'):.4f}")
                    if "rrf_calculation" in product:
                        print(f"Text Rank: {product['rrf_calculation'].get('text_rank', 'N/A')}")
                
                print(f"Title: {product.get('Product Name', 'No title')}")
                print(f"Price: {product.get('price', 'N/A')}")
                
            else:
                # For other result types (merged, etc.)
                print(f"Search Type: {search_type_display}")
                
                if search_type in ["vector", "both"]:
                    print(f"Vector Score: {product.get('vector_score', 'N/A'):.4f}")
                
                if search_type in ["text", "both"]:
                    print(f"Text Score: {product.get('text_score', 'N/A'):.4f}")
                
                print(f"Title: {product.get('Product Name', 'No title')}")
                print(f"Price: {product.get('price', 'N/A')}")
            
            # Clean up and display description
            description = product.get('Description', '')
            if description is None or description == '' or description == '- -: - -' or description == '- -':
                description = "[No description available]"
            elif isinstance(description, str) and len(description) > 150:
                # Truncate long descriptions
                description = description[:150] + "..."
                
            print(f"Description: {description}")
            print("-" * 80)
    else:
        print("No results found")
def hybrid_search(query, limit=5, vector_weight=1.0, text_weight=1.0):
    # For best results, request more results from individual searches
    search_limit = max(15, limit * 3)
    
    _, vector_results, text_results = hybrid_mongodb_search(
        query, 
        vector_limit=search_limit,
        text_limit=search_limit
    )
    
    rrf_results = custom_rerank_with_rrf(
        vector_results, 
        text_results, 
        top_k=search_limit,
        vector_weight=vector_weight,
        text_weight=text_weight
    )
    
    cohere_results = rerank_with_cohere(
        query, 
        rrf_results,  # Using RRF results as input to VoyageAI
        top_k=search_limit
    )
    return (
        vector_results[:limit], 
        text_results[:limit],
        rrf_results[:limit],
        cohere_results[:limit]
    )
import os
import pymongo
import logging
from pprint import pprint
import time
import cohere
import decimal
from decimal import Decimal

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def rerank_with_cohere(query, results_to_rerank, top_k=10, model="rerank-english-v3.0"):
    try:
        if not results_to_rerank:
            logger.warning("No results to rerank")
            return []

        rerank_results = [r.copy() for r in results_to_rerank]

        # Initialize Cohere client
        cohere_api_key = <CODE>
        co = cohere.Client(cohere_api_key)

        # Analyze query
        #query_analysis = analyze_query(query)
        #query_type = query_analysis["probable_type"]
        #if query_type == "keyword":
        #    query = f"Find the descriptions that match: {query}. Give more weight to lexical matches."

        #logger.info(f"Reranking {len(rerank_results)} results for query: '{query}' using model: {model}")

        # Prepare Cohere-compatible documents
        documents = []
        for idx, result in enumerate(rerank_results):
            description = result.get("Description", "No description available")
            if not description.strip() or description.strip() in ['- -', '- -: - -']:
                description = "No description available"
            documents.append({"text": description, "id": str(idx)})

        # Call Cohere rerank
        rerank_response = co.rerank(
            query=query,
            documents=documents,
            model=model,
            top_n=len(documents)
        )

        # Map scores back to original results using result.index
        for result in rerank_response.results:
            idx = result.index
            score = result.relevance_score
            rerank_results[idx]["rerank_score"] = score
            #rerank_results[idx]["query_type"] = query_type

        # Sort by rerank_score
        reranked_results = sorted(rerank_results, key=lambda x: x.get("rerank_score", 0), reverse=True)

        # Return top_k
        return reranked_results[:top_k]

    except Exception as e:
        logger.error(f"Error during Cohere reranking: {str(e)}")
        logger.warning("Reranking failed, falling back to original results")
        return results_to_rerank[:top_k]


def interactive_hybrid_search():
    print("\n" + "=" * 80)
    print("🔍 Amazon Product Hybrid Search with Weighted RRF → VoyageAI Reranking")
    print("=" * 80)
    print("Type your search queries below. Enter 'quit' or 'exit' to end the session.")
    print("=" * 80)
    
    # Default weights
    vector_weight = 1.0
    text_weight = 1.0
    
    while True:
        # Get user input
        query = input("\nEnter your search query (or 'weights' to adjust RRF weights): ").strip()
        
        # Check for exit command
        if query.lower() in ('quit', 'exit', 'q'):
            print("\nThank you for using the hybrid search. Goodbye!")
            break
            
        # Check for weights adjustment command
        if query.lower() == 'weights':
            try:
                # Get vector weight
                vector_input = input("Enter weight for vector search (default 1.0): ").strip()
                if vector_input:
                    vector_weight = float(vector_input)
                    
                # Get text weight
                text_input = input("Enter weight for text search (default 1.0): ").strip()
                if text_input:
                    text_weight = float(text_input)
                    
                print(f"Weights set to: Vector = {vector_weight}, Text = {text_weight}")
                continue
            except ValueError:
                print("Invalid input. Weights must be numbers.")
                continue
            
        # Skip empty queries
        if not query:
            print("Please enter a valid search query.")
            continue
            
        # Optional: Ask for number of results
        try:
            results_count = input("Number of results to show (default 5): ").strip()
            limit = int(results_count) if results_count else 5
        except ValueError:
            print("Invalid number, using default of 5 results.")
            limit = 5
            
        # Execute search with weights
        print(f"\n🔍 Searching for: '{query}'")
        print(f"Using weights: Vector = {vector_weight}, Text = {text_weight}")
        try:
            vector_results, text_results, rrf_results, cohere_results = hybrid_search(
                query, 
                limit=limit,
                vector_weight=vector_weight,
                text_weight=text_weight
            )
            
            # Analyze the query for display
            query_analysis = analyze_query(query)
            query_type = query_analysis["probable_type"]
            print(f"Query analysis: This appears to be a {query_type}-oriented query.")
            
            # Display results from each step
            display_search_results(vector_results, "Vector Search Results")
            display_search_results(text_results, "Full Text Search Results")
            display_search_results(rrf_results, "RRF Combined Results")
            display_search_results(cohere_results, "Cohere Reranked Results")
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            print(f"❌ Search error: {str(e)}")
            print("Please try a different query or check your connection.")

if __name__ == "__main__":
    try:
        interactive_hybrid_search()
    except KeyboardInterrupt:
        print("\n\nSearch session interrupted. Goodbye!")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"An unexpected error occurred: {str(e)}")
