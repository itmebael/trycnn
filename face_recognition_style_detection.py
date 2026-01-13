"""
Face Recognition Style Detection for Pechay
Uses petchay_dataset embeddings for similarity matching (like face recognition)
"""

import numpy as np
from typing import Dict, Any, Optional, List
from db import supabase, get_all_yolo_files

def face_recognition_style_match(image_path: str, cnn_predictor) -> Dict[str, Any]:
    """
    Face Recognition Style Matching:
    1. Extract embedding from uploaded image
    2. Compare with all embeddings in petchay_dataset using cosine similarity
    3. Find closest match (highest similarity)
    4. Return match if similarity > threshold
    """
    try:
        if not cnn_predictor:
            print("[FACE RECOGNITION] CNN predictor not available")
            return {"matched": False, "confidence": 0.0}
        
        # 1. Extract embedding from uploaded image
        print("[FACE RECOGNITION] Extracting embedding from uploaded image...")
        uploaded_embedding = cnn_predictor.extract_features(image_path)
        if uploaded_embedding is None:
            print("[FACE RECOGNITION] Failed to extract embedding")
            return {"matched": False, "confidence": 0.0}
        
        uploaded_embedding = uploaded_embedding.flatten()
        
        # 2. Get all entries from petchay_dataset with embeddings
        print("[FACE RECOGNITION] Querying petchay_dataset for embeddings...")
        if supabase:
            try:
                # Get all entries with embeddings
                response = supabase.table("petchay_dataset").select("*").not_.is_("embedding", "null").execute()
                dataset_entries = response.data if response.data else []
                print(f"[FACE RECOGNITION] Found {len(dataset_entries)} entries with embeddings")
            except Exception as e:
                print(f"[FACE RECOGNITION] Error querying Supabase: {e}")
                dataset_entries = []
        else:
            from db import get_dataset_entries
            dataset_entries = [e for e in get_dataset_entries() if e.get("embedding")]
        
        if not dataset_entries:
            print("[FACE RECOGNITION] No entries with embeddings found")
            return {"matched": False, "confidence": 0.0}
        
        # 3. Compare with all embeddings (face recognition style)
        best_match = None
        best_similarity = 0.0
        best_entry = None
        
        print(f"[FACE RECOGNITION] Comparing with {len(dataset_entries)} embeddings...")
        for entry in dataset_entries:
            entry_embedding = entry.get("embedding")
            if entry_embedding is None:
                continue
            
            try:
                # Convert to numpy array
                if isinstance(entry_embedding, list):
                    entry_emb = np.array(entry_embedding)
                else:
                    entry_emb = entry_embedding
                
                # Ensure same shape
                if entry_emb.shape != uploaded_embedding.shape:
                    # Try to reshape or skip
                    if entry_emb.size == uploaded_embedding.size:
                        entry_emb = entry_emb.reshape(uploaded_embedding.shape)
                    else:
                        continue
                
                # Calculate cosine similarity (face recognition style)
                # Cosine similarity = dot product / (norm1 * norm2)
                dot_product = np.dot(uploaded_embedding, entry_emb)
                norm_uploaded = np.linalg.norm(uploaded_embedding)
                norm_entry = np.linalg.norm(entry_emb)
                
                if norm_uploaded > 0 and norm_entry > 0:
                    similarity = dot_product / (norm_uploaded * norm_entry)
                    
                    # Track best match
                    if similarity > best_similarity:
                        best_similarity = float(similarity)
                        best_entry = entry
                        best_match = {
                            "id": entry.get("id"),
                            "filename": entry.get("filename"),
                            "condition": entry.get("condition", "Healthy"),
                            "disease_name": entry.get("disease_name") or entry.get("label"),
                            "image_url": entry.get("image_url"),
                            "similarity": best_similarity,
                            "quality_score": entry.get("quality_score"),
                            "is_verified": entry.get("is_verified", False)
                        }
            except Exception as e:
                print(f"[FACE RECOGNITION] Error comparing with {entry.get('filename')}: {e}")
                continue
        
        # 4. Check if match is above threshold (face recognition typically uses 0.6-0.8)
        similarity_threshold = 0.7  # 70% similarity threshold
        confidence_threshold = 0.5  # 50% confidence threshold
        
        if best_match and best_similarity >= similarity_threshold:
            confidence = best_similarity * 100  # Convert to percentage
            
            print(f"[FACE RECOGNITION] ✅ Match found!")
            print(f"   Similarity: {best_similarity:.2%}")
            print(f"   Condition: {best_match['condition']}")
            print(f"   Disease: {best_match.get('disease_name', 'N/A')}")
            print(f"   Filename: {best_match['filename']}")
            
            # 5. Get treatment from yolo_files if available
            treatment = None
            if best_match.get("filename"):
                try:
                    from db import get_yolo_file_by_filename
                    yolo_file = get_yolo_file_by_filename(best_match["filename"])
                    if yolo_file:
                        treatment = yolo_file.get("treatment")
                        # Also check if disease_name is better in yolo_files
                        if not best_match.get("disease_name"):
                            dataset_type = yolo_file.get("dataset_type", "")
                            if dataset_type and dataset_type not in ["Healthy", "Diseased"]:
                                best_match["disease_name"] = dataset_type
                except Exception as e:
                    print(f"[FACE RECOGNITION] Error checking yolo_files: {e}")
            
            return {
                "matched": True,
                "condition": best_match["condition"],
                "disease_name": best_match.get("disease_name"),
                "treatment": treatment,
                "confidence": confidence,
                "similarity": best_similarity,
                "matched_file": best_match["filename"],
                "matched_entry": best_entry,
                "detection_method": "face_recognition_embedding_match"
            }
        else:
            print(f"[FACE RECOGNITION] No match above threshold ({similarity_threshold:.0%})")
            print(f"   Best similarity: {best_similarity:.2%}" if best_similarity > 0 else "   No matches found")
            return {
                "matched": False,
                "confidence": best_similarity * 100 if best_similarity > 0 else 0.0,
                "best_similarity": best_similarity,
                "threshold": similarity_threshold
            }
            
    except Exception as e:
        print(f"[FACE RECOGNITION] Error in face recognition matching: {e}")
        import traceback
        traceback.print_exc()
        return {"matched": False, "confidence": 0.0, "error": str(e)}

