"""
Vector store service using pgvector.

TODO: Implement this service to:
1. Generate embeddings for text chunks
2. Store embeddings in PostgreSQL with pgvector
3. Perform similarity search
4. Link related images and tables
"""
from typing import List, Dict, Any, Optional
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text, bindparam
from app.models.document import DocumentChunk
from app.core.config import settings
from sentence_transformers import SentenceTransformer


class VectorStore:
    """
    Vector store for document embeddings and similarity search.
    
    This is a SKELETON implementation. You need to implement the core logic.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')  # Initialize embedding model
        self._ensure_extension()
    
    def _ensure_extension(self):
        """
        Ensure pgvector extension is enabled.
        
        This is implemented as an example.
        """
        try:
            self.db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            self.db.commit()
        except Exception as e:
            print(f"pgvector extension already exists or error: {e}")
            self.db.rollback()
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using sentence-transformers.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embeddings
        """
        # Use sentence-transformers to generate embeddings
        embedding = self.embeddings_model.encode(text, convert_to_numpy=True)
        return embedding
    
    async def store_chunk(
        self, 
        content: str, 
        document_id: int,
        page_number: int,
        chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentChunk:
        """
        Store a text chunk with its embedding.
        
        TODO: Implement chunk storage
        1. Generate embedding for content
        2. Create DocumentChunk record
        3. Store in database with embedding
        4. Include metadata (related images, tables, etc.)
        
        Args:
            content: Text content
            document_id: Document ID
            page_number: Page number
            chunk_index: Index of chunk in document
            metadata: Additional metadata (related_images, related_tables, etc.)
            
        Returns:
            Created DocumentChunk
        """
        raise NotImplementedError("Chunk storage not implemented yet")
    
    async def similarity_search(
        self,
        query: str,
        document_id: Optional[int] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.

        Args:
            query: Search query text
            document_id: Optional document ID to filter
            k: Number of results to return

        Returns:
            List of matching chunks with similarity scores
        """
        try:
            # Generate embedding for the query
            query_embedding = await self.generate_embedding(query)

            # Convert embedding to list for PostgreSQL
            embedding_list = query_embedding.tolist()

            # Build SQL query with pgvector similarity search
            # Use :param syntax with bindparam for proper escaping
            if document_id is not None:
                sql_query = text("""
                    SELECT
                        dc.id,
                        dc.content,
                        dc.page_number,
                        dc.chunk_index,
                        dc.document_id,
                        dc.extra_metadata,
                        1 - (dc.embedding <=> CAST(:query_embedding AS vector)) as similarity
                    FROM document_chunks dc
                    WHERE dc.document_id = :document_id
                    ORDER BY dc.embedding <=> CAST(:query_embedding AS vector)
                    LIMIT :k
                """).bindparams(
                    bindparam("query_embedding", value=str(embedding_list)),
                    bindparam("document_id", value=document_id),
                    bindparam("k", value=k)
                )
            else:
                sql_query = text("""
                    SELECT
                        dc.id,
                        dc.content,
                        dc.page_number,
                        dc.chunk_index,
                        dc.document_id,
                        dc.extra_metadata,
                        1 - (dc.embedding <=> CAST(:query_embedding AS vector)) as similarity
                    FROM document_chunks dc
                    ORDER BY dc.embedding <=> CAST(:query_embedding AS vector)
                    LIMIT :k
                """).bindparams(
                    bindparam("query_embedding", value=str(embedding_list)),
                    bindparam("k", value=k)
                )

            result = self.db.execute(sql_query)
            rows = result.fetchall()

            # Format results
            chunks = []
            for row in rows:
                chunk = {
                    "id": row[0],
                    "content": row[1],
                    "page_number": row[2],
                    "chunk_index": row[3],
                    "document_id": row[4],
                    "metadata": row[5] or {},
                    "score": float(row[6])
                }
                chunks.append(chunk)

            return chunks

        except Exception as e:
            # Rollback the transaction on error
            self.db.rollback()
            print(f"Error in similarity_search: {e}")
            import traceback
            traceback.print_exc()
            # Return empty list instead of raising to allow graceful degradation
            return []
    
    async def get_related_content(
        self,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get related images and tables for given chunks.

        Args:
            chunks: List of chunk dictionaries with document_id and page_number

        Returns:
            Dictionary with images and tables organized by type
        """
        from app.models.document import DocumentImage, DocumentTable

        images = []
        tables = []

        # Get unique document IDs and page numbers from chunks
        doc_pages = set()
        for chunk in chunks:
            if chunk.get("document_id") and chunk.get("page_number"):
                doc_pages.add((chunk["document_id"], chunk["page_number"]))

        if not doc_pages:
            return {"images": [], "tables": []}

        # Query images from the same pages
        for doc_id, page_no in doc_pages:
            # Get images
            doc_images = self.db.query(DocumentImage).filter(
                DocumentImage.document_id == doc_id,
                DocumentImage.page_number == page_no
            ).all()

            for img in doc_images:
                # Convert absolute path to relative URL path
                url = img.file_path.replace("\\", "/")
                if url.startswith("/app/uploads/"):
                    url = url.replace("/app/uploads/", "/uploads/")
                images.append({
                    "id": img.id,
                    "url": url,
                    "caption": img.caption,
                    "page": img.page_number,
                    "width": img.width,
                    "height": img.height
                })

            # Get tables
            doc_tables = self.db.query(DocumentTable).filter(
                DocumentTable.document_id == doc_id,
                DocumentTable.page_number == page_no
            ).all()

            for tbl in doc_tables:
                # Convert absolute path to relative URL path
                url = tbl.image_path.replace("\\", "/")
                if url.startswith("/app/uploads/"):
                    url = url.replace("/app/uploads/", "/uploads/")
                tables.append({
                    "id": tbl.id,
                    "url": url,
                    "caption": tbl.caption,
                    "page": tbl.page_number,
                    "rows": tbl.rows,
                    "columns": tbl.columns,
                    "data": tbl.data
                })

        return {
            "images": images,
            "tables": tables
        }
