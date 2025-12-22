"""
Chat engine service for multimodal RAG.

TODO: Implement this service to:
1. Process user messages
2. Search for relevant context using vector store
3. Find related images and tables
4. Generate responses using LLM
5. Support multi-turn conversations
"""
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from app.models.conversation import Conversation, Message
from app.services.vector_store import VectorStore
from app.core.config import settings
import time
import httpx
import json
import logging

logger = logging.getLogger(__name__)


class ChatEngine:
    """
    Multimodal chat engine with RAG.
    
    This is a SKELETON implementation. You need to implement the core logic.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.vector_store = VectorStore(db)
        self.ollama_base_url = settings.OLLAMA_BASE_URL
        self.ollama_model = settings.OLLAMA_MODEL
    
    async def process_message(
        self,
        conversation_id: int,
        message: str,
        document_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message and generate multimodal response.

        Args:
            conversation_id: Conversation ID
            message: User message
            document_id: Optional document ID to scope search

        Returns:
            Dictionary with answer, sources, and processing time
        """
        start_time = time.time()

        try:
            # 1. Load conversation history
            history = await self._load_conversation_history(conversation_id)

            # 2. Search for relevant context
            context = await self._search_context(message, document_id, k=settings.TOP_K_RESULTS)

            # If no context found, provide a helpful message
            if not context:
                logger.warning(f"No context found for query: {message}")
                return {
                    "answer": "I couldn't find any relevant information in the uploaded documents. Please make sure the document has been processed successfully, or try asking a different question.",
                    "sources": [],
                    "processing_time": time.time() - start_time
                }

            # 3. Find related media (images and tables)
            media = await self._find_related_media(context)

            # 4. Generate response using LLM
            answer = await self._generate_response(message, context, history, media)

            # 5. Format sources
            sources = self._format_sources(context, media)

            processing_time = time.time() - start_time

            return {
                "answer": answer,
                "sources": sources,
                "processing_time": processing_time
            }

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            # Return a user-friendly error message instead of raising
            return {
                "answer": f"Sorry, I encountered an error processing your message: {str(e)}. Please try again or contact support if the issue persists.",
                "sources": [],
                "processing_time": time.time() - start_time
            }
    
    async def _load_conversation_history(
        self,
        conversation_id: int,
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """
        Load recent conversation history.

        Args:
            conversation_id: Conversation ID
            limit: Number of recent messages to load (excluding current)

        Returns:
            List of messages formatted for LLM
        """
        # Get all messages except the most recent one (which is the current user message)
        messages = self.db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at.desc()).limit(limit + 1).all()

        # Remove the first message (most recent = current user message)
        # and reverse to get chronological order
        if messages:
            messages = list(reversed(messages[1:]))  # Skip first, reverse rest
        else:
            messages = []

        # Format for LLM
        history = []
        for msg in messages:
            history.append({
                "role": msg.role,
                "content": msg.content
            })

        return history
    
    async def _search_context(
        self,
        query: str,
        document_id: Optional[int] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant context using vector store.

        Args:
            query: Search query
            document_id: Optional document filter
            k: Number of results

        Returns:
            List of relevant chunks
        """
        chunks = await self.vector_store.similarity_search(
            query=query,
            document_id=document_id,
            k=k
        )
        return chunks
    
    async def _find_related_media(
        self,
        context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find related images and tables from context chunks.

        Args:
            context_chunks: List of relevant text chunks

        Returns:
            Dictionary with images and tables
        """
        media = await self.vector_store.get_related_content(context_chunks)
        return media
    
    async def _generate_response(
        self,
        message: str,
        context: List[Dict[str, Any]],
        history: List[Dict[str, str]],
        media: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """
        Generate response using Ollama LLM.

        Args:
            message: User message
            context: Retrieved context chunks
            history: Conversation history
            media: Related images and tables

        Returns:
            Generated answer
        """
        # Build system prompt
        system_prompt = """You are a helpful AI assistant that answers questions about documents.
You have access to relevant context from the document, including text excerpts, images, and tables.

Guidelines:
- Answer based on the provided context
- Reference specific pages when citing information
- Mention relevant images and tables when they support your answer
- If the context doesn't contain enough information, say so
- Be concise but comprehensive
- Use bullet points and formatting for clarity"""

        # Build context section
        context_text = "\n\n=== RELEVANT CONTEXT ===\n"
        for i, chunk in enumerate(context, 1):
            page_ref = f" (Page {chunk['page_number']})" if chunk.get('page_number') else ""
            context_text += f"\n[{i}]{page_ref}: {chunk['content']}\n"

        # Build media references
        media_text = ""
        if media.get("images"):
            media_text += "\n=== AVAILABLE IMAGES ===\n"
            for img in media["images"]:
                page_ref = f" on page {img['page']}" if img.get('page') else ""
                caption = img.get('caption', 'No caption')
                media_text += f"- Image{page_ref}: {caption}\n"

        if media.get("tables"):
            media_text += "\n=== AVAILABLE TABLES ===\n"
            for tbl in media["tables"]:
                page_ref = f" on page {tbl['page']}" if tbl.get('page') else ""
                caption = tbl.get('caption', 'No caption')
                media_text += f"- Table{page_ref}: {caption}\n"

        # Build full prompt
        user_prompt = f"{context_text}{media_text}\n\n=== QUESTION ===\n{message}"

        # Prepare messages for Ollama
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (previous turns)
        for msg in history:
            messages.append(msg)

        # Add current message with context
        messages.append({"role": "user", "content": user_prompt})

        # Call Ollama API
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    f"{self.ollama_base_url}/api/chat",
                    json={
                        "model": self.ollama_model,
                        "messages": messages,
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()
                answer = result.get("message", {}).get("content", "")

                if not answer:
                    logger.warning("Ollama returned empty response")
                    return "I received an empty response from the AI model. Please try again."

                return answer

        except httpx.ConnectError as e:
            logger.error(f"Ollama connection error: {e}")
            raise Exception(f"Cannot connect to Ollama at {self.ollama_base_url}. Please ensure Ollama is running.")
        except httpx.TimeoutException as e:
            logger.error(f"Ollama timeout error: {e}")
            raise Exception("The AI model took too long to respond. Please try again with a simpler question.")
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e}")
            if e.response.status_code == 404:
                raise Exception(f"Model '{self.ollama_model}' not found. Please pull the model using: ollama pull {self.ollama_model}")
            raise Exception(f"Ollama API error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            logger.error(f"Unexpected Ollama error: {e}")
            raise Exception(f"Failed to generate response: {str(e)}")
    
    def _format_sources(
        self,
        context: List[Dict[str, Any]],
        media: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Format sources for response.
        
        This is implemented as an example.
        """
        sources = []
        
        # Add text sources
        for chunk in context[:3]:  # Top 3 text chunks
            sources.append({
                "type": "text",
                "content": chunk["content"],
                "page": chunk.get("page_number"),
                "score": chunk.get("score", 0.0)
            })
        
        # Add image sources
        for image in media.get("images", []):
            sources.append({
                "type": "image",
                "url": image["url"],
                "caption": image.get("caption"),
                "page": image.get("page")
            })
        
        # Add table sources
        for table in media.get("tables", []):
            sources.append({
                "type": "table",
                "url": table["url"],
                "caption": table.get("caption"),
                "page": table.get("page"),
                "data": table.get("data")
            })
        
        return sources
