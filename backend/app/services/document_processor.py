"""
Document processing service using Docling

TODO: Implement this service to:
1. Parse PDF documents using Docling
2. Extract text, images, and tables
3. Store extracted content in database
4. Generate embeddings for text chunks
"""
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from app.models.document import Document, DocumentChunk, DocumentImage, DocumentTable
from app.services.vector_store import VectorStore
import os
from PyPDF2 import PdfReader
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

import uuid
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io
import warnings
import time
from app.core.config import settings
import logging

# Suppress strict_text warnings
warnings.filterwarnings('ignore', message='.*strict_text.*')
# Set up logger
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    

    """
    Process PDF documents and extract multimodal content.
    
    This is a SKELETON implementation. You need to implement the core logic.
    """

    def __init__(self, db: Session):
        self.db = db
        self.vector_store = VectorStore(db)
    
    async def process_document(self, file_path: str, document_id: int) -> Dict[str, Any]:
        """
        Process a PDF document using PyPDF2.
        
        For now, this extracts text only. Images and tables extraction can be added later.
        """
        start_time = time.time()
        try:
        
            # 1. Update status to 'processing'
            await self._update_document_status(document_id, 'processing', total_pages=0,
            text_chunks_count=0,
            images_count=0,
            tables_count=0,
            error_message=None)
            
            # 2. Extract text from PDF  
           # reader = PdfReader(file_path)
           # Use Docling to extract structured content
            # create pipeline options
            pdf_options = PdfPipelineOptions()
# enable OCR (useful if text extraction is weak or has images)
            pdf_options = PdfPipelineOptions()
            pdf_options.do_ocr = True
            pdf_options.do_table_structure = True
            pdf_options.images_scale = 2.0  # Higher quality images
            pdf_options.generate_page_images = True
            pdf_options.generate_picture_images = True
# create converter with custom PDF format options
            converter = DocumentConverter(
                format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
            }
        )
            result = converter.convert(file_path)
            doc = result.document  # type: Document
            full_text = doc.export_to_text()
            text_chunk_count = 0
        

            if hasattr(doc, 'pages') and doc.pages:
                total_pages = len(doc.pages)
            

                  
            # Extract content
            IMAGE_DIR = os.path.join(settings.UPLOAD_DIR, "images")
            doc = result.document   # MUST be DoclingDocument

            TABLE_DIR = os.path.join(settings.UPLOAD_DIR, "tables")
            
            tables = await self.extract_and_save_tables(doc, document_id, TABLE_DIR)
            text_chunks = await self.extract_text_chunks(doc, document_id)
            await self._save_text_chunks(text_chunks, document_id)
           
            images = await self.extract_images(doc, document_id, IMAGE_DIR)

            text_chunk_count = len(text_chunks)
            image_count = images
            table_count = tables
            logger.info(f"Saved {text_chunk_count} chunks for document {document_id}")
        
              # 7. Update status to 'completed'
            await self._update_document_status(document_id, 'completed', total_pages,
            text_chunk_count,
            image_count,
            table_count)
        
            processing_time = time.time() - start_time
        

            
            return {
                "status": "success",
                "text_chunks": text_chunk_count,
                "images": image_count,
                "tables": table_count,
                "processing_time": processing_time
            }
        except Exception as e:
            total_pages=0
            text_chunk_count=0
            image_count=0
            # 5. Handle errors
            await self._update_document_status(document_id, 'error', total_pages=0,
                                               text_chunks_count=0,
                                               images_count=0,
                                               tables_count=0,
                                               error_message=str(e))

            return {
                "status": "error",
                "message": str(e)
            }
    
    def _chunk_text(self, text: str, document_id: int) -> List[Dict[str, Any]]:
        """
        Split text into chunks for vector storage.
        
        Args:
            text: Full text content from the document
            document_id: Document ID
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        from app.core.config import settings
        
        chunk_size = settings.CHUNK_SIZE
        chunk_overlap = settings.CHUNK_OVERLAP
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If not at the end, try to find a good break point
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                last_period = text.rfind('.', end - 100, end)
                last_newline = text.rfind('\n', end - 100, end)
                break_point = max(last_period, last_newline)
                
                if break_point > start:
                    end = break_point + 1
            
            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append({
                    "document_id": document_id,
                    "content": chunk_content,
                    "chunk_index": len(chunks),
                      "page_number": None  # No page info in fallback mode
                })
            
            # Move start position with overlap
            start = end - chunk_overlap if end < len(text) else end
        
        return chunks

    def _chunk_text_with_page(
    self,
    text: str,
    document_id: int,
    page_number: int,
    start_chunk_index: int = 0
) -> List[Dict[str, Any]]:
        """
        Split text from a single page into chunks with page tracking.
        
        Args:
            text: Text content from the page
            document_id: Document ID
            page_number: Page number (from PDF)
            start_chunk_index: Starting index for chunks
            
        Returns:
            List of chunk dictionaries with content, page number, and metadata
        """
        from app.core.config import settings
    
        chunk_size = settings.CHUNK_SIZE
        chunk_overlap = settings.CHUNK_OVERLAP
        
        chunks = []
        start = 0
        chunk_idx = start_chunk_index
    
        while start < len(text):
            end = start + chunk_size
            
            # If not at the end, try to find a good break point
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                last_period = text.rfind('.', end - 100, end)
                last_newline = text.rfind('\n', end - 100, end)
                break_point = max(last_period, last_newline)
                
                if break_point > start:
                    end = break_point + 1
            
            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append({
                    "document_id": document_id,
                    "content": chunk_content,
                    "chunk_index": chunk_idx,
                    "page_number": page_number,  # ← Page number included!
                    "char_count": len(chunk_content)
                })
                chunk_idx += 1
            
            # Move start position with overlap
                start = end - chunk_overlap if end < len(text) else end
        
        return chunks   
    
    async def _save_text_chunks(self, chunks: List[Dict[str, Any]], document_id: int):
        """
        Save text chunks to database with embeddings.

        Args:
            chunks: List of chunk dictionaries
            document_id: Document ID
        """
        logger.info("Generated embedding for chunk index  of document ID: %s", chunks)

        for chunk in chunks:
            # Generate embeddings
            embedding = await self.vector_store.generate_embedding(chunk["content"])

            # Create the DocumentChunk record
            document_chunk = DocumentChunk(
                document_id=document_id,
                content=chunk["content"],
                embedding=embedding,
                chunk_index=chunk["chunk_index"],
                page_number=chunk["page_number"],


            )

            self.db.add(document_chunk)
            #logger.error("document chunk added to session for document ID: %s", document_chunk.)
        try:
            self.db.commit()
            logger.info("Committed all chunks for document ID: %s", document_id)
        except Exception as e:
            self.db.rollback()
            logger.error("DB commit failed for document %s: %s", document_id, str(e))
            raise
        #self.db.commit()
    
    async def _save_image(
        self,
        pil_image,
        document_id: int,
        page_number: int,
        metadata: Dict[str, Any],
        output_dir: str
    ) -> DocumentImage:
        """
        Save an extracted image to disk and create DB record.

        Args:
            pil_image: PIL Image object or ImageRef
            document_id: Document ID
            page_number: Page number where image appears
            metadata: Dictionary with optional 'caption', 'width', 'height', 'prov'
            output_dir: Directory to save the image file

        Returns:
            DocumentImage record (not yet committed)
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        image_filename = f"{uuid.uuid4().hex}.png"
        file_path = os.path.join(output_dir, image_filename)

        # Handle different image types
        if hasattr(pil_image, 'save') and callable(getattr(pil_image, 'save')):
            # It's an ImageRef or has a save method
            if hasattr(pil_image, '__call__'):
                # Docling's ImageRef with save method
                saved_path = pil_image.save(output_dir, filename=image_filename)
                if saved_path:
                    file_path = saved_path
            else:
                # PIL Image
                pil_image.save(file_path, format='PNG')
        else:
            raise ValueError("pil_image must be a PIL Image or have a save method")

        # Extract metadata
        caption = metadata.get('caption')
        width = metadata.get('width')
        height = metadata.get('height')

        # If dimensions not provided, read from saved file
        if width is None or height is None:
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
            except Exception:
                pass

        # Create DocumentImage record
        document_image = DocumentImage(
            document_id=document_id,
            file_path=file_path,
            page_number=page_number,
            caption=caption,
            width=int(width) if width else None,
            height=int(height) if height else None,
            extra_metadata=metadata.get('extra_metadata', {})
        )
        self.db.add(document_image)
        return document_image
    
    async def _save_table(
        self,
        table_data: List[List[Any]],
        document_id: int,
        page_number: int,
        metadata: Dict[str, Any],
        output_dir: str
    ) -> DocumentTable:
        """
        Save an extracted table to disk and create DB record.

        Args:
            table_data: 2D list of table cells
            document_id: Document ID
            page_number: Page number where table appears
            metadata: Dictionary with optional 'caption', 'rows', 'columns'
            output_dir: Directory to save the table image

        Returns:
            DocumentTable record (not yet committed)
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Render table as image
        img = self._render_table_as_image(table_data)

        # Generate unique filename
        filename = f"{uuid.uuid4().hex}.png"
        file_path = os.path.join(output_dir, filename)

        # Save image to disk
        img.save(file_path)

        # Extract metadata
        caption = metadata.get('caption')
        rows = metadata.get('rows') or (len(table_data) if isinstance(table_data, (list, tuple)) else None)
        columns = metadata.get('columns')
        if columns is None and rows and len(table_data) > 0:
            columns = len(table_data[0]) if isinstance(table_data[0], (list, tuple)) else None

        # Create DocumentTable record
        document_table = DocumentTable(
            document_id=document_id,
            image_path=file_path,
            data=table_data,
            page_number=page_number,
            caption=caption,
            rows=rows,
            columns=columns,
            extra_metadata=metadata.get('extra_metadata', {})
        )
        self.db.add(document_table)
        return document_table
    
    async def _update_document_status(
        self, 
        document_id: int, 
        status: str, 
        total_pages: int,
        text_chunks_count: int,
        images_count: int,
        tables_count: int,
        error_message: str = None
    ):
        """
        Update document processing status.
        
        This is implemented as an example.
        """
        logger.info(f"Updating document {document_id} status to {status} text chunks {text_chunks_count} total pages {total_pages}" )
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.processing_status = status,
            document.total_pages = total_pages
            document.text_chunks_count = text_chunks_count
            document.images_count = images_count
            document.tables_count = tables_count
            if error_message:
                document.error_message = error_message
            self.db.commit()
 
    async def extract_images(self, doc, document_id: int, output_dir: str) -> int:
        """
        Extract images from a Docling document and save to DB using doc.pictures.

        Args:
            doc: DoclingDocument object
            document_id: ID of the document in DB
            output_dir: Directory to save extracted image files

        Returns:
            int: Number of images saved
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        saved_count = 0

        pictures = getattr(doc, "pictures", [])

        if not pictures:
            logger.info("Document has no pictures.")
            return 0

        logger.info(f"Document has {len(pictures)} pictures.")

        for i, pic in enumerate(pictures, start=1):
            try:
                # Get image reference
                image_ref = getattr(pic, "image", None)
                if image_ref is None:
                    logger.warning(f"Picture {i} has no ImageRef, skipping")
                    continue

                # Prepare PIL image for saving
                pil_image = None
                if hasattr(image_ref, 'save'):
                    pil_image = image_ref
                else:
                    # Fallback: try to get PIL Image
                    pil_image = getattr(image_ref, 'pil_image', None)
                    if pil_image is None and hasattr(image_ref, 'to_pil'):
                        pil_image = image_ref.to_pil()

                if pil_image is None:
                    logger.warning(f"Picture {i} could not be converted to image, skipping")
                    continue

                # Extract metadata from Picture object
                page_no = None
                caption = None
                width = None
                height = None

                # Try to get metadata from prov (provenance)
                prov = getattr(pic, "prov", None)
                if prov:
                    if isinstance(prov, list) and len(prov) > 0:
                        prov_item = prov[0]
                        # Try as object attributes first (ProvenanceItem), then as dict
                        if hasattr(prov_item, 'page_no'):
                            page_no = prov_item.page_no
                            bbox = getattr(prov_item, 'bbox', None)
                            if bbox and hasattr(bbox, 'l'):
                                width = bbox.r - bbox.l
                                height = bbox.b - bbox.t
                        elif isinstance(prov_item, dict):
                            page_no = prov_item.get("page_no")
                            bbox = prov_item.get("bbox")
                            if bbox:
                                width = bbox.get("r", 0) - bbox.get("l", 0)
                                height = bbox.get("b", 0) - bbox.get("t", 0)
                    elif isinstance(prov, dict):
                        page_no = prov.get("page_no")
                        bbox = prov.get("bbox")
                        if bbox:
                            width = bbox.get("r", 0) - bbox.get("l", 0)
                            height = bbox.get("b", 0) - bbox.get("t", 0)

                # Try alternative attributes
                if page_no is None:
                    page_no = getattr(pic, "page_number", None)
                if caption is None:
                    caption = getattr(pic, "label", None) or getattr(pic, "caption", None)

                # Build metadata dict
                metadata = {
                    'caption': caption,
                    'width': width,
                    'height': height,
                    'extra_metadata': {}
                }

                # Use the _save_image helper method
                document_image = await self._save_image(
                    pil_image=pil_image,
                    document_id=document_id,
                    page_number=page_no,
                    metadata=metadata,
                    output_dir=output_dir
                )

                # self.db.add(document_image)
                saved_count += 1
                logger.info(f"Successfully saved Picture {i} to {document_image.file_path} - Page: {page_no}, Size: {document_image.width}x{document_image.height}, Caption: {caption}")

            except Exception as e:
                logger.error(f"Failed to process Picture {i}: {e}", exc_info=True)
                continue

        # Commit all saved images
        try:
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to commit images for document {document_id}: {e}")
            raise

        logger.info(f"Saved {saved_count} images for document {document_id}")
        return saved_count

    

    async def extract_text_chunks(self, doc, document_id: int):
        doc_dict = doc.export_to_dict()
        texts = doc_dict.get("texts", [])

        text_by_page = {}

        def get_page_number(item: dict) -> int | None:
            prov = item.get("prov")

            if isinstance(prov, list):
                for p in prov:
                    if isinstance(p, dict):
                        return p.get("page_no")

            if isinstance(prov, dict):
                return prov.get("page_no")

            return None

        # 1️⃣ Group text by page
        for item in texts:
            page_no = get_page_number(item)
            logger.info(f"Page number extracted: {page_no}")

            if page_no is None:
                continue

            txt = item.get("text", "")
            logger.info(f"Extracted text length: {len(txt)}")

            if txt.strip():
                text_by_page.setdefault(page_no, []).append(txt)

        if not text_by_page:
            logger.warning("No page-wise text extracted from Docling")
            return []

        # 2️⃣ Chunk text
        chunks = []
        chunk_index = 0

        for page_no in sorted(text_by_page):
            page_text = "\n".join(text_by_page[page_no])

            page_chunks = self._chunk_text_with_page(
                text=page_text,
                document_id=document_id,
                page_number=page_no,
                start_chunk_index=chunk_index
            )

            for ch in page_chunks:
                chunks.append({
                    "document_id": document_id,
                    "content": ch["content"],
                    "page_number": page_no,
                    "chunk_index": chunk_index,
                    "extra_metadata": {"source": "docling"}
                })
                chunk_index += 1

        logger.info(f"Created {len(chunks)} text chunks for document {document_id}")
        return chunks



    

    async def extract_and_save_tables(self, doc, document_id: int, output_dir: str) -> int:
        """
        Extract tables from the document, render as images, and save to DB.

        Args:
            doc: Document object containing tables
            document_id: DB document ID
            output_dir: Folder to save images

        Returns:
            int: Number of tables saved
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        saved_count = 0

        # Try to get tables from doc.tables first
        tables = getattr(doc, "tables", None)

        # If not found or empty, check other possible locations
        if not tables:
            for attr in ["elements", "blocks", "pages"]:
                possible = getattr(doc, attr, None)
                if possible:
                    for elem in possible:
                        t = getattr(elem, "tables", None)
                        if t:
                            tables = t
                            break
                if tables:
                    break

        if not tables:
            logger.info("No tables found in the document.")
            return 0

        logger.info(f"Found {len(tables)} tables in the document.")

        for i, table in enumerate(tables, start=1):
            try:
                # Try multiple ways to extract table data
                table_data = None

                # Method 1: Try export_to_dataframe and convert to list of lists
                if hasattr(table, 'export_to_dataframe'):
                    try:
                        df = table.export_to_dataframe()
                        if df is not None and not df.empty:
                            # Convert dataframe to list of lists (with headers)
                            table_data = [df.columns.tolist()] + df.values.tolist()
                    except Exception as e:
                        logger.debug(f"Could not export table {i} to dataframe: {e}")

                # Method 2: Try to get data attribute
                if table_data is None:
                    table_data = getattr(table, "data", None)

                # Method 3: Try to convert to dict and extract cells
                if table_data is None and hasattr(table, 'export_to_dict'):
                    try:
                        table_dict = table.export_to_dict()
                        cells = table_dict.get("data", None)
                        if cells:
                            table_data = cells
                    except Exception as e:
                        logger.debug(f"Could not export table {i} to dict: {e}")

                if not table_data:
                    logger.warning(f"Table {i} has no data, skipping")
                    continue

                # Extract metadata from Table object
                page_number = None
                caption = None

                # Try to get page number from prov
                prov = getattr(table, "prov", None)
                if prov:
                    if isinstance(prov, list) and len(prov) > 0:
                        prov_item = prov[0]
                        # Try as object attributes first (ProvenanceItem), then as dict
                        if hasattr(prov_item, 'page_no'):
                            page_number = prov_item.page_no
                        elif isinstance(prov_item, dict):
                            page_number = prov_item.get("page_no")
                    elif isinstance(prov, dict):
                        page_number = prov.get("page_no")

                # Try alternative attributes
                if page_number is None:
                    page_number = getattr(table, "page_number", None)
                if caption is None:
                    caption = getattr(table, "label", None) or getattr(table, "caption", None)

                # Build metadata dict
                metadata = {
                    'caption': caption,
                    'rows': len(table_data) if isinstance(table_data, (list, tuple)) else None,
                    'columns': len(table_data[0]) if table_data and len(table_data) > 0 and isinstance(table_data[0], (list, tuple)) else None,
                    'extra_metadata': {}
                }

                # Use the _save_table helper method
                document_table = await self._save_table(
                    table_data=table_data,
                    document_id=document_id,
                    page_number=page_number,
                    metadata=metadata,
                    output_dir=output_dir
                )

                # self.db.add(document_table)
                saved_count += 1
                logger.info(f"Successfully saved Table {i} - Page: {page_number}, Size: {document_table.rows}x{document_table.columns}, Caption: {caption}")

            except Exception as e:
                logger.error(f"Failed to save Table {i} to DB: {e}", exc_info=True)
                continue

        try:
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to commit tables for document {document_id}: {e}")
            raise

        logger.info(f"Saved {saved_count} tables for document {document_id}")
        return saved_count


    def _render_table_as_image(self, table_data):
        """
        Renders a simple table (list of lists) to an image.

        Args:
            table_data (list of lists): 2D data of the table cells (strings/numbers)

        Returns:
            PIL.Image: Rendered image of the table
        """
        cell_width = 150
        cell_height = 50
        padding = 10
        font_size = 12

        rows = len(table_data)
        columns = max(len(row) for row in table_data) if rows > 0 else 0

        width = columns * cell_width + 2 * padding
        height = rows * cell_height + 2 * padding

        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.load_default()
        except:
            font = None

        for r in range(rows):
            row_data = table_data[r] if r < len(table_data) else []
            for c in range(columns):
                x0 = padding + c * cell_width
                y0 = padding + r * cell_height
                x1 = x0 + cell_width
                y1 = y0 + cell_height

                # Draw cell border
                draw.rectangle([x0, y0, x1, y1], outline="black", width=1)

                # Draw cell text
                if c < len(row_data):
                    cell_text = str(row_data[c]) if row_data[c] is not None else ""

                    # Truncate long text
                    if len(cell_text) > 20:
                        cell_text = cell_text[:17] + "..."

                    # Use textbbox instead of deprecated textsize
                    try:
                        bbox = draw.textbbox((0, 0), cell_text, font=font)
                        w = bbox[2] - bbox[0]
                        h = bbox[3] - bbox[1]
                    except AttributeError:
                        # Fallback for older PIL versions
                        w, h = draw.textsize(cell_text, font=font)

                    text_x = x0 + (cell_width - w) / 2
                    text_y = y0 + (cell_height - h) / 2
                    draw.text((text_x, text_y), cell_text, fill="black", font=font)

        return img
