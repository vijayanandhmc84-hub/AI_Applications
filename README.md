# Multimodal Document Chat System

A production-ready system that allows users to upload PDF documents, extract text, images, and tables, and engage in multimodal chat with intelligent retrieval-augmented generation (RAG).

## Project Overview

This system implements a complete multimodal RAG pipeline that:
- Processes PDF documents to extract text, images, and tables using IBM's Docling
- Stores content in a PostgreSQL vector database with pgvector for semantic search
- Enables intelligent chat conversations that can reference and display relevant images and tables
- Maintains multi-turn conversation context for natural dialogue
- Provides a responsive web interface built with Next.js and TailwindCSS

### Key Capabilities
- **Document Processing**: Extract text chunks, images with captions, and structured tables from PDFs
- **Vector Search**: Semantic similarity search using pgvector with cosine distance
- **Multimodal RAG**: Generate answers enriched with relevant images and tables from the document
- **Conversation Memory**: Track conversation history for contextual follow-up questions
- **Real-time UI**: Interactive chat interface with image/table display and loading states

---

## Tech Stack

### Backend
- **Framework**: FastAPI (async API framework)
- **PDF Processing**: Docling (IBM's document understanding library)
- **Vector Database**: PostgreSQL 15 + pgvector extension
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM**: Ollama (llama3.2) - free local inference
- **ORM**: SQLAlchemy 2.0 with async support
- **Image Processing**: PIL (Pillow)

### Frontend
- **Framework**: Next.js 14 (App Router)
- **Styling**: TailwindCSS
- **State Management**: React Hooks (useState, useEffect)
- **API Client**: Fetch API
- **Language**: TypeScript

### Infrastructure
- **Database**: PostgreSQL 15 + pgvector
- **Containerization**: Docker + Docker Compose
- **Static Files**: FastAPI StaticFiles for image/table serving

---

## Setup Instructions

### Prerequisites
- Docker & Docker Compose installed
- Ollama installed on host machine (for LLM inference)
- At least 8GB RAM (for local LLM)
- 5GB free disk space

### Step 1: Install Ollama

**Windows:**
```bash
# Download from https://ollama.com/download and install
```

**Mac:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 2: Start Ollama and Pull Model

```bash
# Start Ollama service (in a separate terminal)
ollama serve

# Pull the Llama 3.2 model
ollama pull llama3.2

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

### Step 3: Clone and Configure

```bash
# Clone the repository
git clone <repository-url>
cd coding-test-4h-main

# Create .env file in backend directory
# (already configured for Docker with host.docker.internal)
cat backend/.env
```

Expected `.env` content:
```env
# Ollama configuration for Docker
# Use host.docker.internal to access host machine from Docker container
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.2
LLM_PROVIDER=ollama
```

### Step 4: Start the Application

```bash
# Start all services (database, backend, frontend)
docker-compose up -d

# Check service status
docker-compose ps

# View logs if needed
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Step 5: Access the Application

- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Database**: localhost:5432 (postgres/postgres)

### Step 6: Upload and Test

1. Navigate to http://localhost:3000
2. Click "Upload Document"
3. Select a PDF file (sample PDFs in `backend/sample_docs/`)
4. Wait for processing to complete (status will update)
5. Click "Chat" to start asking questions
6. Try questions like:
   - "What is this document about?"
   - "Show me the images"
   - "What are the main findings in the tables?"

---

## Environment Variables

### Backend Configuration

Create `.env` file in `backend/` directory:

```env
# LLM Provider Configuration
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.2
LLM_PROVIDER=ollama

# Database Configuration (auto-configured in docker-compose.yml)
DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/docuchat

# Upload Settings
UPLOAD_DIR=/app/uploads
MAX_FILE_SIZE=52428800  # 50MB

# Vector Search
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K_RESULTS=5

# Document Processing
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

### Frontend Configuration

No `.env` file needed - API URL is configured in code (`http://localhost:8000`).

For production deployment, create `.env.local`:
```env
NEXT_PUBLIC_API_URL=https://your-backend-url.com
```

---

## API Testing Examples

### 1. Upload Document

```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.pdf"
```

**Response:**
```json
{
  "id": 1,
  "filename": "sample.pdf",
  "upload_date": "2025-12-22T10:30:00",
  "processing_status": "processing",
  "message": "Document uploaded successfully. Processing started."
}
```

### 2. Check Processing Status

```bash
curl -X GET "http://localhost:8000/api/documents/1" \
  -H "accept: application/json"
```

**Response:**
```json
{
  "id": 1,
  "filename": "sample.pdf",
  "processing_status": "completed",
  "total_pages": 10,
  "text_chunks_count": 50,
  "images_count": 5,
  "tables_count": 3,
  "upload_date": "2025-12-22T10:30:00"
}
```

### 3. List All Documents

```bash
curl -X GET "http://localhost:8000/api/documents" \
  -H "accept: application/json"
```

**Response:**
```json
{
  "documents": [
    {
      "id": 1,
      "filename": "sample.pdf",
      "processing_status": "completed",
      "upload_date": "2025-12-22T10:30:00"
    }
  ],
  "total": 1
}
```

### 4. Send Chat Message

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the main findings?",
    "conversation_id": null,
    "document_id": 1
  }'
```

**Response:**
```json
{
  "conversation_id": 1,
  "message_id": 2,
  "answer": "Based on the document, the main findings include...",
  "sources": [
    {
      "type": "text",
      "content": "The results show that...",
      "page": 5,
      "score": 0.89
    },
    {
      "type": "image",
      "url": "/uploads/images/abc123.png",
      "caption": "Figure 1: Results overview",
      "page": 5
    },
    {
      "type": "table",
      "url": "/uploads/tables/xyz789.png",
      "caption": "Table 1: Performance metrics",
      "page": 6
    }
  ],
  "processing_time": 2.5
}
```

### 5. Get Conversation History

```bash
curl -X GET "http://localhost:8000/api/conversations/1" \
  -H "accept: application/json"
```

**Response:**
```json
{
  "id": 1,
  "title": "Chat about sample.pdf",
  "created_at": "2025-12-22T10:35:00",
  "messages": [
    {
      "id": 1,
      "role": "user",
      "content": "What are the main findings?",
      "created_at": "2025-12-22T10:35:00"
    },
    {
      "id": 2,
      "role": "assistant",
      "content": "Based on the document...",
      "sources": [...],
      "created_at": "2025-12-22T10:35:03"
    }
  ]
}
```

### 6. Delete Document

```bash
curl -X DELETE "http://localhost:8000/api/documents/1" \
  -H "accept: application/json"
```

**Response:**
```json
{
  "message": "Document deleted successfully",
  "deleted_files": 8
}
```

### 7. Access Static Files

```bash
# View an extracted image
curl -X GET "http://localhost:8000/uploads/images/abc123.png" --output image.png

# View an extracted table
curl -X GET "http://localhost:8000/uploads/tables/xyz789.png" --output table.png
```

---

## Features Implemented

### Core Features
- [x] **PDF Document Upload**
  - File validation (PDF only, max 50MB)
  - Async processing with status tracking
  - Progress updates in database

- [x] **Document Processing with Docling**
  - Text extraction and chunking (500 chars, 50 overlap)
  - Image extraction with captions and metadata
  - Table extraction with structured data and image rendering
  - Page number tracking for all content
  - Bounding box extraction for precise location

- [x] **Vector Store with pgvector**
  - Embedding generation using Sentence Transformers
  - Cosine similarity search with pgvector
  - Metadata storage (page numbers, chunk indices)
  - Document-scoped search filtering

- [x] **Multimodal Chat Engine**
  - RAG pipeline with context retrieval
  - Related image/table finding by page number
  - LLM integration with Ollama
  - Multi-turn conversation support
  - Conversation history management (last 10 turns)
  - Source attribution (text chunks, images, tables)

- [x] **Chat Interface**
  - Real-time message display
  - Image rendering in chat responses
  - Table rendering in chat responses
  - Loading states with animated indicators
  - Error handling with user-friendly messages
  - Auto-scroll to latest message
  - Source type indicators (📷 images, 📊 tables, 📄 text)

### Additional Features
- [x] Document list view with status
- [x] Document deletion with cleanup
- [x] Conversation persistence
- [x] Static file serving for images/tables
- [x] API documentation with Swagger UI
- [x] CORS configuration for frontend
- [x] Async database operations
- [x] Error logging and debugging

---

## Known Limitations

### Performance
- **LLM Response Time**: 30-60 seconds per query (using local Ollama)
  - Mitigation: Use faster models (llama3.2:1b) or cloud APIs (Groq, OpenAI)
- **Document Processing**: Large PDFs (100+ pages) may take 2-5 minutes
  - Mitigation: Could implement background job queue (Celery + Redis)

### Content Extraction
- **Scanned PDFs**: Text extraction requires OCR (not implemented)
  - Workaround: Use Docling's OCR mode or pre-process with Tesseract
- **Complex Tables**: Very complex multi-level tables may not parse perfectly
  - Mitigation: Tables are rendered as images as fallback

### Database
- **Existing Documents**: Documents uploaded before page number fix have NULL page numbers
  - Solution: Re-upload documents or manually update database
- **Vector Index**: No indexes on embeddings (works fine for small datasets)
  - Future: Add IVFFlat or HNSW indexes for large-scale deployments

### UI/UX
- **No Streaming**: Responses appear after complete generation
  - Future: Implement streaming with Server-Sent Events
- **Single User**: No authentication or multi-user support
  - Future: Add JWT authentication and user sessions

### Docker
- **Ollama Access**: Requires `host.docker.internal` (works on Docker Desktop)
  - Linux users: May need to use `--network=host` or run Ollama in container
- **Volume Permissions**: May encounter permission issues on Linux
  - Solution: `chmod -R 777 backend/uploads` or adjust user mapping

---

## Future Improvements

### High Priority
1. **Streaming Responses**: Implement SSE for real-time LLM output
2. **Background Processing**: Use Celery + Redis for async document processing
3. **Re-processing Capability**: Allow users to re-process existing documents
4. **Search Across Documents**: Enable cross-document search and comparison
5. **Better Error Messages**: More specific user-facing error messages

### Performance Optimization
1. **Vector Indexing**: Add pgvector indexes (IVFFlat/HNSW) for faster search
2. **Caching**: Cache embeddings and LLM responses with Redis
3. **Chunk Optimization**: Experiment with chunk sizes and overlaps
4. **Parallel Processing**: Process images/tables/text concurrently

### Feature Enhancements
1. **OCR Support**: Add text extraction for scanned PDFs
2. **Multi-document Chat**: Chat across multiple documents simultaneously
3. **Export Conversations**: Download chat history as PDF/Markdown
4. **Advanced Filters**: Filter by date, document type, tags
5. **Document Preview**: Inline PDF viewer with page navigation
6. **Citation Links**: Click source to jump to specific PDF page

### User Experience
1. **Authentication**: User accounts with JWT tokens
2. **Document Sharing**: Share documents and conversations
3. **Real-time Updates**: WebSocket for processing status
4. **Mobile Responsive**: Optimize for mobile devices
5. **Dark Mode**: Add theme switching

### Deployment & Operations
1. **Production Deployment**: ✅ Ready - See [DEPLOYMENT.md](DEPLOYMENT.md) for Railway, Vercel, Docker, and AWS deployment guides
2. **Monitoring**: Add Prometheus + Grafana
3. **Logging**: Centralized logging with ELK stack
4. **Testing**: Comprehensive unit and integration tests
5. **CI/CD Pipeline**: Automated testing and deployment

### Advanced Features
1. **Multi-language Support**: i18n for UI and document processing
2. **Custom Models**: Allow users to select different LLMs
3. **Fine-tuning**: Fine-tune embeddings on domain-specific documents
4. **Question Suggestions**: Auto-suggest follow-up questions
5. **Summary Generation**: Auto-generate document summaries

---

## Architecture

### System Overview

```
┌─────────────┐
│   Frontend  │ (Next.js + TypeScript)
│  http://localhost:3000
└──────┬──────┘
       │ HTTP REST API
       ▼
┌─────────────┐
│   Backend   │ (FastAPI + Python)
│  http://localhost:8000
└──────┬──────┘
       │
       ├─────────────────┬─────────────────┐
       │                 │                 │
       ▼                 ▼                 ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  Document   │   │  Vector     │   │    Chat     │
│  Processor  │   │   Store     │   │   Engine    │
│  (Docling)  │   │ (pgvector)  │   │   (RAG)     │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │
       └────────┬────────┴────────┬────────┘
                │                 │
                ▼                 ▼
         ┌─────────────┐   ┌─────────────┐
         │ PostgreSQL  │   │   Ollama    │
         │ + pgvector  │   │  (LLM API)  │
         │  :5432      │   │  :11434     │
         └─────────────┘   └─────────────┘
                │
                ▼
         ┌─────────────┐
         │File Storage │
         │  /uploads   │
         │ (images,    │
         │  tables)    │
         └─────────────┘
```

### Data Flow

#### 1. Document Upload & Processing
```
User uploads PDF
    ↓
FastAPI receives file
    ↓
Save to /uploads/documents/
    ↓
Create Document record (status: processing)
    ↓
Docling parses PDF
    ↓
Extract text chunks → Generate embeddings → Store in DocumentChunk
Extract images → Save to /uploads/images/ → Store in DocumentImage
Extract tables → Render as PNG → Store in DocumentTable
    ↓
Update Document (status: completed)
```

#### 2. Chat Query Processing
```
User sends message
    ↓
Create Conversation (if new)
    ↓
Save user Message
    ↓
Generate query embedding
    ↓
Vector search in DocumentChunk (top 5 results)
    ↓
Find related images/tables by page number
    ↓
Build prompt with context + conversation history
    ↓
Call Ollama LLM
    ↓
Format response with sources
    ↓
Save assistant Message
    ↓
Return to frontend with images/tables
```

### Database Schema

```sql
-- Documents table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR NOT NULL,
    file_path VARCHAR NOT NULL,
    upload_date TIMESTAMP DEFAULT NOW(),
    processing_status VARCHAR,  -- 'pending', 'processing', 'completed', 'error'
    total_pages INTEGER,
    text_chunks_count INTEGER,
    images_count INTEGER,
    tables_count INTEGER
);

-- Document chunks with vector embeddings
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    content TEXT NOT NULL,
    embedding vector(384),  -- all-MiniLM-L6-v2 produces 384-dim vectors
    page_number INTEGER,
    chunk_index INTEGER,
    metadata JSONB
);

-- Extracted images
CREATE TABLE document_images (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    file_path VARCHAR NOT NULL,
    page_number INTEGER,
    caption TEXT,
    width INTEGER,
    height INTEGER
);

-- Extracted tables
CREATE TABLE document_tables (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    image_path VARCHAR NOT NULL,  -- Rendered as image
    data JSONB,  -- Structured table data
    page_number INTEGER,
    caption TEXT,
    rows INTEGER,
    columns INTEGER
);

-- Conversations
CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    title VARCHAR,
    created_at TIMESTAMP DEFAULT NOW(),
    document_id INTEGER REFERENCES documents(id)
);

-- Messages
CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER REFERENCES conversations(id),
    role VARCHAR NOT NULL,  -- 'user' or 'assistant'
    content TEXT NOT NULL,
    sources JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## Troubleshooting

### Ollama Connection Issues

**Problem**: Backend can't connect to Ollama

**Solutions:**
```bash
# 1. Verify Ollama is running
curl http://localhost:11434/api/tags

# 2. Check Docker can access host
docker exec -it backend bash
curl http://host.docker.internal:11434/api/tags

# 3. On Linux, use host network or run Ollama in Docker
docker run -d --network=host ollama/ollama
ollama pull llama3.2
```

### Database Connection Issues

**Problem**: Backend can't connect to PostgreSQL

**Solutions:**
```bash
# 1. Check database is running
docker-compose ps

# 2. Check database logs
docker-compose logs db

# 3. Recreate database
docker-compose down -v
docker-compose up -d
```

### Images Not Displaying

**Problem**: Images show broken in chat

**Solutions:**
```bash
# 1. Check file exists
ls backend/uploads/images/

# 2. Check URL in browser
http://localhost:8000/uploads/images/<filename>.png

# 3. Check console for errors
# Open browser DevTools → Console tab

# 4. Re-upload document (older documents may have NULL page numbers)
```

### Frontend Won't Start

**Problem**: Next.js development server errors

**Solutions:**
```bash
# 1. Rebuild frontend container
docker-compose up -d --build frontend

# 2. Check logs
docker-compose logs frontend

# 3. Install dependencies manually
docker exec -it frontend bash
npm install
```

### Document Processing Stuck

**Problem**: Document stays in "processing" status

**Solutions:**
```bash
# 1. Check backend logs for errors
docker-compose logs backend | grep ERROR

# 2. Restart backend
docker-compose restart backend

# 3. Check file size (max 50MB)
ls -lh backend/uploads/documents/

# 4. Try a simpler PDF first
```

---

## Development

### Project Structure

```
coding-test-4h-main/
├── backend/
│   ├── app/
│   │   ├── api/              # API route handlers
│   │   │   ├── chat.py       # Chat endpoint
│   │   │   └── documents.py  # Document CRUD endpoints
│   │   ├── core/             # Core configuration
│   │   │   └── config.py     # Settings and environment
│   │   ├── db/               # Database setup
│   │   │   └── session.py    # SQLAlchemy session
│   │   ├── models/           # Database models
│   │   │   ├── conversation.py
│   │   │   └── document.py
│   │   ├── services/         # Business logic
│   │   │   ├── chat_engine.py       # RAG implementation
│   │   │   ├── document_processor.py # PDF processing
│   │   │   └── vector_store.py       # Vector search
│   │   └── main.py           # FastAPI application
│   ├── uploads/              # File storage
│   │   ├── documents/
│   │   ├── images/
│   │   └── tables/
│   ├── .env                  # Environment variables
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app/                  # Next.js app router
│   │   ├── chat/
│   │   │   └── page.tsx      # Chat interface
│   │   ├── upload/
│   │   │   └── page.tsx      # Upload page
│   │   ├── layout.tsx
│   │   └── page.tsx          # Home page
│   ├── public/
│   ├── Dockerfile
│   ├── package.json
│   └── tsconfig.json
├── docker-compose.yml
└── README.md
```

### Running Tests

```bash
# Backend tests (if implemented)
docker exec -it backend bash
pytest

# Frontend tests (if implemented)
docker exec -it frontend bash
npm test
```

### Database Access

```bash
# Connect to PostgreSQL
docker exec -it postgres psql -U postgres -d docuchat

# Useful queries
SELECT id, filename, processing_status FROM documents;
SELECT COUNT(*) FROM document_chunks WHERE document_id = 1;
SELECT page_number, caption FROM document_images WHERE document_id = 1;
```

### Monitoring Logs

```bash
# Follow all logs
docker-compose logs -f

# Backend only
docker-compose logs -f backend

# Database only
docker-compose logs -f db
```

---

## Screenshots 
   
  Document upload screen
    ![alt text](image.png)

  Document processing completion screen
    ![alt text](image-1.png)

  Chat interface
    ![alt text](image-2.png)

  Answer example with images
    ![alt text](image-3.png)

  Answer example with tables
    ![alt text](image-4.png)

## Production Deployment

This application is production-ready and can be deployed to various platforms:

### Quick Deploy Options

| Platform | Best For | Cost | Setup Time | Guide |
|----------|----------|------|------------|-------|
| **Railway** | Backend + Database | $10-30/mo | 10 min | [See DEPLOYMENT.md](DEPLOYMENT.md#railway-deployment) |
| **Vercel + Railway** | Full Stack | $10-30/mo | 15 min | [See DEPLOYMENT.md](DEPLOYMENT.md#vercel--railway-deployment) |
| **Docker Production** | Self-hosted | Variable | 30 min | [See DEPLOYMENT.md](DEPLOYMENT.md#docker-production-deployment) |
| **AWS** | Enterprise Scale | $100+/mo | 60 min | [See DEPLOYMENT.md](DEPLOYMENT.md#aws-deployment) |

### What's Included

✅ **Railway Configuration** - `railway.toml`, `railway.json`, `Procfile`
✅ **Vercel Configuration** - `vercel.json` for Next.js frontend
✅ **Docker Production Setup** - `docker-compose.prod.yml`, production Dockerfiles
✅ **Nginx Reverse Proxy** - SSL termination, rate limiting, caching
✅ **Environment Templates** - `.env.production.example` with all variables
✅ **Deployment Guide** - Complete step-by-step instructions in [DEPLOYMENT.md](DEPLOYMENT.md)

### Deployment Features

- **Auto-scaling**: Ready for horizontal scaling
- **SSL/TLS**: HTTPS configuration with Let's Encrypt
- **Load Balancing**: Nginx reverse proxy with health checks
- **Rate Limiting**: API protection against abuse
- **Monitoring**: Health check endpoints
- **Backups**: Database backup strategies
- **Security**: Production-hardened configurations

For detailed deployment instructions, see **[DEPLOYMENT.md](DEPLOYMENT.md)**.

---

## Support

For questions or issues:
- **GitHub Issues**: Create an issue in the repository
- **Email**: recruitment@interopera.co
- **Documentation**: See `http://localhost:8000/docs` for API reference
- **Deployment Help**: See [DEPLOYMENT.md](DEPLOYMENT.md) for deployment guides

---

## License

This project is part of a coding assessment for InterOpera-Apps.

---

## Acknowledgments

- **Docling**: IBM's document understanding library
- **pgvector**: PostgreSQL extension for vector similarity search
- **Ollama**: Free local LLM inference
- **Sentence Transformers**: Efficient embedding generation
- **FastAPI**: Modern Python API framework
- **Next.js**: React framework for production

---

**Version**: 2.0 (Completed Implementation)
**Last Updated**: 2025-12-22
**Status**: Production Ready
