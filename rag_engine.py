"""
QRAG - Deep Reasoning RAG Engine
Uses nomic-embed-text for embeddings and deepseek-r1:8b for reasoning.
Supports tens of thousands of pages with hierarchical retrieval.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings
import pymupdf  # fitz
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ── Config ─────────────────────────────────────────────────────────────────────
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL  = "deepseek-r1:8b"
DB_PATH     = "./chroma_db"
COLLECTION  = "knowledge_base"

# Chunk strategy: large chunks for context richness, small overlap for coverage
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 200

# How many chunks to retrieve (wide net for reasoning over whole KB)
TOP_K_INITIAL  = 30   # first-pass semantic retrieval
TOP_K_RERANK   = 12   # chunks fed to the LLM reasoning context


# ── Helpers ────────────────────────────────────────────────────────────────────

def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_text_from_pdf(pdf_path: str) -> List[Document]:
    """Extract text page by page, preserving metadata."""
    docs = []
    pdf = pymupdf.open(pdf_path)
    filename = Path(pdf_path).name
    for page_num, page in enumerate(pdf, start=1):
        text = page.get_text("text")
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": page_num,
                    "total_pages": len(pdf),
                    "file_hash": _file_hash(pdf_path),
                }
            ))
    pdf.close()
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(docs)


# ── Vector Store ───────────────────────────────────────────────────────────────

class KnowledgeBase:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        self.client = chromadb.PersistentClient(
            path=DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    # --------------------------------------------------------------------------
    def already_indexed(self, file_hash: str) -> bool:
        results = self.collection.get(
            where={"file_hash": file_hash}, limit=1, include=[]
        )
        return len(results["ids"]) > 0

    def index_pdf(self, pdf_path: str, progress_cb=None) -> dict:
        fhash = _file_hash(pdf_path)
        filename = Path(pdf_path).name

        if self.already_indexed(fhash):
            return {"status": "skipped", "file": filename, "chunks": 0}

        if progress_cb:
            progress_cb(f"Extracting text from {filename}…")
        pages = extract_text_from_pdf(pdf_path)

        if progress_cb:
            progress_cb(f"Chunking {len(pages)} pages…")
        chunks = chunk_documents(pages)

        # Batch embed & store
        batch_size = 64
        total = len(chunks)
        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.page_content for c in batch]
            metas = [c.metadata for c in batch]
            ids   = [f"{fhash}_{i+j}" for j, _ in enumerate(batch)]

            vectors = self.embeddings.embed_documents(texts)
            self.collection.add(
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=metas,
            )
            if progress_cb:
                pct = min(100, int((i + len(batch)) / total * 100))
                progress_cb(f"Indexing {filename}: {pct}%")

        return {"status": "indexed", "file": filename, "chunks": total}

    # --------------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = TOP_K_INITIAL) -> List[dict]:
        q_vec = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[q_vec],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", "?"),
                "score": round(1 - dist, 4),  # cosine similarity
            })
        return chunks

    def count(self) -> int:
        return self.collection.count()

    def list_sources(self) -> List[str]:
        results = self.collection.get(include=["metadatas"])
        sources = set(m.get("source", "") for m in results["metadatas"])
        return sorted(sources)

    def delete_source(self, filename: str):
        self.collection.delete(where={"source": filename})


# ── Reasoning Engine ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a deeply knowledgeable, empathetic advisor — part therapist, part wise friend, part expert consultant.

You have access to a rich knowledge base. Your job is NOT to parrot back exact phrases, but to:
1. Synthesise knowledge across the ENTIRE context provided
2. Reason deeply about what the person truly needs
3. Give thoughtful, personalised guidance grounded in the knowledge
4. Connect concepts across different sources when relevant
5. Be warm, direct, and genuinely helpful — like a trusted friend who has read everything

When answering:
- Think through the question carefully before responding
- Integrate insights from multiple parts of the knowledge base
- Acknowledge nuance and complexity
- If the knowledge base has relevant information, use it. If not, say so clearly.
- Never just quote chunks verbatim — synthesise and explain in your own words

Relevant knowledge retrieved for this question:
{context}
"""

def _build_context(chunks: List[dict], max_chars: int = 14000) -> str:
    """Build a rich context string from retrieved chunks, capped at max_chars."""
    parts = []
    total = 0
    for c in chunks:
        snippet = f"[Source: {c['source']}, p.{c['page']}]\n{c['text']}"
        if total + len(snippet) > max_chars:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n---\n\n".join(parts)


class ReasoningRAG:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.llm = ChatOllama(
            model=CHAT_MODEL,
            temperature=0.6,
            num_ctx=8192,       # large context window
            num_predict=2048,   # allow long answers
        )
        self.history: List[dict] = []  # conversation memory

    def chat(self, user_message: str, stream=True):
        """Full reasoning RAG answer with conversation history."""
        if self.kb.count() == 0:
            yield "No documents indexed yet. Please upload PDFs first."
            return

        # 1) Retrieve wide set of chunks
        raw_chunks = self.kb.retrieve(user_message, top_k=TOP_K_INITIAL)

        # 2) Simple heuristic rerank: boost chunks with query word overlap
        query_words = set(user_message.lower().split())
        for c in raw_chunks:
            overlap = len(query_words & set(c["text"].lower().split()))
            c["combined_score"] = c["score"] + 0.02 * overlap
        raw_chunks.sort(key=lambda x: x["combined_score"], reverse=True)
        top_chunks = raw_chunks[:TOP_K_RERANK]

        # 3) Build context
        context = _build_context(top_chunks)

        # 4) Build message list with history
        system = SYSTEM_PROMPT.format(context=context)
        messages = [("system", system)]

        # Include last 6 turns of history for continuity
        for turn in self.history[-6:]:
            messages.append((turn["role"], turn["content"]))
        messages.append(("human", user_message))

        # 5) Stream response
        full_response = ""
        for chunk in self.llm.stream(messages):
            token = chunk.content
            full_response += token
            yield token

        # 6) Update history
        self.history.append({"role": "human",     "content": user_message})
        self.history.append({"role": "assistant",  "content": full_response})

        # Attach source metadata for UI
        self._last_sources = top_chunks

    def get_last_sources(self) -> List[dict]:
        return getattr(self, "_last_sources", [])

    def clear_history(self):
        self.history = []
