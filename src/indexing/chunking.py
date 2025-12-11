"""Intelligent chunking module for product documents."""
import json
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter


class ProductChunker:
    """Intelligently chunks product documents for optimal embedding."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: Custom separators for splitting (defaults to intelligent defaults)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default separators optimized for product data
        if separators is None:
            separators = [
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentences
                " ",     # Words
                ""       # Characters (fallback)
            ]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False
        )
    
    def chunk_documents(
        self,
        documents: List[Document],
        preserve_metadata: bool = True
    ) -> List[Document]:
        """
        Chunk documents intelligently while preserving metadata.
        
        Args:
            documents: List of LangChain Document objects
            preserve_metadata: Whether to preserve original metadata in chunks
            
        Returns:
            List of chunked Document objects
        """
        chunked_docs = []
        
        for doc in documents:
            # Split the document
            chunks = self.text_splitter.split_text(doc.page_content)
            
            # Create new Document objects for each chunk
            for i, chunk_text in enumerate(chunks):
                # Preserve all original metadata
                chunk_metadata = doc.metadata.copy() if preserve_metadata else {}
                
                # Add chunk-specific metadata
                chunk_metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk_text)
                })
                
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                )
                chunked_docs.append(chunk_doc)
        
        return chunked_docs
    
    def chunk_product_text(
        self,
        text: str,
        metadata: dict,
        preserve_metadata: bool = True
    ) -> List[Document]:
        """
        Chunk a single product text with metadata.
        
        Args:
            text: Product text to chunk
            metadata: Metadata to attach to chunks
            preserve_metadata: Whether to preserve metadata in chunks
            
        Returns:
            List of chunked Document objects
        """
        doc = Document(
            page_content=text,
            metadata=metadata
        )
        return self.chunk_documents([doc], preserve_metadata=preserve_metadata)
    
    def get_chunk_statistics(
        self,
        chunked_documents: List[Document]
    ) -> dict:
        """
        Get statistics about the chunking process.
        
        Args:
            chunked_documents: List of chunked documents
            
        Returns:
            Dictionary with chunking statistics
        """
        if not chunked_documents:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0
            }
        
        chunk_sizes = [len(doc.page_content) for doc in chunked_documents]
        
        return {
            "total_chunks": len(chunked_documents),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes)
        }
    
    def save_chunks_to_jsonl(
        self,
        chunked_documents: List[Document],
        file_path: str = "data/jsonl/product_chunks.jsonl"
    ) -> int:
        """
        Save chunked documents to a JSONL file.
        
        Args:
            chunked_documents: List of chunked Document objects
            file_path: Path to save the JSONL file
            
        Returns:
            Number of chunks saved
        """
        file_path_obj = Path(file_path)
        # Ensure parent directory exists
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        chunks_saved = 0
        with open(file_path_obj, 'w', encoding='utf-8') as f:
            for doc in chunked_documents:
                # Convert Document to dictionary
                chunk_data = {
                    "text": doc.page_content,
                    "metadata": doc.metadata
                }
                # Write as JSON line
                f.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')
                chunks_saved += 1
        
        return chunks_saved


class MarkdownChunker:
    """
    Chunker for markdown documents (for future handbook indexing).
    This is structured but not yet implemented as per requirements.
    """
    
    def __init__(self):
        """Initialize markdown chunker."""
        # Headers to split on (for future use)
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )
    
    def chunk_markdown(
        self,
        markdown_text: str,
        metadata: Optional[dict] = None,
        preserve_metadata: bool = True
    ) -> List[Document]:
        """
        Chunk markdown documents by headers.
        
        Args:
            markdown_text: Markdown text to chunk
            metadata: Optional metadata to attach to chunks
            preserve_metadata: Whether to preserve metadata in chunks
            
        Returns:
            List of chunked Document objects
        """
        # Split markdown by headers - returns List[Document]
        header_chunks = self.markdown_splitter.split_text(markdown_text)
        
        chunked_docs = []
        for i, chunk_doc in enumerate(header_chunks):
            # Start with original metadata if provided
            chunk_metadata = metadata.copy() if metadata and preserve_metadata else {}
            
            # Add header information from markdown splitter (if present)
            if hasattr(chunk_doc, 'metadata') and chunk_doc.metadata:
                chunk_metadata.update(chunk_doc.metadata)
            
            # Add chunk-specific metadata
            chunk_metadata.update({
                "chunk_index": i,
                "total_chunks": len(header_chunks),
                "chunk_size": len(chunk_doc.page_content)
            })
            
            # Create new document with combined metadata
            final_doc = Document(
                page_content=chunk_doc.page_content,
                metadata=chunk_metadata
            )
            chunked_docs.append(final_doc)
        
        return chunked_docs

