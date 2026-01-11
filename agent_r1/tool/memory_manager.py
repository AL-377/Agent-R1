"""
Global Memory Manager for Medical Dialogue Memory Model
Provides RAG interface for memory retrieval and management
"""

import json
import os
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import faiss
from FlagEmbedding import FlagAutoModel


class MemoryLayer(Enum):
    """Memory layer types"""
    WORKING = "working"  # 工作记忆层
    IDENTITY = "identity"  # 患者身份与基础信息层
    HISTORY = "history"  # 历史诊疗层
    EXPERIENCE = "experience"  # 临床经验与案例层


@dataclass
class MemoryItem:
    """Single memory item"""
    id: str
    layer: str
    content: str
    timestamp: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MemoryManager:
    """
    Global Memory Manager with RAG interface
    
    Manages multi-layer memory system:
    - Working memory: current dialogue key information
    - Long-term memory: identity, history, experience layers
    """
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        index_type: str = "Flat",
        device: str = "cpu",
        top_k: int = 5
    ):
        """
        Initialize Memory Manager
        
        Args:
            embedding_model: Model for generating embeddings
            index_type: FAISS index type
            device: Device for embedding model
            top_k: Default number of results for retrieval
        """
        self.embedding_model_name = embedding_model
        self.index_type = index_type
        self.device = device
        self.top_k = top_k
        
        # Memory storage: layer -> list of MemoryItem
        self.memories: Dict[str, List[MemoryItem]] = {
            layer.value: [] for layer in MemoryLayer
        }
        
        # FAISS index for each layer
        self.indices: Dict[str, faiss.Index] = {}
        
        # Embedding model
        self.model = None
        self.embedding_dim = None
        
        # Initialize embedding model
        self._init_embedding_model()
        
        # Initialize FAISS indices
        self._init_indices()
    
    def _init_embedding_model(self):
        """Initialize the embedding model"""
        print(f"[MemoryManager] Loading embedding model: {self.embedding_model_name}")
        self.model = FlagAutoModel.from_finetuned(
            self.embedding_model_name,
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
            devices=self.device,
        )
        # Get embedding dimension (typically 1024 for bge-large-en-v1.5)
        test_embedding = self.model.encode_queries(["test"])
        self.embedding_dim = test_embedding.shape[1]
        print(f"[MemoryManager] Embedding dimension: {self.embedding_dim}")
    
    def _init_indices(self):
        """Initialize FAISS indices for each memory layer"""
        for layer in MemoryLayer:
            layer_name = layer.value
            if self.index_type == "Flat":
                index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            elif self.index_type.startswith("IVF"):
                # IVF index
                nlist = int(self.index_type.split(",")[0].replace("IVF", ""))
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            else:
                # Default to Flat
                index = faiss.IndexFlatIP(self.embedding_dim)
            
            self.indices[layer_name] = index
            print(f"[MemoryManager] Initialized {self.index_type} index for layer: {layer_name}")
    
    def insert(self, layer: str, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        Insert a new memory item
        
        Args:
            layer: Target memory layer
            content: Memory content (text)
            metadata: Optional metadata
            
        Returns:
            Memory item ID
        """
        if layer not in [l.value for l in MemoryLayer]:
            raise ValueError(f"Invalid memory layer: {layer}")
        
        # Create memory item
        memory_id = str(uuid.uuid4())
        import time
        memory_item = MemoryItem(
            id=memory_id,
            layer=layer,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        # Add to storage
        self.memories[layer].append(memory_item)
        
        # Add to FAISS index
        embedding = self.model.encode_queries([content])
        embedding = np.array(embedding, dtype=np.float32)
        faiss.normalize_L2(embedding)  # Normalize for cosine similarity
        
        # Train index if needed (for IVF)
        if isinstance(self.indices[layer], faiss.IndexIVFFlat):
            if not self.indices[layer].is_trained:
                # Need at least some vectors to train
                if len(self.memories[layer]) > 1:
                    # Train with existing embeddings
                    all_embeddings = self.model.encode_queries([m.content for m in self.memories[layer]])
                    all_embeddings = np.array(all_embeddings, dtype=np.float32)
                    faiss.normalize_L2(all_embeddings)
                    self.indices[layer].train(all_embeddings)
        
        self.indices[layer].add(embedding)
        
        return memory_id
    
    def update(self, layer: str, memory_id: str, new_content: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Update an existing memory item
        
        Args:
            layer: Target memory layer
            memory_id: Memory item ID
            new_content: New memory content
            metadata: Optional new metadata
            
        Returns:
            True if successful, False if memory not found
        """
        if layer not in [l.value for l in MemoryLayer]:
            raise ValueError(f"Invalid memory layer: {layer}")
        
        # Find memory item
        memory_items = self.memories[layer]
        item_idx = None
        for idx, item in enumerate(memory_items):
            if item.id == memory_id:
                item_idx = idx
                break
        
        if item_idx is None:
            return False
        
        # Update content
        old_content = memory_items[item_idx].content
        memory_items[item_idx].content = new_content
        if metadata is not None:
            memory_items[item_idx].metadata.update(metadata)
        
        # Rebuild index for this layer (simple approach)
        # In production, could use more efficient update methods
        self._rebuild_index(layer)
        
        return True
    
    def delete(self, layer: str, memory_id: str) -> bool:
        """
        Delete a memory item
        
        Args:
            layer: Target memory layer
            memory_id: Memory item ID
            
        Returns:
            True if successful, False if memory not found
        """
        if layer not in [l.value for l in MemoryLayer]:
            raise ValueError(f"Invalid memory layer: {layer}")
        
        # Find and remove memory item
        memory_items = self.memories[layer]
        item_idx = None
        for idx, item in enumerate(memory_items):
            if item.id == memory_id:
                item_idx = idx
                break
        
        if item_idx is None:
            return False
        
        # Remove from storage
        memory_items.pop(item_idx)
        
        # Rebuild index
        self._rebuild_index(layer)
        
        return True
    
    def _rebuild_index(self, layer: str):
        """Rebuild FAISS index for a layer"""
        memory_items = self.memories[layer]
        
        # Create new index
        if self.index_type == "Flat":
            index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type.startswith("IVF"):
            nlist = int(self.index_type.split(",")[0].replace("IVF", ""))
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.IndexFlatIP(self.embedding_dim)
        
        if len(memory_items) > 0:
            # Generate embeddings
            embeddings = self.model.encode_queries([item.content for item in memory_items])
            embeddings = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings)
            
            # Train if needed
            if isinstance(index, faiss.IndexIVFFlat) and len(memory_items) > 1:
                index.train(embeddings)
            
            # Add to index
            index.add(embeddings)
        
        self.indices[layer] = index
    
    def retrieve(self, query: str, layers: List[str] = None, top_k: int = None) -> List[Dict[str, Any]]:
        """
        RAG retrieval: search memories across specified layers
        
        Args:
            query: Search query
            layers: List of layers to search (None = all layers)
            top_k: Number of results per layer (None = use default)
            
        Returns:
            List of retrieved memory items with scores
        """
        if layers is None:
            layers = [l.value for l in MemoryLayer]
        
        if top_k is None:
            top_k = self.top_k
        
        results = []
        
        # Generate query embedding
        query_embedding = self.model.encode_queries([query])
        query_embedding = np.array(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search each layer
        for layer in layers:
            if layer not in self.indices:
                continue
            
            index = self.indices[layer]
            memory_items = self.memories[layer]
            
            if len(memory_items) == 0:
                continue
            
            # Search index
            if isinstance(index, faiss.IndexIVFFlat):
                if not index.is_trained:
                    continue
                if hasattr(index, 'nprobe'):
                    index.nprobe = min(64, index.ntotal // 10)  # Adaptive nprobe
            
            scores, indices = index.search(query_embedding, min(top_k, len(memory_items)))
            
            # Format results
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(memory_items):
                    memory_item = memory_items[idx]
                    results.append({
                        "id": memory_item.id,
                        "layer": layer,
                        "content": memory_item.content,
                        "score": float(score),
                        "timestamp": memory_item.timestamp,
                        "metadata": memory_item.metadata
                    })
        
        # Sort by score (descending)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
    
    def get_memory_state(self) -> Dict[str, Any]:
        """
        Get current memory state (for providing to agent at each step)
        
        Returns:
            Dictionary containing memory state for each layer
        """
        state = {}
        for layer in MemoryLayer:
            layer_name = layer.value
            memory_items = self.memories[layer_name]
            state[layer_name] = [
                {
                    "id": item.id,
                    "content": item.content,
                    "timestamp": item.timestamp,
                    "metadata": item.metadata
                }
                for item in memory_items
            ]
        return state
    
    def get_memory_summary(self) -> str:
        """
        Get a text summary of current memory state
        
        Returns:
            Formatted string summary
        """
        summary_parts = []
        for layer in MemoryLayer:
            layer_name = layer.value
            memory_items = self.memories[layer_name]
            if len(memory_items) > 0:
                summary_parts.append(f"\n{layer_name.upper()} MEMORY ({len(memory_items)} items):")
                for item in memory_items[-5:]:  # Show last 5 items
                    summary_parts.append(f"  - [{item.id[:8]}] {item.content[:100]}...")
        
        return "\n".join(summary_parts) if summary_parts else "No memories stored yet."
    
    def save(self, filepath: str):
        """Save memory state to file"""
        data = {
            "memories": {
                layer: [asdict(item) for item in items]
                for layer, items in self.memories.items()
            }
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        """Load memory state from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load memories
        for layer, items_data in data["memories"].items():
            self.memories[layer] = [
                MemoryItem(**item_data) for item_data in items_data
            ]
        
        # Rebuild indices
        for layer in MemoryLayer:
            self._rebuild_index(layer.value)

