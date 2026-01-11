"""
Memory Action Tools for Medical Dialogue Memory Model
Implements insert, update, delete, and wait actions
"""

from typing import Dict, List, Any
from agent_r1.tool.base import BaseTool


class MemoryInsertTool(BaseTool):
    """Tool for inserting new memory items"""
    
    name = "memory_insert"
    description = "Insert a new memory entry into the specified memory layer. Use this to add new information about the patient, symptoms, diagnosis, or clinical experience."
    parameters = {
        "type": "object",
        "properties": {
            "layer": {
                "type": "string",
                "enum": ["working", "identity", "history", "experience"],
                "description": "Target memory layer: 'working' for current dialogue key info, 'identity' for patient identity and basic info, 'history' for historical diagnosis, 'experience' for clinical experience and cases"
            },
            "content": {
                "type": "string",
                "description": "Memory content to store (text format)"
            },
            "metadata": {
                "type": "object",
                "description": "Optional metadata for the memory item",
                "additionalProperties": True
            }
        },
        "required": ["layer", "content"]
    }
    
    def __init__(self, memory_manager=None):
        """
        Initialize Memory Insert Tool
        
        Args:
            memory_manager: MemoryManager instance (will be set by environment)
        """
        super().__init__()
        self.memory_manager = memory_manager
    
    def execute(self, args: Dict, **kwargs) -> Dict[str, Any]:
        """
        Execute memory insert operation
        
        Args:
            args: Tool parameters containing:
                - layer: Target memory layer
                - content: Memory content
                - metadata: Optional metadata
            kwargs: Additional context (may contain memory_manager)
        
        Returns:
            Dict with format {"content": result_message, "success": bool, "memory_id": str}
        """
        if self.memory_manager is None:
            # Try to get from kwargs
            self.memory_manager = kwargs.get("memory_manager")
        
        if self.memory_manager is None:
            return {
                "content": "Memory manager not initialized",
                "success": False,
                "memory_id": None
            }
        
        try:
            layer = args.get("layer")
            content = args.get("content", "").strip()
            metadata = args.get("metadata")
            
            if not content:
                return {
                    "content": "Memory content cannot be empty",
                    "success": False,
                    "memory_id": None
                }
            
            # Insert memory
            memory_id = self.memory_manager.insert(
                layer=layer,
                content=content,
                metadata=metadata
            )
            
            return {
                "content": f"Successfully inserted memory into {layer} layer. Memory ID: {memory_id}",
                "success": True,
                "memory_id": memory_id
            }
        except Exception as e:
            return {
                "content": f"Failed to insert memory: {str(e)}",
                "success": False,
                "memory_id": None
            }
    
    def batch_execute(self, args_list: List[Dict], **kwargs) -> List[Dict[str, Any]]:
        """Batch execute memory insert operations"""
        return [self.execute(args, **kwargs) for args in args_list]


class MemoryUpdateTool(BaseTool):
    """Tool for updating existing memory items"""
    
    name = "memory_update"
    description = "Update an existing memory entry in the specified memory layer. Use this to modify or correct previously stored information."
    parameters = {
        "type": "object",
        "properties": {
            "layer": {
                "type": "string",
                "enum": ["working", "identity", "history", "experience"],
                "description": "Target memory layer"
            },
            "memory_id": {
                "type": "string",
                "description": "Unique identifier of the memory item to update"
            },
            "content": {
                "type": "string",
                "description": "New memory content to replace the existing content"
            },
            "metadata": {
                "type": "object",
                "description": "Optional metadata to update",
                "additionalProperties": True
            }
        },
        "required": ["layer", "memory_id", "content"]
    }
    
    def __init__(self, memory_manager=None):
        super().__init__()
        self.memory_manager = memory_manager
    
    def execute(self, args: Dict, **kwargs) -> Dict[str, Any]:
        """
        Execute memory update operation
        
        Args:
            args: Tool parameters containing:
                - layer: Target memory layer
                - memory_id: Memory item ID
                - content: New memory content
                - metadata: Optional metadata
            kwargs: Additional context
        
        Returns:
            Dict with format {"content": result_message, "success": bool}
        """
        if self.memory_manager is None:
            self.memory_manager = kwargs.get("memory_manager")
        
        if self.memory_manager is None:
            return {
                "content": "Memory manager not initialized",
                "success": False
            }
        
        try:
            layer = args.get("layer")
            memory_id = args.get("memory_id")
            content = args.get("content", "").strip()
            metadata = args.get("metadata")
            
            if not content:
                return {
                    "content": "Memory content cannot be empty",
                    "success": False
                }
            
            # Update memory
            success = self.memory_manager.update(
                layer=layer,
                memory_id=memory_id,
                new_content=content,
                metadata=metadata
            )
            
            if success:
                return {
                    "content": f"Successfully updated memory {memory_id} in {layer} layer",
                    "success": True
                }
            else:
                return {
                    "content": f"Memory {memory_id} not found in {layer} layer",
                    "success": False
                }
        except Exception as e:
            return {
                "content": f"Failed to update memory: {str(e)}",
                "success": False
            }
    
    def batch_execute(self, args_list: List[Dict], **kwargs) -> List[Dict[str, Any]]:
        """Batch execute memory update operations"""
        return [self.execute(args, **kwargs) for args in args_list]


class MemoryDeleteTool(BaseTool):
    """Tool for deleting memory items"""
    
    name = "memory_delete"
    description = "Delete an existing memory entry from the specified memory layer. Use this to remove outdated or incorrect information."
    parameters = {
        "type": "object",
        "properties": {
            "layer": {
                "type": "string",
                "enum": ["working", "identity", "history", "experience"],
                "description": "Target memory layer"
            },
            "memory_id": {
                "type": "string",
                "description": "Unique identifier of the memory item to delete"
            }
        },
        "required": ["layer", "memory_id"]
    }
    
    def __init__(self, memory_manager=None):
        super().__init__()
        self.memory_manager = memory_manager
    
    def execute(self, args: Dict, **kwargs) -> Dict[str, Any]:
        """
        Execute memory delete operation
        
        Args:
            args: Tool parameters containing:
                - layer: Target memory layer
                - memory_id: Memory item ID
            kwargs: Additional context
        
        Returns:
            Dict with format {"content": result_message, "success": bool}
        """
        if self.memory_manager is None:
            self.memory_manager = kwargs.get("memory_manager")
        
        if self.memory_manager is None:
            return {
                "content": "Memory manager not initialized",
                "success": False
            }
        
        try:
            layer = args.get("layer")
            memory_id = args.get("memory_id")
            
            # Delete memory
            success = self.memory_manager.delete(
                layer=layer,
                memory_id=memory_id
            )
            
            if success:
                return {
                    "content": f"Successfully deleted memory {memory_id} from {layer} layer",
                    "success": True
                }
            else:
                return {
                    "content": f"Memory {memory_id} not found in {layer} layer",
                    "success": False
                }
        except Exception as e:
            return {
                "content": f"Failed to delete memory: {str(e)}",
                "success": False
            }
    
    def batch_execute(self, args_list: List[Dict], **kwargs) -> List[Dict[str, Any]]:
        """Batch execute memory delete operations"""
        return [self.execute(args, **kwargs) for args in args_list]


class MemoryWaitTool(BaseTool):
    """Tool for wait action (no operation)"""
    
    name = "memory_wait"
    description = "Wait action - indicates that no memory operation is needed at this step. Use this when the current dialogue does not require any memory updates."
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    def __init__(self, memory_manager=None):
        super().__init__()
        self.memory_manager = memory_manager
    
    def execute(self, args: Dict, **kwargs) -> Dict[str, Any]:
        """
        Execute wait operation (no-op)
        
        Args:
            args: Empty dict (no parameters needed)
            kwargs: Additional context
        
        Returns:
            Dict with format {"content": result_message, "success": bool}
        """
        return {
            "content": "Wait action executed - no memory operation performed",
            "success": True
        }
    
    def batch_execute(self, args_list: List[Dict], **kwargs) -> List[Dict[str, Any]]:
        """Batch execute wait operations"""
        return [self.execute(args, **kwargs) for args in args_list]

