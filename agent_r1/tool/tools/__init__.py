def _default_tool(name):
    if name == "search":
        from agent_r1.tool.tools.search_tool import SearchTool
        return SearchTool()
    elif name == "wiki_search":
        from agent_r1.tool.tools.wiki_search_tool import WikiSearchTool
        return WikiSearchTool()
    elif name == "python":
        from agent_r1.tool.tools.python_tool import PythonTool
        return PythonTool()
    elif name == "memory_insert":
        from agent_r1.tool.tools.memory_tools import MemoryInsertTool
        return MemoryInsertTool()
    elif name == "memory_update":
        from agent_r1.tool.tools.memory_tools import MemoryUpdateTool
        return MemoryUpdateTool()
    elif name == "memory_delete":
        from agent_r1.tool.tools.memory_tools import MemoryDeleteTool
        return MemoryDeleteTool()
    elif name == "memory_wait":
        from agent_r1.tool.tools.memory_tools import MemoryWaitTool
        return MemoryWaitTool()
    else:
        raise NotImplementedError(f"Tool {name} not implemented")