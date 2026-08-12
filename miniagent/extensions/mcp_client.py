"""Lightweight MCP (Model Context Protocol) client for MiniAgent.

Connects to MCP-compatible tool servers via stdio or SSE transport,
discovers their tools, and registers them as regular MiniAgent tools.

Usage:
    from miniagent.mcp_client import load_mcp_tools
    
    tools = load_mcp_tools("npx @anthropic/mcp-server-filesystem /tmp")
    for tool in tools:
        agent.add_tool(tool)
"""

from __future__ import annotations

import json
import subprocess
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional

from ..logger import get_logger
from rich.console import Console          # 新增

logger = get_logger(__name__)
console = Console()                       # 新建控制台实例


class MCPClient:
    """A minimal MCP client using stdio (JSON-RPC over stdin/stdout)."""

    def __init__(self, command: str, env: Optional[Dict[str, str]] = None):
        """
        Args:
            command: Shell command to start the MCP server (e.g. "npx @anthropic/mcp-server-filesystem /tmp").
            env: Optional environment variables for the server process.
        """
        self.command = command
        self.env = env
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._responses: Dict[str, Any] = {}
        self._reader_thread: Optional[threading.Thread] = None


    '''
    def start(self) -> None:
        """Start the MCP server process."""
        import os
        import shlex
        merged_env = {**os.environ, **(self.env or {})}
        self._process = subprocess.Popen(
            shlex.split(self.command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            #stderr=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=merged_env,
        )
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

        # Initialize the MCP session
        self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "miniagent", "version": "0.1.0"},
        })
        # Send initialized notification
        self._send_notification("initialized", {})
        logger.info(f"MCP server started: {self.command}")
    '''

    #重写start
    def start(self) -> None:
        """启动 MCP 服务器进程（使用 UTF-8 编码，分离 stderr）"""
        import os
        import shlex
        import time
        import threading
        import subprocess

        #print("[DEBUG] start() called")
        console.print("[dim][DEBUG] start() called[/dim]")

        # 合并环境变量
        merged_env = {**os.environ, **(self.env or {})}
    
        # 构建命令（保留原样）
        #print(f"[DEBUG] command: {self.command}")
        console.print(f"[dim][DEBUG] command: {self.command}[/dim]")
    
        # 启动子进程：强制 UTF-8 编码，分离 stderr 以避免污染 stdout
        self._process = subprocess.Popen(
            shlex.split(self.command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,        # 单独捕获 stderr，不混入 stdout
            text=True,
            env=merged_env,
            encoding='utf-8',              # 强制 UTF-8 解码 stdout
            errors='replace'               # 遇到非法字节替换为 �（防止崩溃）
        )
    
        #print(f"[DEBUG] process started, pid={self._process.pid}")
        console.print(f"[dim][DEBUG] process started, pid={self._process.pid}[/dim]")

        # 启动 stdout 读取线程（处理 JSON-RPC 响应）
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()
        #print("[DEBUG] reader thread started")
        console.print("[dim][DEBUG] reader thread started[/dim]")

        # 启动 stderr 读取线程（打印日志，便于调试）
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()
        #print("[DEBUG] stderr reader thread started")
        console.print("[dim][DEBUG] stderr reader thread started[/dim]")

        # 等待一下让进程稳定
        time.sleep(0.5)
        #print(f"[DEBUG] After sleep, process poll={self._process.poll()}")
        console.print(f"[dim][DEBUG] After sleep, process poll={self._process.poll()}[/dim]")

        # 给子进程更多启动时间（原代码 1s）
        time.sleep(1)
    
        # 发送初始化请求
        #print("[DEBUG] sending initialize...")
        console.print("[dim][DEBUG] sending initialize...[/dim]")

        self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "miniagent", "version": "0.1.0"},
        })
        #print("[DEBUG] initialize sent, sending initialized notification...")
        console.print("[dim][DEBUG] initialize sent, sending initialized notification...[/dim]")

        self._send_notification("initialized", {})
        #print("[DEBUG] start() finished")
        console.print("[dim][DEBUG] start() finished[/dim]")


    #修改结束






    def stop(self) -> None:
        """Stop the MCP server process and clean up resources."""
        if self._process:
            try:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            finally:
                self._process = None
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2)
        logger.info("MCP server stopped")

    def list_tools(self) -> List[Dict[str, Any]]:
        """Discover available tools from the MCP server."""
        result = self._send_request("tools/list", {})
        return result.get("tools", []) if result else []

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        result = self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        if not result:
            return {"error": "No response from MCP server"}

        # MCP returns content as a list of {type, text} objects
        contents = result.get("content", [])
        texts = [c.get("text", "") for c in contents if c.get("type") == "text"]
        return "\n".join(texts) if texts else str(result)



    '''
    def _send_request(self, method: str, params: Dict[str, Any]) -> Optional[Dict]:
        """Send a JSON-RPC request and wait for response."""
        req_id = str(uuid.uuid4())[:8]
        message = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        self._write(message)
        logger.info(f"[SENT] {method} (id={req_id})")          # ← 添加：确认请求已发送

        # Wait for response (simple polling, max 120s)
        import time
        for _ in range(1200):
            with self._lock:
                if req_id in self._responses:
                    resp = self._responses.pop(req_id)
                    if "error" in resp:
                        logger.error(f"MCP error: {resp['error']}")
                        return None
                    logger.info(f"[RECV] {method} response (id={req_id})")  # ← 添加：收到响应
                    return resp.get("result")
            time.sleep(0.1)

        logger.error(f"MCP request timed out: {method}")
        return None
    '''

    #修改
    def _send_request(self, method: str, params: Dict[str, Any], timeout: int = 120) -> Optional[Dict]:
        """
        Send a JSON-RPC request and wait for response.

        Args:
            method: Method name (e.g., "initialize", "tools/list").
            params: Request parameters.
            timeout: Maximum seconds to wait for response.

        Returns:
            Result dictionary or None on error/timeout.
        """
        import time
        req_id = str(uuid.uuid4())[:8]
        message = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }

        # Send request
        self._write(message)
        logger.info(f"[SENT] {method} (id={req_id})")

        # Poll for response with timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._lock:
                if req_id in self._responses:
                    resp = self._responses.pop(req_id)
                    if "error" in resp:
                        logger.error(f"MCP error for {method}: {resp['error']}")
                        return None
                    logger.info(f"[RECV] {method} response (id={req_id})")
                    return resp.get("result")
            time.sleep(0.05)  # 50ms polling interval

        # Timeout reached
        logger.error(f"MCP request timed out: {method} (waited {timeout}s)")
        # Check if process is still alive
        if self._process and self._process.poll() is not None:
            logger.error(f"MCP process exited with code {self._process.returncode}")
        return None


    #修改结束





    def _send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        self._write(message)



    '''
    def _write(self, message: Dict) -> None:
        """Write a JSON-RPC message to the server's stdin."""
        if not self._process or not self._process.stdin:
            return
        content = json.dumps(message)
        # MCP uses Content-Length header framing
        frame = f"Content-Length: {len(content)}\r\n\r\n{content}"
        try:
            self._process.stdin.write(frame)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError):
            logger.error("MCP server stdin broken")
    '''


    #新修改
    def _write(self, message: Dict) -> None:
        """Write a JSON-RPC message as a single JSON line (no Content-Length)."""
        if not self._process or not self._process.stdin:
            return
        try:
            self._process.stdin.write(json.dumps(message) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            logger.error(f"Failed to write to MCP server stdin: {e}")


    #修改结束


    

    '''
    def _read_loop(self) -> None:
        """Background thread: read JSON-RPC responses from stdout."""
        if not self._process or not self._process.stdout:
            return

        while self._process and self._process.poll() is None:
            try:
                # 读取一行（可能是 Content-Length 头或纯 JSON）
                line = self._process.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if line.startswith("Content-Length:"):
                    # 标准 MCP 帧格式
                    content_length = int(line.split(":")[1].strip())
                    self._process.stdout.readline()  # 空行
                    content = self._process.stdout.read(content_length)
                    if content:
                        msg = json.loads(content)
                        msg_id = msg.get("id")
                        if msg_id:
                            with self._lock:
                                self._responses[msg_id] = msg
                else:
                    # 纯 JSON 行（不带头）
                    try:
                        msg = json.loads(line)
                        msg_id = msg.get("id")
                        if msg_id:
                            with self._lock:
                                self._responses[msg_id] = msg
                    except json.JSONDecodeError:
                        logger.debug(f"Unparseable output: {line}")
            except Exception:
                logger.debug("MCP read error", exc_info=True)
                break
    '''
    
    #新增代码
    def _read_loop(self) -> None:
        """Read JSON-RPC responses line by line from the server's stdout."""
        if not self._process or not self._process.stdout:
            return
        while self._process.poll() is None:
            try:
                line = self._process.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    msg_id = msg.get("id")
                    if msg_id is not None:
                        with self._lock:
                            self._responses[msg_id] = msg
                except json.JSONDecodeError:
                    # 忽略非 JSON 行（mcp-remote 可能打印日志到 stdout）
                    logger.debug(f"Ignored non-JSON line: {line}")
            except Exception as e:
                logger.error(f"Read loop error: {e}")
                break

    #新增代码结束


    #新增方法
    def _read_stderr(self) -> None:
        """读取子进程的 stderr 并打印日志（避免阻塞）"""
        if not self._process or not self._process.stderr:
            return
        try:
            for line in iter(self._process.stderr.readline, ''):
                if line:
                    # 可以加时间戳，或直接输出
                    #print(f"[MCP STDERR] {line.rstrip()}")
                    console.print(f"[dim][MCP STDERR] {line.rstrip()}[/dim]")
        except Exception as e:
            #print(f"[MCP STDERR] reader error: {e}")
            console.print(f"[dim][MCP STDERR] reader error: {e}[/dim]")

    #新增方法结束
              


def load_mcp_tools(command: str, env: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """
    Start an MCP server and convert its tools into MiniAgent tool dicts.
    
    Args:
        command: Shell command to start the MCP server.
        env: Optional environment variables.
        
    Returns:
        List of tool dicts compatible with agent.add_tool().
    """
    client = MCPClient(command, env)
    try:
        client.start()
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        return []

    mcp_tools = client.list_tools()
    agent_tools = []

    for t in mcp_tools:
        tool_name = t["name"]
        # Create a closure that captures the client and tool name
        def _make_executor(c: MCPClient, n: str) -> Callable:
            def executor(**kwargs: Any) -> Any:
                return c.call_tool(n, kwargs)
            executor.__name__ = n
            executor.__doc__ = t.get("description", n)
            return executor

        agent_tools.append({
            "name": tool_name,
            "description": t.get("description", tool_name),
            "parameters": t.get("inputSchema", {"type": "object", "properties": {}}),
            "executor": _make_executor(client, tool_name),
            "_mcp_client": client,  # keep reference so GC doesn't kill the process
        })

    logger.info(f"Loaded {len(agent_tools)} tools from MCP server: {command}")
    return agent_tools
