from fastapi import WebSocket
from typing import List
import logging

logger = logging.getLogger("analytics_ws_manager")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New client connected to Search Analytics WebSocket. Total active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Search Analytics Client disconnected. Total active: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        logger.info(f"Broadcasting WS message type: '{message.get('type')}' to {len(self.active_connections)} clients")
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Error broadcasting message to client: {e}")
                disconnected.append(connection)
        
        # Clean up dead connections
        for conn in disconnected:
            self.disconnect(conn)

    def broadcast_threadsafe(self, message: dict):
        import asyncio
        loop = getattr(self, "loop", None)
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)
        else:
            try:
                curr_loop = asyncio.get_event_loop()
                if curr_loop.is_running():
                    curr_loop.create_task(self.broadcast(message))
                else:
                    curr_loop.run_until_complete(self.broadcast(message))
            except RuntimeError:
                try:
                    asyncio.run(self.broadcast(message))
                except Exception as e:
                    logger.error(f"Failed to run broadcast in new loop: {e}")
            except Exception as e:
                logger.error(f"Failed to broadcast from thread: {e}")

ws_manager = ConnectionManager()
