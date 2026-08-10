import logging
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from app.features.search_analytics.websocket.manager import ws_manager
from app.features.search_analytics.repositories.analytics_repository import analytics_repository
from app.features.search_analytics.auth.auth_handler import decode_access_token

logger = logging.getLogger("analytics_ws_router")

router = APIRouter()

@router.websocket("/analytics/search/realtime")
async def websocket_realtime_analytics(websocket: WebSocket, token: Optional[str] = Query(None)):
    """
    WebSocket endpoint for real-time dashboard updates.
    Handles connection lifecycle, records audit events with authenticated username, and answers heartbeat pings.
    """
    client_ip = websocket.client.host
    username = "anonymous"

    if token:
        payload = decode_access_token(token)
        if payload and payload.get("sub"):
            username = payload.get("sub")

    await ws_manager.connect(websocket)
    
    # Audit log connection
    analytics_repository.insert_audit_event_direct(
        "WEBSOCKET_CONNECT", 
        username, 
        client_ip, 
        f"Websocket client ({username}) connected to realtime search channel."
    )
    
    try:
        while True:
            # Block waiting for text message from client
            # Standard heartbeat protocol: client sends 'ping', backend replies 'pong'
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        analytics_repository.insert_audit_event_direct(
            "WEBSOCKET_DISCONNECT", 
            username, 
            client_ip, 
            f"Websocket client ({username}) disconnected."
        )
    except Exception as e:
        ws_manager.disconnect(websocket)
        logger.warning(f"Error in websocket connection: {e}")
        analytics_repository.insert_audit_event_direct(
            "WEBSOCKET_ERROR", 
            username, 
            client_ip, 
            f"Websocket connection error: {str(e)}"
        )

