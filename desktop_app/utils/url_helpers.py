"""URL resolution utilities for unified network access."""

def get_http_url(base_url: str, path: str = "") -> str:
    """Resolve a base URL to an HTTP(S) schema.
    
    Args:
        base_url: The configured server URL (e.g. 'ws://localhost:8000', 'https://api.com')
        path: Optional path to append (e.g. '/api/v1/propose')
        
    Returns:
        Formatted HTTP(S) url
    """
    url = base_url.strip()
    if url.startswith("ws://"):
        url = url.replace("ws://", "http://", 1)
    elif url.startswith("wss://"):
        url = url.replace("wss://", "https://", 1)
    elif not url.startswith("http"):
        url = f"http://{url}"
        
    if path:
        if not path.startswith("/"):
            path = "/" + path
        url = url.rstrip("/") + path
        
    return url

def get_ws_url(base_url: str, path: str = "") -> str:
    """Resolve a base URL to a WebSocket (WS/WSS) schema.
    
    Args:
        base_url: The configured server URL
        path: Optional path to append
        
    Returns:
        Formatted WS(S) url
    """
    url = base_url.strip()
    if url.startswith("http://"):
        url = url.replace("http://", "ws://", 1)
    elif url.startswith("https://"):
        url = url.replace("https://", "wss://", 1)
    elif not url.startswith("ws"):
        url = f"ws://{url}"
        
    if path:
        if not path.startswith("/"):
            path = "/" + path
        url = url.rstrip("/") + path
        
    return url
