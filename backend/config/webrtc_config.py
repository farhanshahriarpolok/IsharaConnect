"""WebRTC ICE & COTURN Configuration for NAT Traversal."""

import os
from typing import Any, Dict, List, Optional


def get_ice_servers() -> List[Dict[str, Any]]:
    """Returns configured STUN and COTURN ICE server configurations for RTCPeerConnection."""
    stun_server = os.environ.get("STUN_SERVER", "stun:stun.l.google.com:19302")
    turn_server = os.environ.get("TURN_SERVER", "turn:turn.isharaconnect.org:3478")
    turn_username = os.environ.get("TURN_USERNAME", "ishara_guest")
    turn_credential = os.environ.get("TURN_CREDENTIAL", "ishara_guest_token_2026")

    ice_servers: List[Dict[str, Any]] = [
        {"urls": [stun_server]}
    ]

    # Add secondary STUN fallbacks
    ice_servers.append({"urls": ["stun:stun1.l.google.com:19302", "stun:stun2.l.google.com:19302"]})

    # Add COTURN TURN Relay configuration
    if turn_server:
        ice_servers.append({
            "urls": [turn_server],
            "username": turn_username,
            "credential": turn_credential
        })

    return ice_servers


def get_rtc_configuration() -> Dict[str, Any]:
    """Returns standard RTCConfiguration payload with bundle policy and ice servers."""
    return {
        "iceServers": get_ice_servers(),
        "iceTransportPolicy": os.environ.get("ICE_TRANSPORT_POLICY", "all"),
        "bundlePolicy": "max-bundle",
        "rtcpMuxPolicy": "require"
    }
