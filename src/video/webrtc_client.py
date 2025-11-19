# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/18/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Low-level WebRTC client for Spot PTZ camera implementing SDP negotiation,
# RTCPeerConnection management, ICE connection handling, and video frame queuing
# Acknowledgements: Adapted from Boston Dynamics SDK webrtc_client.py example,
# aiortc RTCPeerConnection documentation, Claude for async event handler design

"""WebRTC client for Spot PTZ camera streaming.

Adapted from Boston Dynamics SDK examples (spot-sdk/python/examples/spot_cam/webrtc_client.py)
with modifications for PTZ-specific streaming.
"""
import asyncio
import base64
import logging
from typing import Optional

import requests
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCConfiguration
from aiortc.contrib.media import MediaBlackhole

logger = logging.getLogger(__name__)

DEFAULT_WEB_REQUEST_TIMEOUT = 10.0


class SpotPtzVideoTrack(MediaStreamTrack):
    """Video track wrapper that puts frames into a queue."""
    
    def __init__(self, track, queue):
        super().__init__()
        self.track = track
        self.queue = queue
    
    async def recv(self):
        """Receive frame and put in queue."""
        frame = await self.track.recv()
        await self.queue.put(frame)
        return frame


class SpotPtzWebRTCClient:
    """WebRTC client for Spot PTZ camera streaming.
    
    Handles SDP negotiation and manages RTCPeerConnection for video streaming.
    """
    
    def __init__(
        self,
        hostname: str,
        token: str,
        sdp_port: int = 31102,
        sdp_filename: str = "h264.sdp",
        cam_ssl_cert: Optional[str] = False,
    ):
        """Initialize WebRTC client.
        
        Args:
            hostname: Spot robot hostname/IP
            token: Authentication token from robot.user_token
            sdp_port: Port for SDP negotiation (default: 31102)
            sdp_filename: SDP endpoint filename (default: h264.sdp)
            cam_ssl_cert: Path to SSL cert, or False to disable verification
        """
        # WebRTC configuration (no STUN/TURN servers needed for local connection)
        rtc_config = RTCConfiguration(iceServers=[])
        self.pc = RTCPeerConnection(configuration=rtc_config)
        
        # Frame queues
        self.video_frame_queue = asyncio.Queue()
        self.audio_frame_queue = asyncio.Queue()
        
        # Connection parameters
        self.hostname = hostname
        self.token = token
        self.sdp_port = sdp_port
        self.sdp_filename = sdp_filename
        self.cam_ssl_cert = cam_ssl_cert
        
        # Audio sink (discard audio frames)
        self.audio_sink: Optional[MediaBlackhole] = None
        self.audio_sink_task: Optional[asyncio.Task] = None
        
        logger.debug(f"WebRTC client initialized for {hostname}:{sdp_port}/{sdp_filename}")
    
    def _get_sdp_offer_from_spot_cam(self) -> tuple[str, str]:
        """Request SDP offer from Spot CAM.
        
        Returns:
            Tuple of (offer_id, sdp_offer_text)
        """
        headers = {'Authorization': f'Bearer {self.token}'}
        server_url = f'https://{self.hostname}:{self.sdp_port}/{self.sdp_filename}'
        
        logger.debug(f"Requesting SDP offer from {server_url}")
        response = requests.get(
            server_url,
            verify=self.cam_ssl_cert,
            headers=headers,
            timeout=DEFAULT_WEB_REQUEST_TIMEOUT
        )
        response.raise_for_status()
        
        result = response.json()
        offer_id = result['id']
        sdp_offer = base64.b64decode(result['sdp']).decode()
        
        logger.debug(f"Received SDP offer (id={offer_id})")
        return offer_id, sdp_offer
    
    def _send_sdp_answer_to_spot_cam(self, offer_id: str, sdp_answer: bytes):
        """Send SDP answer to Spot CAM.
        
        Args:
            offer_id: Offer ID from original SDP request
            sdp_answer: SDP answer as bytes
        """
        headers = {'Authorization': f'Bearer {self.token}'}
        server_url = f'https://{self.hostname}:{self.sdp_port}/{self.sdp_filename}'
        
        payload = {
            'id': offer_id,
            'sdp': base64.b64encode(sdp_answer).decode('utf8')
        }
        
        logger.debug(f"Sending SDP answer to {server_url}")
        response = requests.post(
            server_url,
            verify=self.cam_ssl_cert,
            json=payload,
            headers=headers,
            timeout=DEFAULT_WEB_REQUEST_TIMEOUT
        )
        response.raise_for_status()
        logger.debug("SDP answer sent successfully")
    
    async def start(self):
        """Start WebRTC connection and setup event handlers."""
        # Get SDP offer from Spot CAM
        offer_id, sdp_offer = self._get_sdp_offer_from_spot_cam()
        
        # Setup event handlers
        @self.pc.on('icegatheringstatechange')
        async def _on_ice_gathering_state_change():
            logger.debug(f'ICE gathering state: {self.pc.iceGatheringState}')
        
        @self.pc.on('signalingstatechange')
        async def _on_signaling_state_change():
            logger.debug(f'Signaling state: {self.pc.signalingState}')
        
        @self.pc.on('icecandidate')
        async def _on_ice_candidate(event):
            if event.candidate:
                logger.debug(f'ICE candidate: {event.candidate}')
        
        @self.pc.on('iceconnectionstatechange')
        async def _on_ice_connection_state_change():
            state = self.pc.iceConnectionState
            logger.info(f'ICE connection state: {state}')
            
            if state == 'checking':
                # Send SDP answer when ICE checking starts
                self._send_sdp_answer_to_spot_cam(
                    offer_id,
                    self.pc.localDescription.sdp.encode()
                )
            elif state == 'failed':
                logger.error("ICE connection failed")
            elif state == 'closed':
                logger.info("ICE connection closed")
        
        @self.pc.on('track')
        def _on_track(track):
            logger.info(f'Received track: {track.kind}')
            
            if track.kind == 'video':
                # Wrap video track to put frames in queue
                video_track = SpotPtzVideoTrack(track, self.video_frame_queue)
                video_track.kind = 'video'
                self.pc.addTrack(video_track)
                logger.info("Video track added to queue")
            
            elif track.kind == 'audio':
                # Discard audio frames (not needed for facial recognition)
                self.audio_sink = MediaBlackhole()
                self.audio_sink.addTrack(track)
                loop = asyncio.get_event_loop()
                self.audio_sink_task = loop.create_task(self.audio_sink.start())
                logger.debug("Audio track discarded")
        
        # Set remote description (SDP offer)
        desc = RTCSessionDescription(sdp_offer, 'offer')
        await self.pc.setRemoteDescription(desc)
        
        # Create and set local description (SDP answer)
        sdp_answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(sdp_answer)
        
        logger.info("WebRTC client started, waiting for connection...")
    
    async def stop(self):
        """Stop WebRTC connection and cleanup."""
        logger.info("Stopping WebRTC client...")
        
        # Stop audio sink if running
        if self.audio_sink_task:
            self.audio_sink_task.cancel()
            try:
                await self.audio_sink_task
            except asyncio.CancelledError:
                pass
        
        # Close peer connection
        await self.pc.close()
        
        logger.info("WebRTC client stopped")
