import os
import os.path

import socketio

from openhands.server.app import app as base_app
from openhands.server.listen_socket import sio
from openhands.server.middleware import (
    CacheControlMiddleware,
    InMemoryRateLimiter,
    LocalhostCORSMiddleware,
    RateLimitMiddleware,
)
from openhands.server.static import SPAStaticFiles

serve_frontend = os.getenv('SERVE_FRONTEND', 'true').lower() == 'true'
frontend_build_dir = './frontend/build'
if serve_frontend and os.path.isdir(frontend_build_dir):
    base_app.mount(
        '/', SPAStaticFiles(directory=frontend_build_dir, html=True), name='dist'
    )
else:
    # Skip mounting when frontend is not present or serving is disabled.
    # This allows CLI-only setups to remove the frontend folder safely.
    pass

base_app.add_middleware(LocalhostCORSMiddleware)
base_app.add_middleware(CacheControlMiddleware)
base_app.add_middleware(
    RateLimitMiddleware,
    rate_limiter=InMemoryRateLimiter(requests=10, seconds=1),
)

app = socketio.ASGIApp(sio, other_asgi_app=base_app)
