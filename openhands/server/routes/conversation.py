import asyncio
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

from openhands.core.logger import openhands_logger as logger
from openhands.events.async_event_store_wrapper import AsyncEventStoreWrapper
from openhands.events.event_filter import EventFilter
from openhands.events.event_store import EventStore
from openhands.events.serialization.event import event_to_dict
from openhands.events.stream import EventStreamSubscriber
from openhands.memory.memory import Memory
from openhands.microagent.types import InputMetadata
from openhands.runtime.base import Runtime
from openhands.server.dependencies import get_dependencies
from openhands.server.session.conversation import ServerConversation
from openhands.server.shared import conversation_manager, file_store
from openhands.server.user_auth import get_user_id
from openhands.server.utils import get_conversation, get_conversation_metadata
from openhands.storage.data_models.conversation_metadata import ConversationMetadata

app = APIRouter(
    prefix='/api/conversations/{conversation_id}', dependencies=get_dependencies()
)


@app.get('/config')
async def get_remote_runtime_config(
    conversation: ServerConversation = Depends(get_conversation),
) -> JSONResponse:
    """Retrieve the runtime configuration.

    Currently, this is the session ID and runtime ID (if available).
    """
    runtime = conversation.runtime
    runtime_id = runtime.runtime_id if hasattr(runtime, 'runtime_id') else None
    session_id = runtime.sid if hasattr(runtime, 'sid') else None
    return JSONResponse(
        content={
            'runtime_id': runtime_id,
            'session_id': session_id,
        }
    )


@app.get('/vscode-url')
async def get_vscode_url(
    conversation: ServerConversation = Depends(get_conversation),
) -> JSONResponse:
    """Get the VSCode URL.

    This endpoint allows getting the VSCode URL.

    Args:
        request (Request): The incoming FastAPI request object.

    Returns:
        JSONResponse: A JSON response indicating the success of the operation.
    """
    try:
        runtime: Runtime = conversation.runtime
        logger.debug(f'Runtime type: {type(runtime)}')
        logger.debug(f'Runtime VSCode URL: {runtime.vscode_url}')
        return JSONResponse(
            status_code=status.HTTP_200_OK, content={'vscode_url': runtime.vscode_url}
        )
    except Exception as e:
        logger.error(f'Error getting VSCode URL: {e}')
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                'vscode_url': None,
                'error': f'Error getting VSCode URL: {e}',
            },
        )


@app.get('/web-hosts')
async def get_hosts(
    conversation: ServerConversation = Depends(get_conversation),
) -> JSONResponse:
    """Get the hosts used by the runtime.

    This endpoint allows getting the hosts used by the runtime.

    Args:
        request (Request): The incoming FastAPI request object.

    Returns:
        JSONResponse: A JSON response indicating the success of the operation.
    """
    try:
        runtime: Runtime = conversation.runtime
        logger.debug(f'Runtime type: {type(runtime)}')
        logger.debug(f'Runtime hosts: {runtime.web_hosts}')
        return JSONResponse(status_code=200, content={'hosts': runtime.web_hosts})
    except Exception as e:
        logger.error(f'Error getting runtime hosts: {e}')
        return JSONResponse(
            status_code=500,
            content={
                'hosts': None,
                'error': f'Error getting runtime hosts: {e}',
            },
        )


@app.get('/events')
async def search_events(
    conversation_id: str,
    start_id: int = 0,
    end_id: int | None = None,
    reverse: bool = False,
    filter: EventFilter | None = None,
    limit: int = 20,
    metadata: ConversationMetadata = Depends(get_conversation_metadata),
    user_id: str | None = Depends(get_user_id),
):
    """Search through the event stream with filtering and pagination.

    Args:
        conversation_id: The conversation ID
        start_id: Starting ID in the event stream. Defaults to 0
        end_id: Ending ID in the event stream
        reverse: Whether to retrieve events in reverse order. Defaults to False.
        filter: Filter for events
        limit: Maximum number of events to return. Must be between 1 and 100. Defaults to 20
        metadata: Conversation metadata (injected by dependency)
        user_id: User ID (injected by dependency)

    Returns:
        dict: Dictionary containing:
            - events: List of matching events
            - has_more: Whether there are more matching events after this batch
    Raises:
        HTTPException: If conversation is not found or access is denied
        ValueError: If limit is less than 1 or greater than 100
    """
    if limit < 0 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid limit'
        )

    # Create an event store to access the events directly
    event_store = EventStore(
        sid=conversation_id,
        file_store=file_store,
        user_id=user_id,
    )

    # Get matching events from the store
    events = list(
        event_store.search_events(
            start_id=start_id,
            end_id=end_id,
            reverse=reverse,
            filter=filter,
            limit=limit + 1,
        )
    )

    # Check if there are more events
    has_more = len(events) > limit
    if has_more:
        events = events[:limit]  # Remove the extra event

    events_json = [event_to_dict(event) for event in events]
    return {
        'events': events_json,
        'has_more': has_more,
    }


@app.post('/events')
async def add_event(
    request: Request, conversation: ServerConversation = Depends(get_conversation)
):
    data = await request.json()
    await conversation_manager.send_event_to_conversation(conversation.sid, data)
    return JSONResponse({'success': True})


@app.get('/events/stream')
async def stream_events(
    request: Request,
    start_id: int = 0,
    exclude_hidden: bool = True,
    conversation: ServerConversation = Depends(get_conversation),
):
    """Stream conversation events via Server-Sent Events (SSE).

    Replays events from start_id, then streams new events as they arrive.
    Each item is sent as event: oh_event with JSON data.
    """

    async def event_generator() -> AsyncIterator[dict]:
        # Initial replay
        try:
            async_store = AsyncEventStoreWrapper(
                conversation.event_stream,
                start_id=start_id,
                filter=EventFilter(exclude_hidden=exclude_hidden),
            )
        except Exception as e:
            logger.error(f'Error initializing async event store: {e}')
            yield {"event": "error", "data": {"message": "Failed to initialize stream"}}
            return

        last_id = start_id - 1

        async for ev in async_store:
            data = event_to_dict(ev)
            if isinstance(data, dict) and 'id' in data:
                try:
                    last_id = max(last_id, int(data['id']))
                except Exception:
                    pass
            yield {"event": "oh_event", "data": data}

        # Live updates
        queue: asyncio.Queue[dict] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        callback_id = f'sse:{uuid.uuid4().hex}'

        def _callback(ev):
            try:
                # Filter hidden events using the event attribute
                if exclude_hidden and getattr(ev, 'hidden', False):
                    return
                data = event_to_dict(ev)
                ev_id = data.get('id')
                try:
                    if ev_id is not None and int(ev_id) <= last_id:
                        return
                except Exception:
                    pass

                def _put():
                    try:
                        queue.put_nowait({"event": "oh_event", "data": data})
                    except Exception:
                        pass

                loop.call_soon_threadsafe(_put)
            except Exception as e:
                logger.error(f'Error in SSE callback: {e}')

        # Subscribe to live events
        conversation.event_stream.subscribe(
            EventStreamSubscriber.SERVER, _callback, callback_id
        )

        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=1.0)
                    # Update last_id to avoid duplicates
                    data = item.get('data', {})
                    if isinstance(data, dict) and 'id' in data:
                        try:
                            ev_id_int = int(data['id'])
                            if ev_id_int > last_id:
                                last_id = ev_id_int
                        except Exception:
                            pass
                    yield item
                except asyncio.TimeoutError:
                    # Heartbeat
                    yield {"event": "heartbeat", "data": "ping"}
        finally:
            try:
                conversation.event_stream.unsubscribe(
                    EventStreamSubscriber.SERVER, callback_id
                )
            except Exception:
                pass

    return EventSourceResponse(event_generator())


class MicroagentResponse(BaseModel):
    """Response model for microagents endpoint."""

    name: str
    type: str
    content: str
    triggers: list[str] = []
    inputs: list[InputMetadata] = []
    tools: list[str] = []


@app.get('/microagents')
async def get_microagents(
    conversation: ServerConversation = Depends(get_conversation),
) -> JSONResponse:
    """Get all microagents associated with the conversation.

    This endpoint returns all repository and knowledge microagents that are loaded for the conversation.

    Returns:
        JSONResponse: A JSON response containing the list of microagents.
    """
    try:
        # Get the agent session for this conversation
        agent_session = conversation_manager.get_agent_session(conversation.sid)

        if not agent_session:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={'error': 'Agent session not found for this conversation'},
            )

        # Access the memory to get the microagents
        memory: Memory | None = agent_session.memory
        if memory is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    'error': 'Memory is not yet initialized for this conversation'
                },
            )

        # Prepare the response
        microagents = []

        # Add repo microagents
        for name, r_agent in memory.repo_microagents.items():
            microagents.append(
                MicroagentResponse(
                    name=name,
                    type='repo',
                    content=r_agent.content,
                    triggers=[],
                    inputs=r_agent.metadata.inputs,
                    tools=[
                        server.name
                        for server in r_agent.metadata.mcp_tools.stdio_servers
                    ]
                    if r_agent.metadata.mcp_tools
                    else [],
                )
            )

        # Add knowledge microagents
        for name, k_agent in memory.knowledge_microagents.items():
            microagents.append(
                MicroagentResponse(
                    name=name,
                    type='knowledge',
                    content=k_agent.content,
                    triggers=k_agent.triggers,
                    inputs=k_agent.metadata.inputs,
                    tools=[
                        server.name
                        for server in k_agent.metadata.mcp_tools.stdio_servers
                    ]
                    if k_agent.metadata.mcp_tools
                    else [],
                )
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={'microagents': [m.dict() for m in microagents]},
        )
    except Exception as e:
        logger.error(f'Error getting microagents: {e}')
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'error': f'Error getting microagents: {e}'},
        )
