"""Compatibility shim for validators and local entrypoints.

The FastAPI app now lives in server/app.py and still uses CORSMiddleware.
The default uvicorn port remains 7860.
"""

from server.app import app, main


if __name__ == "__main__":
    main()
