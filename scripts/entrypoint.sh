#!/bin/sh
exec uvicorn app.main:app --port 8000
