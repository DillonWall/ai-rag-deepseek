#!/usr/bin/bash
docker run -d -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
