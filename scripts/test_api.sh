#!/usr/bin/env bash

export OPENAI_API_KEY="YOUR_API_KEY"

curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"