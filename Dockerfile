# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build using openenv-base
# This Dockerfile is flexible and works for both:
# - In-repo environments (with local OpenEnv sources)
# - Standalone environments (with openenv from PyPI/Git)
# The build script (openenv build) handles context detection and sets appropriate build args.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Licensed under the BSD-style license found in the LICENSE file in the root directory.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Licensed under the BSD-style license found in the LICENSE file in the root directory.


# ============================================================
# ----------- STAGE 1: BUILDER STAGE --------------------------
# This stage installs dependencies and prepares the environment
# ============================================================

FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir fastapi uvicorn pydantic openai requests

ENV PYTHONPATH=/app

CMD ["uvicorn", "round_1.server.app:app", "--host", "0.0.0.0", "--port", "7860"]