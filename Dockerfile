# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Builds a Docker image capable of running the code in the book
FROM gcr.io/deeplearning-platform-release/tf2-gpu

RUN python3 -m pip install --upgrade apache-beam[gcp] cloudml-hypertune

RUN mkdir -p /src/practical-ml-vision-book

# copy all the chapters that start with 01_* 02_* etc.
COPY . /src/practical-ml-vision-book/

RUN ls -l /src/practical-ml-vision-book/

WORKDIR /src/practical-ml-vision-book
