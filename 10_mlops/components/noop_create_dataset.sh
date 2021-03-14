#!/bin/bash

OUTPUT_DIR=$1
shift
COMPONENT_OUT=$1
shift

# for subsequent components
mkdir -p $(dirname $COMPONENT_OUT)
echo "$OUTPUT_DIR" > $COMPONENT_OUT