#!/bin/bash

for filename in `ls *.ipynb */*.ipynb 2> /dev/null`; do
  echo $filename
  sed -i 's+cloud-ml-data/img+practical-ml-vision-book/flowers_5_jpeg+g' ${filename}
done
