#!/bin/sh

image=$1

mkdir -p test_dir/model
mkdir -p test_dir/output

rm test_dir/model/*
rm test_dir/output/*

docker run -v $(pwd)/test_dir/model:/opt/ml/model -v $(pwd)/../../data:/opt/ml/input/data -v $(pwd)/test_dir/input/config:/opt/ml/input/config -v $(pwd)/test_dir/output:/opt/ml/output --rm ${image} train
