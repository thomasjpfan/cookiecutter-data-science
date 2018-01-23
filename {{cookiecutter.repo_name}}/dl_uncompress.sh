#!/usr/bin/env bash
set -eo pipefail

dl() {
	local filename="data/raw/$1"

	if [ -f "$filename" ]; then
		echo "$filename already downloaded"
		return
	fi

	echo "Downloading $filename"
	# Download script
}

extract() {
	local input="data/raw/$1"
	local output="data/raw/$2"

	if [ -f "$output" ]; then
		echo "$output already extracted"
		return
	fi

	echo "Extracting $input to $output"
	# Extract script
}

cd data/raw
