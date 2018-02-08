#!/usr/bin/env bash
set -eo pipefail

dl() {
	local filename="$1"

	if [ -f "$filename" ]; then
		echo "$filename already downloaded"
		return
	fi

	echo "Downloading $filename"
	# Download script
}

extract_single() {
	local input="$1"
	local output="$2"

	if [ -f "$output" ]; then
		echo "$output already extracted"
		return
	fi

	echo "Extracting $input to $output"
	# Extract script
	unzip "$input"
}

extract_folder() {
	local input="$1"
	local output="$2"

	if [ -f "$output" ]; then
		echo "$output already extracted"
		return
	fi

	echo "Extracting $input to $output"
	# Extract script
	unzip "$input" -d "$output"
}

cd data/raw
