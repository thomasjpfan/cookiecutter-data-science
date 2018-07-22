#!/usr/bin/env bash
set -eo pipefail

NAME=""

dl_kaggle() {
	local filename="$1"
	kaggle competitions download -c $NAME -f "$filename" -w
}

dl() {
	local filename="$1"
	local url="$2"

	echo "Downloading $filename from $url"
	# Download script
}

extract_single() {
	local input="$1"
	local output="$2"
	local fn="${output##*/}"

	if [ -f "$fn" ]; then
		echo "$fn already extracted"
		return
	fi

	echo "Extracting $input to $fn"
	# Extract script
	unzip "$input"

	folder=$(dirname "$output")
	if [ "$folder" == "." ]; then
		return
	fi

	mv "${output}" "$output"
	base=$(echo "$2" | cut -d "/" -f1)
	rm -rf "${base}"

}

extract_into_folder() {
	local input="$1"
	local output="$2"

	if [ -d "$output" ]; then
		echo "$output already extracted"
		return
	fi

	echo "Extracting $input to $output"
	# Extract script
	unzip "$input" -d "$output"
}

cd data/raw
