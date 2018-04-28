#!/usr/bin/env bash
set -eo pipefail

NAME=""

dl_kaggle() {
	local filename="$1"
	if [ -f "$filename" ]; then
		echo "$filename already downloaded"
		return
	fi

	kaggle competitions download -c $NAME -f "$filename" -w
}

dl() {
	local filename="$1"
	local url="$2"

	echo "Downloading $filename from $url"
	# Download script
}

cd data/raw
