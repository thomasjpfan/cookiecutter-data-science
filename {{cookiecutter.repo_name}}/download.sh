#!/usr/bin/env bash
set -eo pipefail

NAME=""

dl_kaggle() {
	local filename="$1"
	if [ -f "$filename" ]; then
		echo "$filename already downloaded"
		return
	fi
	echo "Downloading ${filename}"
	kaggle competitions download -c $NAME -f "$filename" -w
}

dl_kaggle_all() {
	all_files=$(kaggle competitions files -c home-credit-default-risk -v | tail -n +2 | cut -d ',' -f 1)
	for filename in "$all_files"; do
		dl_kaggle "$filename"
	done
}

extract() {
	if [ -z "$1" ]; then
		echo "Usage: extract <path/file_name>.<zip|rar|bz2|gz|tar|tbz2|tgz|Z|7z|xz|ex|tar.bz2|tar.gz|tar.xz>"
		echo "       extract <path/file_name_1.ext> [path/file_name_2.ext] [path/file_name_3.ext]"
	else
		for n in "$@"; do
			if [ -f "$n" ]; then
				case "${n%,}" in
				*.tar.bz2 | *.tar.gz | *.tar.xz | *.tbz2 | *.tgz | *.txz | *.tar)
					tar xvf "$n"
					;;
				*.lzma) unlzma ./"$n" ;;
				*.bz2) bunzip2 ./"$n" ;;
				*.rar) unrar x -ad ./"$n" ;;
				*.gz) gunzip ./"$n" ;;
				*.zip) unzip ./"$n" ;;
				*.z) uncompress ./"$n" ;;
				*.7z | *.arj | *.cab | *.chm | *.deb | *.dmg | *.iso | *.lzh | *.msi | *.rpm | *.udf | *.wim | *.xar)
					7z x ./"$n"
					;;
				*.xz) unxz ./"$n" ;;
				*.exe) cabextract ./"$n" ;;
				*.cpio) cpio -id <./"$n" ;;
				*)
					echo "extract: '$n' - unknown archive method"
					return 1
					;;
				esac
			else
				echo "'$n' - file does not exist"
				return 1
			fi
		done
	fi
}

extract_single() {
	local input="$1"
	local output="$2"

	if [ -d "$output" ] || [ -f "$output" ]; then
		echo "${output} already exists"
		return
	fi
	echo "Extracing ${input} to ${output}"
	extract "$input"
}

cd data/raw
