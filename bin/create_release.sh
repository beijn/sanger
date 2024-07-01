#!/bin/bash

if [ -f release.zip ]; then rm release.zip; fi

path="$1"
version="$2"

# if either path or version are empty
if [ -z "$path" ] || [ -z "$version" ]; then
    echo "Usage: $0 <path to release's results folder> <version tag>"
    exit 1
fi

if [ -d model_export ]; then rm -rf model_export; fi
cp -r "$path/model_export" model_export

zip -r model.zip "model_export"
rm -rf "model_export"
#zip -u release.zip release.yml
#zip -u release.zip release.py

#printf "@ release.yml\n@=environment.yml\n" | zipnote -w release.zip
#printf "@ release.py\n@=main.py\n" | zipnote -w release.zip

gh release create --target main "$version" "model.zip" -t "$version" -n ""  --latest
