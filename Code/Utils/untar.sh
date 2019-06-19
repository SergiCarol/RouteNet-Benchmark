#!/bin/sh
#usage:
# nameOfScript myArchive.tar.gz
# nameOfScript myArchive.gz
# nameOfScript myArchive.tar
#
# Result:
# myArchive   //folder
fileName="${1%.*}" #extracted filename

#handle the case of archive.tar.gz
trailingExtension="${fileName##*.}"

if [ "$trailingExtension" = "tar" ]  
then
    fileName="${fileName%.*}"  #remove trailing  tar.
fi

mkdir "$fileName"
tar -xf "$1" --strip-components=0 -C "$fileName"
