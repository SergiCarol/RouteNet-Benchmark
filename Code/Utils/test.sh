for file in *; do
	echo $file
	if [ "$file" != "test.sh" ]
	then
    	mv "$file/$file.txt" "$file.txt"
    	rm -rf "$file"
    fi
done
