version 1.0

workflow Concatenate {
	input {
		Array[File] inputs
	}

	call concatenate { input: inputs = inputs}

	output {
		File concatenated = concatenate.concatenated
	}
}

task concatenate {
	input {
		Array[File] inputs
		Int? disk_space
	}

	command <<<
        touch TMP   $ first column is the heading word, second column is the file name

        for file in ~{sep=' ' inputs}; do
            echo "file is" $file

            echo "first line is"
            head -n 1 $file

            echo "first line lowercase is"
            head -n 1 $file | tr '[:upper:]' '[:lower:]'


            # convert first line to lowercase and extract first word
            head -n 1 $file | tr '[:upper:]' '[:lower:]' | while read first_word _; do
                echo "first word is"
                echo $first_word
                echo $first_word $file >> TMP
            done
        done

        #debug
        echo "here's the sorting file:"
        cat TMP

        echo "here are the files in order"
        sort k1,1 TMP | while read heading filename; do echo $filename; done

        # sort by heading word and cat the files in order
        touch result.txt
        sort k1,1 TMP | while read heading filename; do cat $filename >> result.txt; echo "" >> result.txt; done
	>>>

    runtime {
        docker: "continuumio/anaconda:latest"
        disks: "local-disk " + select_first([disk_space, 100]) + " HDD"
    }

    output {
        File concatenated = "result.txt"
    }
}