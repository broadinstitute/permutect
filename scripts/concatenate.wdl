version 1.0

workflow Concatenate {
	input {
		Array[File] inputs
        Array[String] headings
	}

	call concatenate { input: inputs = inputs, headings = headings }

	output {
		File concatenated = concatenate.concatenated
	}
}

task concatenate {
	input {
		Array[File] inputs
        Array[String] headings
		Int? disk_space
	}

	command {
        # FOFN, one file per line
        touch file_names.txt
        for file in ~{sep=' ' inputs}; do
            echo $file >> file_names.txt
		done

        touch headings.txt
        for heading in ~{sep=' ' headings}; do
            echo $heading >> headings.txt
		done

        paste file_names.txt headings.txt > tmp.txt

        touch result.txt
        while read file heading; do
            echo $heading >> result.txt
			cat $file >> result.txt
			echo "" >> result.txt
        done < tmp.txt
	}

    runtime {
        docker: "continuumio/anaconda:latest"
        disks: "local-disk " + select_first([disk_space, 100]) + " HDD"
    }

    output {
        File concatenated = "result.txt"
    }
}