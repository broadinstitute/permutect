version 1.0


workflow MergeDatasets {
    input {
        Array[File] train_tar
        String? extra_args

        String permutect_docker
    }

    call EditDataset {
        input:
            train_tar = train_tar,
            permutect_docker = permutect_docker,
            edit_type = "keep_everything",
            extra_args = extra_args
    }

    output {
        File merged_dataset_tarfile = EditDataset.output_dataset_tarfile
    }
}

task EditDataset {
    input {
        Array[File] train_tar
        String edit_type
        String? extra_args

        String permutect_docker
        Int? preemptible
        Int? max_retries
        Int? disk_space
        Int? cpu
        Int? mem
    }

    # the input datasets and the edited dataset, which is no bigger than them, on an SSD, since the task reads and
    # writes them all
    Int default_disk_space = ceil(2 * size(train_tar, "GB")) + 10

    # Mem is in units of GB but our memory runtime value is in MB.  edit_dataset memory-maps the datasets and edits them
    # 100,000 data at a time, so its memory doesn't grow with them: in November 2026 merging four 1.4 GB datasets used at
    # most 1.2 GB of the machine's memory.
    Int machine_mem = if defined(mem) then mem * 1000 else 4000

    command <<<
        set -e

        edit_dataset \
            --train_tar ~{sep=' ' train_tar} \
            --dataset_edit ~{edit_type} \
            --output edited_dataset.tar \
            ~{extra_args}
    >>>

    runtime {
        docker: permutect_docker
        bootDiskSizeGb: 4  # Cromwell on GCP Batch adds 30 GB for the OS and the docker image (7.2 GB unpacked)
        memory: machine_mem + " MB"
        disks: "local-disk " + select_first([disk_space, default_disk_space]) + " SSD"
        preemptible: select_first([preemptible, 10])
        maxRetries: select_first([max_retries, 0])
        cpu: select_first([cpu, 1])
    }

    output {
        File output_dataset_tarfile = "edited_dataset.tar"
    }
}
