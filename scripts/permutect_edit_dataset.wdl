version 1.0


workflow EditDataset {
    input {
        File train_tar
        Int? new_source
        String edit_type
        String? extra_args

        String permutect_docker
    }

    call EditDataset {
        input:
            train_tar = train_tar,
            permutect_docker = permutect_docker,
            new_source = new_source,
            edit_type = edit_type,
            extra_args = extra_args
    }

    output {
        File edited_dataset_tarfile = EditDataset.output_dataset_tarfile
    }
}

task EditDataset {
    input {
        File train_tar
        Int? new_source
        String edit_type
        String? extra_args

        String permutect_docker
        Int? preemptible
        Int? max_retries
        Int? disk_space
        Int? cpu
        Int? mem
        Boolean use_ssd = true
    }

    # the input datasets and the edited dataset, which is no bigger than them
    Int default_disk_space = ceil(2 * size(train_tar, "GB")) + 10

    # Mem is in units of GB but our memory runtime value is in MB.  edit_dataset memory-maps the datasets and edits them
    # 100,000 data at a time, so its memory doesn't grow with them: in November 2026 merging four 1.4 GB datasets used at
    # most 1.2 GB of the machine's memory.
    Int machine_mem = if defined(mem) then mem * 1000 else 4000

    command <<<
        set -e

        edit_dataset \
            --train_tar ~{train_tar} \
            ~{" --source " + new_source} \
            --dataset_edit ~{edit_type} \
            --output edited_dataset.tar \
            ~{extra_args}
    >>>

    runtime {
        docker: permutect_docker
        bootDiskSizeGb: 4  # Cromwell on GCP Batch adds 30 GB for the OS and the docker image (7.2 GB unpacked)
        memory: machine_mem + " MB"
        disks: "local-disk " + select_first([disk_space, default_disk_space]) + if use_ssd then " SSD" else " HDD"
        preemptible: select_first([preemptible, 10])
        maxRetries: select_first([max_retries, 0])
        cpu: select_first([cpu, 1])
    }

    output {
        File output_dataset_tarfile = "edited_dataset.tar"
    }
}
