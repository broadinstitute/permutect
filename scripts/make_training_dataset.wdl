version 1.0

# run Mutect2 without filtering to get plain text training data, then run preprocess_dataset.

import "https://api.firecloud.org/ga4gh/v1/tools/davidben:mutect2-testing/versions/61/plain-WDL/descriptor" as m2

workflow MakeTrainingDataset {
    input {
        # basic inputs
        # Files of the genomic intervals to call, such as .interval_list or .bed files, or [] to call the whole genome.
        # Mutect2's WDL gives GATK each file with its own -L, and GATK takes the union of their intervals.  In a
        # Terra configuration, an attribute expression such as workspace.intervals gives a one-element array.  Elements
        # that are "" are ignored, since Terra may turn unset data table values into "" (see below).  (Earlier versions
        # took a single, optional file.)
        Array[File] intervals
        File? masked_intervals
        # The reference FASTA.  Its companion files are not inputs: Mutect2 (GATK) finds them by name in the FASTA's
        # directory and accepts no other names, so they must exist there under these names:
        # - its index, the FASTA's path plus .fai, eg Homo_sapiens_assembly38.fasta.fai
        # - its sequence dictionary, the FASTA's path with its extension (.fasta, .fa, .fna, .fas or .txt, optionally
        #   followed by .gz) replaced by .dict, eg Homo_sapiens_assembly38.dict
        # - for a bgzipped FASTA (.gz), its block index, the FASTA's path plus .gzi
        # Mutect2's WDL passes these files to its tasks along with the FASTA, so that backends that localize files, such as
        # a local Cromwell, put them beside it, while on Terra they are streamed from the bucket.  A missing companion
        # file fails the workflow early, with an error naming the file.
        File ref_fasta
        # The sample's reads, which Mutect2's WDL takes as tumor_reads: an array whose first element is a BAM or CRAM file
        # and whose optional second element is its index.  GATK finds the index only by name in the reads' directory, so
        # it must be there under one of these names, whether or not the array includes it:
        # - for x.bam: x.bai, x.bam.bai, x.csi or x.bam.csi
        # - for x.cram: x.crai or x.cram.crai
        # On Terra, Mutect2 streams the reads from the bucket and finds the index there, so the BAM or CRAM alone
        # suffices: in a configuration, an attribute expression such as this.bam gives a one-element array, and
        # [this.bam, this.bai] a two-element one.  On a backend that localizes files, such as a local Cromwell, include
        # the index, so that it is localized beside the reads.  Including it always works.  Elements that are "" are
        # ignored, since Terra may turn unset data table values into "" (see below).  (Earlier versions took the index as
        # the separate input reads_index.)
        Array[File]+ reads

        # NOTE: this is a HACK because call-caching on Terra is unreliable.
        String? cached_plain_text_dataset

        # VCFs, each an array of the VCF and, optionally, its index: [vcf] or [vcf, idx], or [] for none.  GATK finds a
        # VCF's index only by name beside the VCF: the VCF's path plus .tbi for a bgzipped VCF (.vcf.gz or .vcf.bgz), or
        # plus .idx for a plain .vcf.  Given [vcf], the workflow derives that path and passes the index along with the
        # VCF, so that backends that localize files put it beside the VCF, while on Terra the tasks stream both.  So
        # [vcf] suffices on every backend, and an index given as the second element must be that same file.  In a Terra
        # configuration, an attribute expression such as workspace.gnomad gives a one-element array.  Elements that are
        # "" are ignored, since Terra may turn unset data table values into "" (see below).  (Earlier versions took each
        # index as a separate input, such as germline_resource_idx.)
        # The germline resource and the blacklist are optional but strongly recommended, so each must be given, if only
        # as [] to go without it.
        # a germline resource with population allele frequencies, eg gnomAD's, with which Mutect2 recognizes germline
        # variants
        Array[File] germline_resource
        # a blacklist of sites, which Mutect2 takes as its panel of normals
        Array[File] blacklist_vcf
        # a truth VCF, which labels the training dataset.  It's genuinely optional and may be left unset.
        Array[File]? permutect_training_dataset_truth_vcf

        # extra arguments
        String? m2_extra_args

        # additional modes and outputs
        File? dragstr_model
        Boolean make_bamout = false
        Boolean compress_vcfs = false


        # runtime
        String permutect_docker
        File? gatk_override
        Int scatter_count
        Int preemptible = 2
        Int max_retries = 1
        Int small_task_mem = 4
        Int small_task_disk = 100
        String? gcs_project_for_requester_pays

        # Use as a last resort to increase the disk given to every task in case of ill behaving data
        Int emergency_extra_disk = 0
    }

    # WDL version 1.0 does not have an empty Optional literal.  Such a literal is very useful because Terra has a bug
    # where whenever a data table is updated, empty values silently and invisibly get converted to empty strings "".
    # Thus it is useful to recognize empty strings and declare empty Optionals.  A declaration inside a conditional is
    # an Optional outside of it, so NONE is always an empty Optional.
    if (false) {
        File NONE = ""
    }

    # The input arrays without their "" elements, which Terra may have made from unset data table values, eg [this.bam,
    # this.bai] becoming [bam, ""].  WDL has no filter function, so a scatter without calls keeps the other elements,
    # and select_all collects them.  Keep given_arrays and the indices into nonblank below in the same order.
    Array[Array[File]] given_arrays = [reads, intervals, germline_resource, blacklist_vcf,
        select_first([permutect_training_dataset_truth_vcf, []])]
    scatter (given_array in given_arrays) {
        scatter (element in given_array) {
            if (element != "") {
                File nonblank_element = element
            }
        }
        Array[File] nonblank = select_all(nonblank_element)
    }
    Array[File] given_reads = nonblank[0]
    Array[File] given_intervals = nonblank[1]
    Array[File] given_germline_resource = nonblank[2]
    Array[File] given_blacklist_vcf = nonblank[3]
    Array[File] given_permutect_training_dataset_truth_vcf = nonblank[4]

    File? masks_hack = if (select_first([masked_intervals,""]) == "") then NONE else masked_intervals
    File? dragstr_model_hack = if select_first([dragstr_model, ""]) == "" then NONE else dragstr_model
    File? gatk_override_hack = if select_first([gatk_override, ""]) == "" then NONE else gatk_override
    String? gcs_project_for_requester_pays_hack = if select_first([gcs_project_for_requester_pays, ""]) == "" then NONE else gcs_project_for_requester_pays
    String? cached_plain_text_dataset_hack = if select_first([cached_plain_text_dataset, ""]) == "" then NONE else cached_plain_text_dataset

    if (!defined(cached_plain_text_dataset_hack)) {
        call m2.Mutect2 {
            input:
                intervals = given_intervals,
                masked_intervals = masks_hack,
                ref_fasta = ref_fasta,
                tumor_reads = given_reads,
                normal_reads = [],
                germline_resource = given_germline_resource,
                pon = given_blacklist_vcf,
                # the training dataset doesn't use pileup summaries
                common_variants_vcf = [],
                m2_extra_args = m2_extra_args,
                dragstr_model = dragstr_model_hack,
                make_bamout = make_bamout,
                compress_vcfs = compress_vcfs,
                make_permutect_training_dataset = true,
                permutect_training_dataset_truth_vcf = given_permutect_training_dataset_truth_vcf,
                skip_filtering = true,
                gatk_docker = permutect_docker,
                gatk_override = gatk_override_hack,
                scatter_count = scatter_count,
                preemptible = preemptible,
                max_retries =  max_retries,
                small_task_mem = small_task_mem,
                small_task_disk = small_task_disk,
                gcs_project_for_requester_pays = gcs_project_for_requester_pays_hack,
                emergency_extra_disk = emergency_extra_disk
        }
    }

    call Preprocess {
        input:
            training_dataset = select_first([cached_plain_text_dataset_hack, Mutect2.permutect_training_dataset]),
            permutect_docker = permutect_docker
    }

    output {
        File? bamout = Mutect2.bamout
        File? bamout_index = Mutect2.bamout_index
        File? mutect_stats = Mutect2.mutect_stats
        File? permutect_contigs_table = Mutect2.permutect_contigs_table
        File? permutect_read_groups_table = Mutect2.permutect_read_groups_table
        String plain_text_dataset = select_first([cached_plain_text_dataset_hack, Mutect2.permutect_training_dataset])
        File train_tar = Preprocess.train_tar
    }

}

task Preprocess {
    input {
        File training_dataset
        Int? source_label

        String permutect_docker
        Int? preemptible
        Int? max_retries
        Int? disk_space
        Int? cpu
        Int? mem
        Boolean use_ssd = true
    }

    Int num_cpus = select_first([cpu, 4])
    # Mem is in units of GB but our memory runtime value is in MB.  Memory doesn't grow with the dataset: each of the
    # processes normalizes 10,000 data at a time in well under 1 GB, and the main process needs about 0.5 GB.
    Int machine_mem = if defined(mem) then mem * 1000 else (2 + num_cpus) * 1000
    # the plain text dataset, and the preprocessed data twice, in temporary memory maps and in the tarfile, each less
    # than half the size of the plain text
    Int default_disk_space = ceil(2 * size(training_dataset, "GB")) + 10

    command <<<
        set -e

        # one process per CPU reads and normalizes chunks of the data
        preprocess_dataset --training_datasets ~{training_dataset} ~{"--sources " + source_label} \
            --num_processes ~{num_cpus} --output train.tar
    >>>

    runtime {
        docker: permutect_docker
        bootDiskSizeGb: 4  # Cromwell on GCP Batch adds 30 GB for the OS and the docker image (7.2 GB unpacked)
        memory: machine_mem + " MB"
        disks: "local-disk " + select_first([disk_space, default_disk_space]) + if use_ssd then " SSD" else " HDD"
        preemptible: select_first([preemptible, 2])
        maxRetries: select_first([max_retries, 0])
        cpu: num_cpus
    }

    output {
        File train_tar = "train.tar"
    }
}