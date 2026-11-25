version 1.0

## NOTE: this is essentially copied from https://github.com/broadinstitute/warp/blob/develop/tasks/broad/DragenTasks.wdl
## with minor modifications

workflow DragstrCalibration {
    input {
        # The reference FASTA.  Its companion files are not inputs: GATK finds them by name in the FASTA's directory and
        # accepts no other names, so they must exist there under these names:
        # - its index, the FASTA's path plus .fai, eg Homo_sapiens_assembly38.fasta.fai
        # - its sequence dictionary, the FASTA's path with its extension (.fasta, .fa, .fna, .fas or .txt, optionally
        #   followed by .gz) replaced by .dict, eg Homo_sapiens_assembly38.dict
        # - for a bgzipped FASTA (.gz), its block index, the FASTA's path plus .gzi
        # The workflow passes these files to its task along with the FASTA, so that backends that localize files, such as
        # a local Cromwell, put them beside it, while on Terra the task streams them from the bucket.  A missing companion
        # file fails the workflow early, with an error naming the file.
        File ref_fasta
        File reads
        File reads_index
    }

    # The reference FASTA and its companion files, derived as Mutect2's WDL derives them: the index, the FASTA's path plus
    # .fai; the dictionary, the FASTA's path with its extension replaced by .dict; and for a bgzipped FASTA the block
    # index, its path plus .gzi.  The task uses reference[0] as the FASTA.
    String ref_fasta_path = "~{ref_fasta}"
    Array[String] ref_gzi = if sub(ref_fasta_path, "\\.gz$", "") != ref_fasta_path then [ref_fasta_path + ".gzi"] else []
    Array[File] reference = flatten([[ref_fasta_path, ref_fasta_path + ".fai",
        sub(ref_fasta_path, "\\.(fasta|fas|fa|fna|txt)(\\.gz)?$", "") + ".dict"], ref_gzi])

    call CalibrateDragstrModel {
        input:
            reference = reference,
            reads = reads,
            reads_index = reads_index
    }

    output {
        File dragstr_model = CalibrateDragstrModel.dragstr_model
    }
}

task CalibrateDragstrModel {
  input {
    # the reference FASTA, then its .fai index, its .dict dictionary and, if it's bgzipped, its .gzi block index
    Array[File] reference
    File reads
    File reads_index

    String docker = "us.gcr.io/broad-gatk/gatk:4.5.0.0"
    Int preemptible_tries = 1
    Int threads = 4
    Int? disk_space
    Int? mem
  }

  String parallel_args  = "--threads " + threads

  # Mem is in units of GB but our command and memory runtime values are in MB
  Int machine_mem = if defined(mem) then mem * 1000 else 4000
  Int command_mem = machine_mem - 500

  parameter_meta{
      reference: {localization_optional: true}
      reads: {localization_optional: true}
      reads_index: {localization_optional: true}
    }

  command <<<
    set -x

    gatk ComposeSTRTableFile \
      -R ~{reference[0]} \
      -O str_table.tsv


    gatk --java-options "-Xmx~{command_mem}m" CalibrateDragstrModel \
        -R ~{reference[0]} \
        -I ~{reads} \
        -str str_table.tsv \
        -O params.dragstr \
        ~{parallel_args}
  >>>

  runtime {
     docker: docker
     disks: "local-disk " + select_first([disk_space, 100]) + " SSD"
     memory: machine_mem + " MB"
     preemptible: preemptible_tries
     cpu: threads
  }

  output {
    File dragstr_model = "params.dragstr"
  }
}
