version 1.0

import "https://api.firecloud.org/ga4gh/v1/tools/davidben:mutect2-testing/versions/53/plain-WDL/descriptor" as m2

workflow Permutect {
    input {
        File artifact_model

        File? intervals
        File? masks
        File ref_fasta
        File ref_fai
        File ref_dict
        Int scatter_count
        Int? num_spectrum_iterations
        Float? spectrum_learning_rate
        File primary_bam
        File primary_bai
        File? control_bam
        File? control_bai
        File? gnomad
        File? gnomad_idx
        File? variants_for_contamination
        File? variants_for_contamination_idx
        File? realignment_index_bundle
        File? dragstr_model
        String? realignment_extra_args
        Boolean skip_m2_filtering = true
        Boolean run_orientation_bias_mixture_model_filter = false
        Boolean filter_pcr_errors = false   # run pre_permutect_filtering, which applies the hairpin filter
        String? m2_extra_args
        String? split_intervals_extra_args
        File? common_hets_for_segmentation  # Mutect2 runs ModelSegments if and only if this is given
        String? collect_allelic_counts_extra_args
        Int? model_segments_min_total_allele_count
        String? model_segments_extra_args
        Int batch_size
        Int num_workers
        Int? gpu_count
        File? test_dataset_truth_vcf    # used for evaluation
        File? test_dataset_truth_vcf_idx
        String? concordance_header

        # Mutect2's outputs from an earlier run, ie the m2_outputs output of that run: an uncompressed tar archive of
        # the Permutect test dataset, the Mutect2 VCF and its index, the contigs table, the Mutect2 stats, and the tumor
        # and normal minor allele fraction segments if Mutect2 made them.  If this is given, Mutect2 is not run.
        # This HACK is required to get around Terra's unreliable call-caching
        # note: defining it as a String instead of a File is essential because otherwise
        # a cached input gs://[some hash]/final-outputs/file.txt is written as
        # gs://[some hash]/final-outputs/[another hash]/final-outputs/file.txt
        # and eventually the file name gets too long and throws an error in Terra/cromwell.
        # if we define it as a String it uses the literal value
        String? cached_m2_outputs

        String? permutect_filtering_extra_args
        String? gcs_project_for_requester_pays
        File? gatk_override
        String permutect_docker
        Int? preemptible
        Int? max_retries
    }

    # I (David Benjamin, that is) always run Permutect on Terra, which has the following extremely irritating bug:
    # whenever you upload new data to your Terra workspace, values in the table that were previously unset
    # (i.e. None, empty Optional) are **silently and invisibly** (there's no sign in the GUI) rewritten as
    # empty String literals.  That is, your Nones become ""s.  This ruins workflows with optional inputs, of course.
    # The workaround is to explicitly test for Strings equal to "", which are actually empty Optionals that
    # Terra converted.

    # WDL version 1.0 does not have an empty Optional literal, but a declaration inside a conditional is an Optional
    # outside of it, so NONE is always an empty Optional.
    if (false) {
        File NONE = ""
    }

    File? intervals_hack = if (select_first([intervals,""]) == "") then NONE else intervals
    File? masks_hack = if (select_first([masks,""]) == "") then NONE else masks

    File? control_bam_hack = if select_first([control_bam, ""]) == "" then NONE else control_bam
    File? control_bai_hack = if select_first([control_bai, ""]) == "" then NONE else control_bai
    File? gnomad_hack = if select_first([gnomad, ""]) == "" then NONE else gnomad
    File? gnomad_idx_hack = if select_first([gnomad_idx, ""]) == "" then NONE else gnomad_idx

    File? variants_for_contamination_hack = if select_first([variants_for_contamination, ""]) == "" then NONE else variants_for_contamination
    File? variants_for_contamination_idx_hack = if select_first([variants_for_contamination_idx, ""]) == "" then NONE else variants_for_contamination_idx
    File? realignment_index_bundle_hack = if select_first([realignment_index_bundle, ""]) == "" then NONE else realignment_index_bundle
    File? dragstr_model_hack = if select_first([dragstr_model, ""]) == "" then NONE else dragstr_model
    File? common_hets_for_segmentation_hack = if select_first([common_hets_for_segmentation, ""]) == "" then NONE else common_hets_for_segmentation
    File? test_dataset_truth_vcf_hack = if select_first([test_dataset_truth_vcf, ""]) == "" then NONE else test_dataset_truth_vcf
    File? test_dataset_truth_vcf_idx_hack = if select_first([test_dataset_truth_vcf_idx, ""]) == "" then NONE else test_dataset_truth_vcf_idx

    String? cached_m2_outputs_hack = if (select_first([cached_m2_outputs,""]) == "") then NONE else cached_m2_outputs

    String? gcs_project_for_requester_pays_hack = if select_first([gcs_project_for_requester_pays, ""]) == "" then NONE else gcs_project_for_requester_pays

    Boolean use_cache = defined(cached_m2_outputs_hack)
    if(!use_cache) {
        call m2.Mutect2 {
            input:
                make_permutect_training_dataset = false,
                make_permutect_test_dataset = true,
                permutect_test_dataset_truth_vcf = test_dataset_truth_vcf_hack,
                permutect_test_dataset_truth_vcf_idx = test_dataset_truth_vcf_idx_hack,
                intervals = intervals_hack,
                masked_intervals = masks_hack,
                ref_fasta = ref_fasta,
                ref_fai = ref_fai,
                ref_dict = ref_dict,
                tumor_reads = primary_bam,
                tumor_reads_index = primary_bai,
                normal_reads = control_bam_hack,
                normal_reads_index = control_bai_hack,

                scatter_count = scatter_count,
                germline_resource = gnomad_hack,
                germline_resource_idx = gnomad_idx_hack,
                variants_for_contamination = variants_for_contamination_hack,
                variants_for_contamination_idx = variants_for_contamination_idx_hack,
                skip_filtering = skip_m2_filtering,
                realignment_index_bundle = realignment_index_bundle_hack,
                realignment_extra_args = realignment_extra_args,
                dragstr_model = dragstr_model_hack,
                run_orientation_bias_mixture_model_filter = run_orientation_bias_mixture_model_filter,
                m2_extra_args = m2_extra_args,
                split_intervals_extra_args = split_intervals_extra_args,
                common_hets_for_segmentation = common_hets_for_segmentation_hack,
                collect_allelic_counts_extra_args = collect_allelic_counts_extra_args,
                model_segments_min_total_allele_count = model_segments_min_total_allele_count,
                model_segments_extra_args = model_segments_extra_args,
                make_bamout = false,

                gatk_docker = permutect_docker,
                gcs_project_for_requester_pays = gcs_project_for_requester_pays_hack,
                gatk_override = gatk_override,
                preemptible = preemptible,
                max_retries = max_retries
        }

        call ArchiveM2Outputs {
            input:
                test_dataset = select_first([Mutect2.permutect_test_dataset]),
                mutect2_vcf = Mutect2.output_vcf,
                mutect2_vcf_idx = Mutect2.output_vcf_idx,
                contigs_table = Mutect2.permutect_contigs_table,
                mutect_stats = Mutect2.mutect_stats,
                maf_segments = Mutect2.modeled_segments_tumor,
                normal_maf_segments = Mutect2.modeled_segments_normal,
                permutect_docker = permutect_docker,
                preemptible = preemptible,
                max_retries = max_retries
        }
    }

    if (use_cache) {
        call ExtractM2Outputs {
            input:
                cached_m2_outputs = select_first([cached_m2_outputs_hack]),
                permutect_docker = permutect_docker,
                preemptible = preemptible,
                max_retries = max_retries
        }
    }

    call SplitMultiallelics {
        input:
            input_vcf = select_first([Mutect2.output_vcf, ExtractM2Outputs.mutect2_vcf]),
            input_vcf_idx = select_first([Mutect2.output_vcf_idx, ExtractM2Outputs.mutect2_vcf_idx]),
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            gatk_docker = permutect_docker
    }

    if (filter_pcr_errors) {
        call PrePermutectFiltering {
            input:
                input_vcf = SplitMultiallelics.output_vcf,
                ref_fasta = ref_fasta,
                ref_fai = ref_fai,
                permutect_docker = permutect_docker
        }
    }

    # minor allele fraction segments from Mutect2's ModelSegments if Mutect2 ran, otherwise the cached segments.
    # Unlike select_first, this allows the segments to be absent.
    File? tumor_segments = if defined(Mutect2.modeled_segments_tumor) then Mutect2.modeled_segments_tumor else ExtractM2Outputs.maf_segments
    File? normal_segments = if defined(Mutect2.modeled_segments_normal) then Mutect2.modeled_segments_normal else ExtractM2Outputs.normal_maf_segments

    call PermutectFiltering {
        input:
            mutect2_vcf = select_first([PrePermutectFiltering.output_vcf, SplitMultiallelics.output_vcf]),
            mutect2_vcf_idx = select_first([PrePermutectFiltering.output_vcf_idx, SplitMultiallelics.output_vcf_idx]),
            artifact_model = artifact_model,
            test_dataset = select_first([Mutect2.permutect_test_dataset, ExtractM2Outputs.test_dataset]),
            contigs_table = select_first([Mutect2.permutect_contigs_table, ExtractM2Outputs.contigs_table]),
            maf_segments = tumor_segments,
            normal_maf_segments = normal_segments,
            mutect_stats = select_first([Mutect2.mutect_stats, ExtractM2Outputs.mutect_stats]),
            batch_size = batch_size,
            num_workers = num_workers,
            gpu_count = gpu_count,
            num_spectrum_iterations = num_spectrum_iterations,
            spectrum_learning_rate = spectrum_learning_rate,
            permutect_filtering_extra_args = permutect_filtering_extra_args,
            permutect_docker = permutect_docker,
    }

    if (defined(test_dataset_truth_vcf_hack)){
        call Concordance as PermutectConcordance {
            input:
                intervals = intervals_hack,
                masks = masks_hack,
                truth_vcf = select_first([test_dataset_truth_vcf_hack]),
                truth_vcf_idx = select_first([test_dataset_truth_vcf_idx_hack]),
                eval_vcf = PermutectFiltering.output_vcf,
                eval_vcf_idx = PermutectFiltering.output_vcf_idx,
                gatk_docker = permutect_docker,
                concordance_header = concordance_header,
        }

        call Concordance as M2Concordance {
            input:
                intervals = intervals_hack,
                masks = masks_hack,
                truth_vcf = select_first([test_dataset_truth_vcf_hack]),
                truth_vcf_idx = select_first([test_dataset_truth_vcf_idx_hack]),
                eval_vcf = select_first([Mutect2.output_vcf, ExtractM2Outputs.mutect2_vcf]),
                eval_vcf_idx = select_first([Mutect2.output_vcf_idx, ExtractM2Outputs.mutect2_vcf_idx]),
                gatk_docker = permutect_docker,
                concordance_header = concordance_header
        }
    }

    output {
        File output_vcf = PermutectFiltering.output_vcf
        File output_vcf_idx = PermutectFiltering.output_vcf_idx
        File tensorboard_report = PermutectFiltering.tensorboard_report
        # Mutect2's outputs, for use as cached_m2_outputs in later runs: the archive of this run's Mutect2 outputs, or
        # else the given cached_m2_outputs, passed through as a String for the reason given where it is declared
        String m2_outputs = select_first([ArchiveM2Outputs.m2_outputs, cached_m2_outputs_hack])

        File? fn = PermutectConcordance.fn
        File? fn_idx = PermutectConcordance.fn_idx
        File? fp = PermutectConcordance.fp
        File? fp_idx = PermutectConcordance.fp_idx
        File? tp = PermutectConcordance.tp
        File? tp_idx = PermutectConcordance.tp_idx
        File? ffn = PermutectConcordance.ffn
        File? ffn_idx = PermutectConcordance.ffn_idx
        File? ftn = PermutectConcordance.ftn
        File? ftn_idx = PermutectConcordance.ftn_idx
        File? concordance_summary = PermutectConcordance.summary
        File? filter_analysis = PermutectConcordance.filter_analysis

        File? m2_concordance_summary = M2Concordance.summary

        File? modeled_segments_tumor = Mutect2.modeled_segments_tumor
        File? modeled_segments_plot_tumor = Mutect2.modeled_segments_plot_tumor
        File? modeled_segments_normal = Mutect2.modeled_segments_normal
        File? modeled_segments_plot_normal = Mutect2.modeled_segments_plot_normal
    }
}

 task PermutectFiltering {
    input {
        File artifact_model
        File test_dataset
        File contigs_table
        File mutect2_vcf
        File mutect2_vcf_idx
        File? maf_segments
        File? normal_maf_segments
        File mutect_stats
        Int? num_spectrum_iterations
        Float? spectrum_learning_rate
        Int batch_size
        Int num_workers
        Int? gpu_count
        String? permutect_filtering_extra_args

        String permutect_docker
        Int? preemptible
        Int? max_retries
        Int? disk_space
        Int? cpu
        Int? mem
    }

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = if defined(mem) then mem * 1000 else 16000
    Int command_mem = machine_mem - 500

    command <<<
        set -euo pipefail
        genomic_span=`grep "callable" ~{mutect_stats} | while read name value; do echo $value; done`

        filter_variants --input ~{mutect2_vcf} --test_dataset ~{test_dataset} \
            --artifact_model ~{artifact_model} \
            --contigs_table ~{contigs_table} \
            --output permutect-filtered.vcf \
            --tensorboard_dir tensorboard \
            --batch_size ~{batch_size} --num_workers ~{num_workers} \
            ~{" --num_spectrum_iterations " + num_spectrum_iterations} \
            ~{" --spectrum_learning_rate " + spectrum_learning_rate} \
            ~{" --maf_segments " + maf_segments} ~{" --normal_maf_segments " + normal_maf_segments} \
            --genomic_span $genomic_span ~{permutect_filtering_extra_args}

        tar cvf tensorboard.tar tensorboard/

        # compress
        bcftools view permutect-filtered.vcf -Oz -o permutect-filtered.vcf.gz

        bcftools index -t permutect-filtered.vcf.gz
    >>>

    runtime {
        docker: permutect_docker
        bootDiskSizeGb: 12
        memory: machine_mem + " MB"
        disks: "local-disk " + select_first([disk_space, 100]) + " SSD"
        preemptible: select_first([preemptible, 0])
        maxRetries: select_first([max_retries, 0])
        cpu: select_first([cpu, 2])
        gpuType: "nvidia-tesla-t4"
        gpuCount: select_first([gpu_count, 1])
        # at least the driver that the docker image's CUDA needs.  Cromwell on Google Batch, and so Terra, ignores this
        # and installs Batch's default driver.
        nvidiaDriverVersion: "580.65.06"
        zones : ["us-central1-a", "us-central1-b", "us-central1-c"]
    }

    output {
        File output_vcf = "permutect-filtered.vcf.gz"
        File output_vcf_idx = "permutect-filtered.vcf.gz.tbi"
        File tensorboard_report = "tensorboard.tar"
    }
}

task PrePermutectFiltering {
    input {
        File input_vcf
        File ref_fasta
        File ref_fai
        String permutect_docker
        Int? preemptible
        Int max_retries = 0
        Int disk_space = 100
        Int cpu = 1
        Int mem = 4
    }

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = mem * 1000
    Int command_mem = machine_mem - 500

    command <<<
        set -e

        pre_permutect_filtering --input ~{input_vcf} --reference ~{ref_fasta} --output pre-filtered.vcf

        gatk --java-options "-Xmx~{command_mem}m" IndexFeatureFile -I pre-filtered.vcf
    >>>

    runtime {
        docker: permutect_docker
        bootDiskSizeGb: 12
        memory: machine_mem + " MB"
        disks: "local-disk " + disk_space + " SSD"
        preemptible: select_first([preemptible, 0])
        maxRetries: max_retries
        cpu: cpu
    }

    output {
        File output_vcf = "pre-filtered.vcf"
        File output_vcf_idx = "pre-filtered.vcf.idx"
    }
}

task SplitMultiallelics {
    input {
        File input_vcf
        File input_vcf_idx
        File ref_fasta
        File ref_fai
        File ref_dict
        String gatk_docker
        Int? preemptible
        Int max_retries = 0
        Int disk_space = 100
        Int cpu = 1
        Int mem = 4
    }

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = mem * 1000
    Int command_mem = machine_mem - 500

    command <<<
        set -euo pipefail

        bcftools norm -m -any -f ~{ref_fasta} ~{input_vcf} > split.vcf

        gatk --java-options "-Xmx~{command_mem}m" IndexFeatureFile -I split.vcf

        gatk --java-options "-Xmx~{command_mem}m" SelectVariants -V split.vcf -O output.vcf --lenient \
            -DGA DP -DGA AF -DGA F1R2 -DGA F2R1 -DGA FAD -DGA SB \
            -DA AS_FilterStatus -DA AS_SB_TABLE -DA ECNT -DA GERMQ -DA MBQ -DA MFRL -DA MMQ -DA MPOS
    >>>

    runtime {
        docker: gatk_docker
        bootDiskSizeGb: 12
        memory: machine_mem + " MB"
        disks: "local-disk " + disk_space + " SSD"
        preemptible: select_first([preemptible, 0])
        maxRetries: max_retries
        cpu: cpu
    }

    output {
        File output_vcf = "output.vcf"
        File output_vcf_idx = "output.vcf.idx"
    }
}

task Concordance {
    input {
    	File? intervals
    	File? masks
    	File truth_vcf
    	File truth_vcf_idx
    	File eval_vcf
    	File eval_vcf_idx
        String? concordance_header

    	# runtime
    	String gatk_docker = "us.gcr.io/broad-gatk/gatk"
    	Int preemptible = 0
	}

    command {
        # this shouldn't be necessary but the Terra hacks on top of Terra hacks to deal with all the bugs recently
        # have reached a breaking point
        gatk IndexFeatureFile -I ~{eval_vcf}

        gatk --java-options "-Xmx2g" Concordance \
            ~{"-L " + intervals} \
            ~{"-XL " + masks} \
            -truth ~{truth_vcf} -eval ~{eval_vcf} \
            -tpfn "tpfn.vcf" \
            -tpfp "tpfp.vcf" \
            -ftnfn "ftnfn.vcf" \
            -filter-analysis "filter-analysis.txt" \
            -summary "summary_no_header.txt"

        echo ~{concordance_header} > TMP
        cat TMP summary_no_header.txt > summary.txt

        grep '#' tpfn.vcf > HEAD
        grep STATUS=FN tpfn.vcf > BODY
        cat HEAD BODY > false_negatives.vcf

        grep '#' tpfp.vcf > HEAD
        grep STATUS=FP tpfp.vcf > BODY
        cat HEAD BODY > false_positives.vcf

        grep '#' tpfp.vcf > HEAD
        grep STATUS=TP tpfp.vcf > BODY
        cat HEAD BODY > true_positives.vcf

        grep '#' ftnfn.vcf > HEAD
        grep STATUS=FFN ftnfn.vcf > BODY
        cat HEAD BODY > filtered_false_negatives.vcf
        grep STATUS=FTN ftnfn.vcf > BODY
        cat HEAD BODY > filtered_true_negatives.vcf

        for vcf in false_negatives.vcf false_positives.vcf true_positives.vcf filtered_false_negatives.vcf filtered_true_negatives.vcf; do
            #HACK: IndexFeatureFile throws error if vcf is empty, which is possible here especially in the case of false negatives
            gatk --java-options "-Xmx2g" SelectVariants -V $vcf -O tmp.vcf
            mv tmp.vcf $vcf
            mv tmp.vcf.idx $vcf.idx
        done
    }

    runtime {
        memory: "5 GB"
        bootDiskSizeGb: 12
        docker: "${gatk_docker}"
        disks: "local-disk " + 100 + " HDD"
        preemptible: select_first([preemptible, 2])
    }

    output {
        File fn = "false_negatives.vcf"
        File fn_idx = "false_negatives.vcf.idx"
        File fp = "false_positives.vcf"
        File fp_idx = "false_positives.vcf.idx"
        File tp = "true_positives.vcf"
        File tp_idx = "true_positives.vcf.idx"
        File ffn = "filtered_false_negatives.vcf"
        File ffn_idx = "filtered_false_negatives.vcf.idx"
        File ftn = "filtered_true_negatives.vcf"
        File ftn_idx = "filtered_true_negatives.vcf.idx"
        File summary = "summary.txt"
        File filter_analysis = "filter-analysis.txt"
    }
}

# bundle Mutect2's outputs into a single uncompressed tar archive, which ExtractM2Outputs unpacks in later runs
task ArchiveM2Outputs {
    input {
        File test_dataset
        File mutect2_vcf
        File mutect2_vcf_idx
        File contigs_table
        File mutect_stats
        File? maf_segments
        File? normal_maf_segments
        String permutect_docker
        Int? preemptible
        Int? max_retries
    }

    # room for the inputs and the archive
    Int disk_space = ceil(2 * size([test_dataset, mutect2_vcf, mutect2_vcf_idx, contigs_table, mutect_stats], "GB")) + 10

    command <<<
        set -euo pipefail

        # Mutect2 makes an uncompressed VCF with a .idx index unless given compress_vcfs, which this workflow never does
        case "~{mutect2_vcf}" in
            *.vcf) ;;
            *) echo "Expected an uncompressed .vcf from Mutect2, got ~{mutect2_vcf}" >&2; exit 1;;
        esac

        # links with the fixed names that ExtractM2Outputs expects
        mkdir archive
        ln -s ~{test_dataset} archive/test_dataset.txt
        ln -s ~{mutect2_vcf} archive/mutect2.vcf
        ln -s ~{mutect2_vcf_idx} archive/mutect2.vcf.idx
        ln -s ~{contigs_table} archive/contigs.table
        ln -s ~{mutect_stats} archive/mutect.stats
        maf_segments="~{default="" maf_segments}"
        normal_maf_segments="~{default="" normal_maf_segments}"
        if [ -n "$maf_segments" ]; then ln -s "$maf_segments" archive/maf_segments.seg; fi
        if [ -n "$normal_maf_segments" ]; then ln -s "$normal_maf_segments" archive/normal_maf_segments.seg; fi

        # uncompressed (no -z), and archiving the files that the links point to (-h)
        tar -chf cached_m2_outputs.tar -C archive .
    >>>

    runtime {
        docker: permutect_docker
        bootDiskSizeGb: 12
        memory: "2 GB"
        disks: "local-disk " + disk_space + " SSD"
        preemptible: select_first([preemptible, 0])
        maxRetries: select_first([max_retries, 0])
        cpu: 1
    }

    output {
        File m2_outputs = "cached_m2_outputs.tar"
    }
}

# unpack the archive made by ArchiveM2Outputs
task ExtractM2Outputs {
    input {
        File cached_m2_outputs
        String permutect_docker
        Int? preemptible
        Int? max_retries
    }

    # room for the archive and its contents
    Int disk_space = ceil(2 * size(cached_m2_outputs, "GB")) + 10

    command <<<
        set -euo pipefail
        tar -xf ~{cached_m2_outputs}
    >>>

    runtime {
        docker: permutect_docker
        bootDiskSizeGb: 12
        memory: "2 GB"
        disks: "local-disk " + disk_space + " SSD"
        preemptible: select_first([preemptible, 0])
        maxRetries: select_first([max_retries, 0])
        cpu: 1
    }

    # the names that ArchiveM2Outputs gives the files
    output {
        File test_dataset = "test_dataset.txt"
        File mutect2_vcf = "mutect2.vcf"
        File mutect2_vcf_idx = "mutect2.vcf.idx"
        File contigs_table = "contigs.table"
        File mutect_stats = "mutect.stats"
        # absent if Mutect2 didn't make them
        File? maf_segments = "maf_segments.seg"
        File? normal_maf_segments = "normal_maf_segments.seg"
    }
}
