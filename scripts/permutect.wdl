version 1.0

import "https://api.firecloud.org/ga4gh/v1/tools/davidben:mutect2-testing/versions/60/plain-WDL/descriptor" as m2

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
        # ignore control_bam and control_bai and run Mutect2 on the primary sample alone, eg to compare a tumor-normal
        # pair's tumor-only calls with its matched-normal calls.  Without control_bam, the run is tumor-only regardless.
        Boolean tumor_only = false
        File? germline_resource
        # common SNPs with population allele frequencies, for segmenting the tumor and normal by minor allele fraction
        # and estimating the tumor's contamination with segment_pileups.  Their allele frequencies must be accurate, eg
        # from gnomAD.  For genomes these should be common SNPs throughout the genome, not only in exons.
        File? common_variants_vcf
        File? test_dataset_truth_vcf    # used for evaluation

        # Indices of the VCFs above, usually unnecessary on both local and cloud backends.  Each VCF's index is assumed to
        # be the VCF's path plus .tbi for a bgzipped VCF (.vcf.gz or .vcf.bgz) or .idx for a plain .vcf.  Specify an
        # index only to override that, for example for a non-standard extension.
        File? germline_resource_idx
        File? common_variants_vcf_idx
        File? test_dataset_truth_vcf_idx

        File? realignment_index_bundle
        File? dragstr_model
        String? realignment_extra_args
        Boolean skip_m2_filtering = true
        # Mutect2 collects F1R2 counts, from which learn_read_orientation_model learns the priors of GATK's read
        # orientation artifact model.  With cached Mutect2 outputs, the cache must come from a run with this true.
        Boolean run_orientation_bias_mixture_model_filter = false
        # apply pre_permutect_filtering's hairpin filter.  pre_permutect_filtering runs regardless, for its multiallelic
        # indel filter.
        Boolean filter_pcr_errors = false
        String? m2_extra_args
        String? split_intervals_extra_args
        Int batch_size
        Int num_workers
        Int? gpu_count
        String? concordance_header
        # split each multi-nucleotide variant (MNV) into its constituent SNVs, eg to compare with a truth VCF that represents
        # MNVs that way.  Concordance then evaluates the split VCFs; output_vcf keeps the MNVs.
        Boolean split_mnvs = false

        # Mutect2's outputs from an earlier run, ie the m2_outputs output of that run: an uncompressed tar archive of
        # the Permutect test dataset, the Mutect2 VCF and its index, the contigs table, the Mutect2 stats, the tumor
        # and normal pileups at the common variants if Mutect2 made them, and the F1R2 counts of the scatters if
        # Mutect2 ran with run_orientation_bias_mixture_model_filter.  If this is given, Mutect2 is not run.
        # This HACK is required to get around Terra's unreliable call-caching
        # note: defining it as a String instead of a File is essential because otherwise
        # a cached input gs://[some hash]/final-outputs/file.txt is written as
        # gs://[some hash]/final-outputs/[another hash]/final-outputs/file.txt
        # and eventually the file name gets too long and throws an error in Terra/cromwell.
        # if we define it as a String it uses the literal value
        String? cached_m2_outputs
        # likewise for tumor-only runs, ie runs with tumor_only true or without control_bam: the tumor_only_m2_outputs
        # output of an earlier run.  A tumor-only run uses this instead of cached_m2_outputs, which may hold
        # matched-normal outputs.
        String? tumor_only_cached_m2_outputs

        String? segment_pileups_extra_args
        Int? segment_pileups_mem
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
    File? germline_resource_hack = if select_first([germline_resource, ""]) == "" then NONE else germline_resource
    File? germline_resource_idx_hack = if select_first([germline_resource_idx, ""]) == "" then NONE else germline_resource_idx

    File? common_variants_vcf_hack = if select_first([common_variants_vcf, ""]) == "" then NONE else common_variants_vcf
    File? common_variants_vcf_idx_hack = if select_first([common_variants_vcf_idx, ""]) == "" then NONE else common_variants_vcf_idx
    File? realignment_index_bundle_hack = if select_first([realignment_index_bundle, ""]) == "" then NONE else realignment_index_bundle
    File? dragstr_model_hack = if select_first([dragstr_model, ""]) == "" then NONE else dragstr_model
    File? test_dataset_truth_vcf_hack = if select_first([test_dataset_truth_vcf, ""]) == "" then NONE else test_dataset_truth_vcf
    File? test_dataset_truth_vcf_idx_hack = if select_first([test_dataset_truth_vcf_idx, ""]) == "" then NONE else test_dataset_truth_vcf_idx

    # The truth VCF's index, unless given, is its path plus .tbi if it's bgzipped or .idx if not.  Concordance needs it
    # here.  Mutect2 finds the indices of the other VCFs the same way.
    if (defined(test_dataset_truth_vcf_hack)) {
        String test_dataset_truth_vcf_path = "~{test_dataset_truth_vcf_hack}"
        File conventional_test_dataset_truth_vcf_idx = test_dataset_truth_vcf_path + (if sub(test_dataset_truth_vcf_path, "\\.b?gz$", "") != test_dataset_truth_vcf_path then ".tbi" else ".idx")
    }
    File? test_dataset_truth_vcf_idx_or_conventional = if defined(test_dataset_truth_vcf_idx_hack) then test_dataset_truth_vcf_idx_hack else conventional_test_dataset_truth_vcf_idx

    File? gatk_override_hack = if select_first([gatk_override, ""]) == "" then NONE else gatk_override

    String? cached_m2_outputs_hack = if (select_first([cached_m2_outputs,""]) == "") then NONE else cached_m2_outputs
    String? tumor_only_cached_m2_outputs_hack = if (select_first([tumor_only_cached_m2_outputs,""]) == "") then NONE else tumor_only_cached_m2_outputs

    String? gcs_project_for_requester_pays_hack = if select_first([gcs_project_for_requester_pays, ""]) == "" then NONE else gcs_project_for_requester_pays

    # whether Mutect2 runs without a normal, which decides the cached Mutect2 outputs to use and the outputs to write
    Boolean tumor_only_mode = tumor_only || !defined(control_bam_hack)
    File? normal_bam = if tumor_only_mode then NONE else control_bam_hack
    File? normal_bai = if tumor_only_mode then NONE else control_bai_hack
    String? cached_m2_outputs_to_use = if tumor_only_mode then tumor_only_cached_m2_outputs_hack else cached_m2_outputs_hack

    Boolean use_cache = defined(cached_m2_outputs_to_use)
    if(!use_cache) {
        call m2.Mutect2 {
            input:
                make_permutect_training_dataset = false,
                make_permutect_test_dataset = true,
                permutect_test_dataset_truth_vcf = test_dataset_truth_vcf_hack,
                permutect_test_dataset_truth_vcf_idx = test_dataset_truth_vcf_idx_or_conventional,
                intervals = intervals_hack,
                masked_intervals = masks_hack,
                ref_fasta = ref_fasta,
                ref_fai = ref_fai,
                ref_dict = ref_dict,
                tumor_reads = primary_bam,
                tumor_reads_index = primary_bai,
                normal_reads = normal_bam,
                normal_reads_index = normal_bai,

                scatter_count = scatter_count,
                germline_resource = germline_resource_hack,
                germline_resource_idx = germline_resource_idx_hack,
                common_variants_vcf = common_variants_vcf_hack,
                common_variants_vcf_idx = common_variants_vcf_idx_hack,
                skip_filtering = skip_m2_filtering,
                # segment_pileups estimates the contamination instead
                skip_calculate_contamination = true,
                realignment_index_bundle = realignment_index_bundle_hack,
                realignment_extra_args = realignment_extra_args,
                dragstr_model = dragstr_model_hack,
                run_orientation_bias_mixture_model_filter = run_orientation_bias_mixture_model_filter,
                m2_extra_args = m2_extra_args,
                split_intervals_extra_args = split_intervals_extra_args,
                # segment_pileups segments the samples using pileups at common variants with population allele
                # frequencies up to 0.99, rather than the lower maximum of CalculateContamination
                pileups_max_population_af = 0.99,
                make_bamout = false,

                gatk_docker = permutect_docker,
                gcs_project_for_requester_pays = gcs_project_for_requester_pays_hack,
                gatk_override = gatk_override_hack,
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
                tumor_pileups = Mutect2.tumor_pileups,
                normal_pileups = Mutect2.normal_pileups,
                f1r2_counts = Mutect2.f1r2_counts,
                permutect_docker = permutect_docker,
                preemptible = preemptible,
                max_retries = max_retries
        }
    }

    if (use_cache) {
        call ExtractM2Outputs {
            input:
                cached_m2_outputs = select_first([cached_m2_outputs_to_use]),
                permutect_docker = permutect_docker,
                preemptible = preemptible,
                max_retries = max_retries
        }
    }

    # Mutect2's outputs for this run's mode: the archive of this run's Mutect2 outputs, or else the given cache
    String this_run_m2_outputs = select_first([ArchiveM2Outputs.m2_outputs, cached_m2_outputs_to_use])

    call SplitMultiallelics {
        input:
            input_vcf = select_first([Mutect2.output_vcf, ExtractM2Outputs.mutect2_vcf]),
            input_vcf_idx = select_first([Mutect2.output_vcf_idx, ExtractM2Outputs.mutect2_vcf_idx]),
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            gatk_docker = permutect_docker
    }

    call PrePermutectFiltering {
        input:
            input_vcf = SplitMultiallelics.output_vcf,
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            hairpin_filter = filter_pcr_errors,
            permutect_docker = permutect_docker
    }

    # Mutect2's pileups at the common variants if Mutect2 ran, otherwise the cached pileups.  Unlike select_first, this
    # allows them to be absent, as they are without common variants.
    File? tumor_pileups = if defined(Mutect2.tumor_pileups) then Mutect2.tumor_pileups else ExtractM2Outputs.tumor_pileups
    File? normal_pileups = if defined(Mutect2.normal_pileups) then Mutect2.normal_pileups else ExtractM2Outputs.normal_pileups
    # likewise the F1R2 counts of Mutect2's scatters, which it makes with run_orientation_bias_mixture_model_filter
    Array[File] f1r2_counts = select_first([Mutect2.f1r2_counts, ExtractM2Outputs.f1r2_counts])

    # A cache from a run without run_orientation_bias_mixture_model_filter has no F1R2 counts, and the task's
    # non-empty array input then fails
    if (run_orientation_bias_mixture_model_filter) {
        call LearnReadOrientationModel {
            input:
                f1r2_counts = f1r2_counts,
                permutect_docker = permutect_docker,
                preemptible = preemptible,
                max_retries = max_retries
        }
    }

    # without pileups, filter_variants uses minor allele fractions of 1/2 and default germline likelihoods
    if (defined(tumor_pileups)) {
        call SegmentPileups {
            input:
                tumor_pileups = select_first([tumor_pileups]),
                normal_pileups = normal_pileups,
                segment_pileups_extra_args = segment_pileups_extra_args,
                permutect_docker = permutect_docker,
                mem = segment_pileups_mem,
                preemptible = preemptible,
                max_retries = max_retries
        }
    }

    call PermutectFiltering {
        input:
            mutect2_vcf = PrePermutectFiltering.output_vcf,
            mutect2_vcf_idx = PrePermutectFiltering.output_vcf_idx,
            artifact_model = artifact_model,
            test_dataset = select_first([Mutect2.permutect_test_dataset, ExtractM2Outputs.test_dataset]),
            contigs_table = select_first([Mutect2.permutect_contigs_table, ExtractM2Outputs.contigs_table]),
            germline_model = SegmentPileups.germline_model,
            mutect_stats = select_first([Mutect2.mutect_stats, ExtractM2Outputs.mutect_stats]),
            batch_size = batch_size,
            num_workers = num_workers,
            gpu_count = gpu_count,
            num_spectrum_iterations = num_spectrum_iterations,
            spectrum_learning_rate = spectrum_learning_rate,
            permutect_filtering_extra_args = permutect_filtering_extra_args,
            permutect_docker = permutect_docker,
    }

    if (split_mnvs) {
        call SplitMNVs as SplitPermutectMNVs {
            input:
                input_vcf = PermutectFiltering.output_vcf,
                permutect_docker = permutect_docker
        }
    }

    if (defined(test_dataset_truth_vcf_hack)){
        if (split_mnvs) {
            call SplitMNVs as SplitM2MNVs {
                input:
                    input_vcf = select_first([Mutect2.output_vcf, ExtractM2Outputs.mutect2_vcf]),
                    permutect_docker = permutect_docker
            }
        }

        call Concordance as PermutectConcordance {
            input:
                intervals = intervals_hack,
                masks = masks_hack,
                truth_vcf = select_first([test_dataset_truth_vcf_hack]),
                truth_vcf_idx = select_first([test_dataset_truth_vcf_idx_or_conventional]),
                eval_vcf = select_first([SplitPermutectMNVs.output_vcf, PermutectFiltering.output_vcf]),
                eval_vcf_idx = select_first([SplitPermutectMNVs.output_vcf_idx, PermutectFiltering.output_vcf_idx]),
                gatk_docker = permutect_docker,
                concordance_header = concordance_header,
        }

        call Concordance as M2Concordance {
            input:
                intervals = intervals_hack,
                masks = masks_hack,
                truth_vcf = select_first([test_dataset_truth_vcf_hack]),
                truth_vcf_idx = select_first([test_dataset_truth_vcf_idx_or_conventional]),
                eval_vcf = select_first([SplitM2MNVs.output_vcf, Mutect2.output_vcf, ExtractM2Outputs.mutect2_vcf]),
                eval_vcf_idx = select_first([SplitM2MNVs.output_vcf_idx, Mutect2.output_vcf_idx, ExtractM2Outputs.mutect2_vcf_idx]),
                gatk_docker = permutect_docker,
                concordance_header = concordance_header
        }
    }

    output {
        # Permutect's calls go to output_vcf in a matched-normal run and to tumor_only_output_vcf in a tumor-only run
        File? output_vcf = if tumor_only_mode then NONE else PermutectFiltering.output_vcf
        File? output_vcf_idx = if tumor_only_mode then NONE else PermutectFiltering.output_vcf_idx
        File? tumor_only_output_vcf = if tumor_only_mode then PermutectFiltering.output_vcf else NONE
        File? tumor_only_output_vcf_idx = if tumor_only_mode then PermutectFiltering.output_vcf_idx else NONE
        # output_vcf with each MNV split into SNVs, if split_mnvs is true
        File? mnv_split_vcf = SplitPermutectMNVs.output_vcf
        File? mnv_split_vcf_idx = SplitPermutectMNVs.output_vcf_idx
        File tensorboard_report = PermutectFiltering.tensorboard_report
        # Mutect2's outputs, for use as cached_m2_outputs in later runs: the archive of this run's Mutect2 outputs, or
        # else the given cached_m2_outputs, passed through as a String for the reason given where it is declared.
        # A tumor-only run passes cached_m2_outputs through, so that it doesn't overwrite matched-normal outputs in Terra.
        String? m2_outputs = if tumor_only_mode then cached_m2_outputs_hack else this_run_m2_outputs
        # likewise for tumor_only_cached_m2_outputs
        String? tumor_only_m2_outputs = if tumor_only_mode then this_run_m2_outputs else tumor_only_cached_m2_outputs_hack

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

        # segment_pileups' outputs, if there were common variants
        File? germline_model = SegmentPileups.germline_model
        File? tumor_segments = SegmentPileups.tumor_segments
        File? tumor_segments_plot = SegmentPileups.tumor_segments_plot
        File? normal_segments = SegmentPileups.normal_segments
        File? normal_segments_plot = SegmentPileups.normal_segments_plot
        File? contamination_table = SegmentPileups.contamination_table

        # learn_read_orientation_model's priors, if run_orientation_bias_mixture_model_filter is true
        File? read_orientation_priors = LearnReadOrientationModel.priors
    }
}

 task PermutectFiltering {
    input {
        File artifact_model
        File test_dataset
        File contigs_table
        File mutect2_vcf
        File mutect2_vcf_idx
        File? germline_model
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
    Int num_cpus = select_first([cpu, 4])

    command <<<
        set -euo pipefail
        genomic_span=`grep "callable" ~{mutect_stats} | while read name value; do echo $value; done`

        # one process per CPU reads and normalizes chunks of the test dataset
        filter_variants --input ~{mutect2_vcf} --test_dataset ~{test_dataset} \
            --artifact_model ~{artifact_model} \
            --contigs_table ~{contigs_table} \
            --output permutect-filtered.vcf \
            --tensorboard_dir tensorboard \
            --batch_size ~{batch_size} --num_workers ~{num_workers} --num_processes ~{num_cpus} \
            ~{" --num_spectrum_iterations " + num_spectrum_iterations} \
            ~{" --spectrum_learning_rate " + spectrum_learning_rate} \
            ~{" --germline_model " + germline_model} \
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
        cpu: num_cpus
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
        Boolean hairpin_filter
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

        pre_permutect_filtering --input ~{input_vcf} --reference ~{ref_fasta} --output pre-filtered.vcf \
            ~{if hairpin_filter then "" else "--no_hairpin_filter"}

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

# decompose each multi-nucleotide variant (MNV) into its constituent SNVs, each of which keeps the MNV's FILTER, INFO and
# FORMAT fields.  Multiallelic records are split first, because atomizing a multiallelic record with an MNV allele gives
# star alleles.  Other variants are unchanged.
task SplitMNVs {
    input {
        File input_vcf
        String permutect_docker
        Int? preemptible
        Int max_retries = 0
        Int disk_space = 20
        Int cpu = 1
        Int mem = 2
    }

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = mem * 1000

    command <<<
        set -euo pipefail
        bcftools norm -m -any --atomize ~{input_vcf} -Oz -o mnvs-split.vcf.gz
        bcftools index -t mnvs-split.vcf.gz
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
        File output_vcf = "mnvs-split.vcf.gz"
        File output_vcf_idx = "mnvs-split.vcf.gz.tbi"
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

    command <<<
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

        # write the header and the records with a given STATUS of a Concordance output to a new indexed VCF
        extract_status() {
            grep '#' $1 > HEAD
            grep STATUS=$2 $1 > BODY
            cat HEAD BODY > $3
            #HACK: IndexFeatureFile throws error if vcf is empty, which is possible here especially in the case of false negatives
            gatk --java-options "-Xmx2g" SelectVariants -V $3 -O tmp.vcf
            mv tmp.vcf $3
            mv tmp.vcf.idx $3.idx
        }

        extract_status tpfn.vcf FN false_negatives.vcf
        extract_status tpfp.vcf FP false_positives.vcf
        extract_status tpfp.vcf TP true_positives.vcf
        extract_status ftnfn.vcf FFN filtered_false_negatives.vcf
        extract_status ftnfn.vcf FTN filtered_true_negatives.vcf
    >>>

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

# segment the tumor, and the normal if there is one, by minor allele fraction, and estimate the tumor's contamination
task SegmentPileups {
    input {
        File tumor_pileups
        File? normal_pileups
        String? segment_pileups_extra_args

        String permutect_docker
        Int? preemptible
        Int? max_retries
        Int? cpu
        Int? mem
    }

    # Mem is in units of GB but our runtime value is in MB.  TODO: the segmentation's memory grows with the number of
    # sites, and for the tens of millions of common SNPs of a genome each sites x MAF states array takes about 11 GB.
    Int machine_mem = if defined(mem) then mem * 1000 else 16000
    Int disk_space = ceil(2 * size(tumor_pileups, "GB") + 2 * size(normal_pileups, "GB")) + 10

    command <<<
        set -euo pipefail
        normal_pileups="~{default="" normal_pileups}"
        normal_args=""
        if [ -n "$normal_pileups" ]; then
            normal_args="--normal_input $normal_pileups --normal_output normal.segments.table --normal_plot normal.segments.png"
        fi

        segment_pileups --input ~{tumor_pileups} --output tumor.segments.table --plot tumor.segments.png \
            $normal_args --contamination_table contamination.table --germline_model germline-model.pt \
            ~{segment_pileups_extra_args}
    >>>

    runtime {
        docker: permutect_docker
        bootDiskSizeGb: 12
        memory: machine_mem + " MB"
        disks: "local-disk " + disk_space + " SSD"
        preemptible: select_first([preemptible, 0])
        maxRetries: select_first([max_retries, 0])
        cpu: select_first([cpu, 4])
    }

    output {
        File germline_model = "germline-model.pt"
        File tumor_segments = "tumor.segments.table"
        File tumor_segments_plot = "tumor.segments.png"
        File? normal_segments = "normal.segments.table"
        File? normal_segments_plot = "normal.segments.png"
        File contamination_table = "contamination.table"
    }
}

# learn the priors of GATK's read orientation artifact model from the F1R2 counts of Mutect2's scatters, as GATK
# LearnReadOrientationModel does, in the format of its output
task LearnReadOrientationModel {
    input {
        Array[File]+ f1r2_counts

        String permutect_docker
        Int? preemptible
        Int? max_retries
        Int? mem
    }

    # Mem is in units of GB but our runtime value is in MB.  The tool reads the counts without extracting them.
    Int machine_mem = if defined(mem) then mem * 1000 else 4000
    Int disk_space = ceil(size(f1r2_counts, "GB")) + 10

    command <<<
        set -euo pipefail
        learn_read_orientation_model --input ~{sep=" " f1r2_counts} --output orientation-priors.tar.gz
    >>>

    runtime {
        docker: permutect_docker
        bootDiskSizeGb: 12
        memory: machine_mem + " MB"
        disks: "local-disk " + disk_space + " SSD"
        preemptible: select_first([preemptible, 0])
        maxRetries: select_first([max_retries, 0])
        cpu: 1
    }

    output {
        File priors = "orientation-priors.tar.gz"
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
        File? tumor_pileups
        File? normal_pileups
        Array[File] f1r2_counts = []
        String permutect_docker
        Int? preemptible
        Int? max_retries
    }

    # room for the inputs and the archive
    Int disk_space = ceil(2 * (size([test_dataset, mutect2_vcf, mutect2_vcf_idx, contigs_table, mutect_stats], "GB")
        + size(tumor_pileups, "GB") + size(normal_pileups, "GB") + size(f1r2_counts, "GB"))) + 10

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
        tumor_pileups="~{default="" tumor_pileups}"
        normal_pileups="~{default="" normal_pileups}"
        if [ -n "$tumor_pileups" ]; then ln -s "$tumor_pileups" archive/tumor-pileups.table; fi
        if [ -n "$normal_pileups" ]; then ln -s "$normal_pileups" archive/normal-pileups.table; fi
        # the F1R2 counts of the scatters, numbered since each scatter's is named f1r2.tar.gz
        mkdir archive/f1r2-counts
        n=0
        for counts in ~{sep=" " f1r2_counts}; do
            ln -s "$counts" archive/f1r2-counts/scatter-$(printf "%04d" $n).tar.gz
            n=$((n + 1))
        done

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
        # absent if Mutect2 didn't make them, ie without common variants, and the normal's without a normal
        File? tumor_pileups = "tumor-pileups.table"
        File? normal_pileups = "normal-pileups.table"
        # the F1R2 counts of the scatters, none if Mutect2 ran without run_orientation_bias_mixture_model_filter
        Array[File] f1r2_counts = glob("f1r2-counts/*.tar.gz")
    }
}
