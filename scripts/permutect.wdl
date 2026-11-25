version 1.0

import "https://api.firecloud.org/ga4gh/v1/tools/davidben:mutect2-testing/versions/61/plain-WDL/descriptor" as m2

workflow Permutect {
    input {
        File artifact_model

        # Files of the genomic intervals to call, such as .interval_list or .bed files, or [] to call the whole genome.
        # Mutect2's WDL and Concordance give GATK each file with its own -L, and GATK takes the union of their
        # intervals.  In a Terra configuration, an attribute expression such as workspace.intervals gives a one-element
        # array.  Elements that are "" are ignored, since Terra may turn unset data table values into "" (see below).
        # (Earlier versions took a single, optional file.)
        Array[File] intervals
        File? masks
        # The reference FASTA.  Its companion files are not inputs: Mutect2 (GATK), bcftools and pysam find them by name in
        # the FASTA's directory, and GATK accepts no other names, so they must exist there under these names:
        # - its index, the FASTA's path plus .fai, eg Homo_sapiens_assembly38.fasta.fai
        # - its sequence dictionary, the FASTA's path with its extension (.fasta, .fa, .fna, .fas or .txt, optionally
        #   followed by .gz) replaced by .dict, eg Homo_sapiens_assembly38.dict
        # - for a bgzipped FASTA (.gz), its block index, the FASTA's path plus .gzi
        # The workflow passes these files to tasks along with the FASTA, so that backends that localize files, such as a
        # local Cromwell, put them beside it, while on Terra Mutect2 streams them from the bucket.  A missing companion
        # file fails the workflow early, with an error naming the file.
        File ref_fasta
        Int scatter_count
        Int? num_spectrum_iterations
        Float? spectrum_learning_rate
        Int? spectrum_batch_size
        # The primary (tumor) sample's reads, which Mutect2's WDL takes as tumor_reads: an array whose first element is a
        # BAM or CRAM file and whose optional second element is its index.  GATK finds the index only by name in the
        # reads' directory, so it must be there under one of these names, whether or not the array includes it:
        # - for x.bam: x.bai, x.bam.bai, x.csi or x.bam.csi
        # - for x.cram: x.crai or x.cram.crai
        # On Terra, Mutect2 streams the reads from the bucket and finds the index there, so the BAM or CRAM alone
        # suffices: in a configuration, an attribute expression such as this.tumor_bam gives a one-element array, and
        # [this.tumor_bam, this.tumor_bai] a two-element one.  On a backend that localizes files, such as a local
        # Cromwell, include the index, so that it is localized beside the reads.  Including it always works.  Elements
        # that are "" are ignored, since Terra may turn unset data table values into "" (see below).  (Earlier versions
        # took the index as the separate input primary_bai.)
        Array[File]+ primary_bam
        # The control (normal) sample's reads, likewise, or [] for a tumor-only run.  In a Terra configuration,
        # this.normal_bam or [this.normal_bam, this.normal_bai] gives [] if the attributes are unset.  (Earlier versions
        # took the index as the separate input control_bai, and the reads as an optional input that was left unset for a
        # tumor-only run.)
        Array[File] control_bam
        # ignore control_bam and run Mutect2 on the primary sample alone, eg to compare a tumor-normal pair's tumor-only
        # calls with its matched-normal calls.  With control_bam empty, the run is tumor-only regardless.
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
        # filter_variants' DataLoader workers, by default 2 if PermutectFiltering has at least 4 CPUs and none otherwise
        Int? num_workers
        Int? gpu_count
        String? concordance_header
        # split each multi-nucleotide variant (MNV) into its constituent SNVs, eg to compare with a truth VCF that represents
        # MNVs that way.  Concordance then evaluates the split VCF; output_vcf keeps the MNVs.
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

    File? masks_hack = if (select_first([masks,""]) == "") then NONE else masks

    # The reads and intervals arrays without their "" elements, which Terra may have made from unset data table values,
    # eg [this.normal_bam, this.normal_bai] becoming ["", ""].  WDL has no filter function, so a scatter without calls
    # keeps the other elements, and select_all collects them.
    scatter (primary_file in primary_bam) {
        if (primary_file != "") {
            File given_primary_file = primary_file
        }
    }
    scatter (control_file in control_bam) {
        if (control_file != "") {
            File given_control_file = control_file
        }
    }
    scatter (interval_file in intervals) {
        if (interval_file != "") {
            File given_interval_file = interval_file
        }
    }
    Array[File] primary_reads = select_all(given_primary_file)
    Array[File] control_reads = select_all(given_control_file)
    Array[File] given_intervals = select_all(given_interval_file)

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

    # The reference FASTA and its companion files for this workflow's tasks, derived as Mutect2's WDL derives them for
    # its own: the index, the FASTA's path plus .fai; the dictionary, the FASTA's path with its extension replaced by
    # .dict; and for a bgzipped FASTA the block index, its path plus .gzi.  Tasks use reference[0] as the FASTA.
    String ref_fasta_path = "~{ref_fasta}"
    Array[String] ref_gzi = if sub(ref_fasta_path, "\\.gz$", "") != ref_fasta_path then [ref_fasta_path + ".gzi"] else []
    Array[File] reference = flatten([[ref_fasta_path, ref_fasta_path + ".fai",
        sub(ref_fasta_path, "\\.(fasta|fas|fa|fna|txt)(\\.gz)?$", "") + ".dict"], ref_gzi])

    String? cached_m2_outputs_hack = if (select_first([cached_m2_outputs,""]) == "") then NONE else cached_m2_outputs
    String? tumor_only_cached_m2_outputs_hack = if (select_first([tumor_only_cached_m2_outputs,""]) == "") then NONE else tumor_only_cached_m2_outputs

    String? gcs_project_for_requester_pays_hack = if select_first([gcs_project_for_requester_pays, ""]) == "" then NONE else gcs_project_for_requester_pays

    # whether Mutect2 runs without a normal, which decides the cached Mutect2 outputs to use and the outputs to write
    Boolean tumor_only_mode = tumor_only || length(control_reads) == 0
    Array[File] normal_reads = if tumor_only_mode then [] else control_reads
    String? cached_m2_outputs_to_use = if tumor_only_mode then tumor_only_cached_m2_outputs_hack else cached_m2_outputs_hack

    Boolean use_cache = defined(cached_m2_outputs_to_use)
    if(!use_cache) {
        call m2.Mutect2 {
            input:
                make_permutect_training_dataset = false,
                make_permutect_test_dataset = true,
                permutect_test_dataset_truth_vcf = test_dataset_truth_vcf_hack,
                permutect_test_dataset_truth_vcf_idx = test_dataset_truth_vcf_idx_or_conventional,
                intervals = given_intervals,
                masked_intervals = masks_hack,
                ref_fasta = ref_fasta,
                tumor_reads = primary_reads,
                normal_reads = normal_reads,

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
            reference = reference,
            gatk_docker = permutect_docker
    }

    # Mutect2's pileups at the common variants if Mutect2 ran, otherwise the cached pileups.  Unlike select_first, this
    # allows them to be absent, as they are without common variants.
    File? tumor_pileups = if defined(Mutect2.tumor_pileups) then Mutect2.tumor_pileups else ExtractM2Outputs.tumor_pileups
    File? normal_pileups = if defined(Mutect2.normal_pileups) then Mutect2.normal_pileups else ExtractM2Outputs.normal_pileups
    # likewise the F1R2 counts of Mutect2's scatters, which it makes with run_orientation_bias_mixture_model_filter
    Array[File] f1r2_counts = select_first([Mutect2.f1r2_counts, ExtractM2Outputs.f1r2_counts])

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

    # A cache from a run without run_orientation_bias_mixture_model_filter has no F1R2 counts, and the task's
    # non-empty array input then fails.  Without pileups, there is no germline model to pass along.
    if (run_orientation_bias_mixture_model_filter) {
        call LearnReadOrientationModel {
            input:
                f1r2_counts = f1r2_counts,
                germline_model = SegmentPileups.germline_model,
                permutect_docker = permutect_docker,
                preemptible = preemptible,
                max_retries = max_retries
        }
    }

    call PermutectFiltering {
        input:
            mutect2_vcf = SplitMultiallelics.output_vcf,
            reference = reference,
            hairpin_filter = filter_pcr_errors,
            artifact_model = artifact_model,
            test_dataset = select_first([Mutect2.permutect_test_dataset, ExtractM2Outputs.test_dataset]),
            contigs_table = select_first([Mutect2.permutect_contigs_table, ExtractM2Outputs.contigs_table]),
            germline_model = SegmentPileups.germline_model,
            read_orientation_model = LearnReadOrientationModel.read_orientation_model,
            mutect_stats = select_first([Mutect2.mutect_stats, ExtractM2Outputs.mutect_stats]),
            batch_size = batch_size,
            num_workers = num_workers,
            gpu_count = gpu_count,
            num_spectrum_iterations = num_spectrum_iterations,
            spectrum_learning_rate = spectrum_learning_rate,
            spectrum_batch_size = spectrum_batch_size,
            permutect_filtering_extra_args = permutect_filtering_extra_args,
            permutect_docker = permutect_docker,
    }

    if (defined(test_dataset_truth_vcf_hack)){
        call Concordance as PermutectConcordance {
            input:
                intervals = given_intervals,
                masks = masks_hack,
                truth_vcf = select_first([test_dataset_truth_vcf_hack]),
                truth_vcf_idx = select_first([test_dataset_truth_vcf_idx_or_conventional]),
                eval_vcf = PermutectFiltering.output_vcf,
                eval_vcf_idx = PermutectFiltering.output_vcf_idx,
                split_mnvs = split_mnvs,
                gatk_docker = permutect_docker,
                concordance_header = concordance_header,
        }
    }

    output {
        # Permutect's calls, in both matched-normal and tumor-only runs
        File output_vcf = PermutectFiltering.output_vcf
        File output_vcf_idx = PermutectFiltering.output_vcf_idx
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

        # segment_pileups' outputs, if there were common variants
        File? germline_model = SegmentPileups.germline_model
        File? tumor_segments = SegmentPileups.tumor_segments
        File? tumor_segments_plot = SegmentPileups.tumor_segments_plot
        File? normal_segments = SegmentPileups.normal_segments
        File? normal_segments_plot = SegmentPileups.normal_segments_plot
        File? contamination_table = SegmentPileups.contamination_table

        # learn_read_orientation_model's model, if run_orientation_bias_mixture_model_filter is true
        File? read_orientation_model = LearnReadOrientationModel.read_orientation_model
    }
}

 task PermutectFiltering {
    input {
        File artifact_model
        File test_dataset
        File contigs_table
        # Mutect2's VCF with its multiallelic records split, which pre_permutect_filtering filters for filter_variants
        File mutect2_vcf
        # the reference FASTA, then its .fai index, its .dict dictionary and, if it's bgzipped, its .gzi block index
        Array[File] reference
        # whether pre_permutect_filtering applies its hairpin filter as well as its multiallelic indel filter
        Boolean hairpin_filter
        File? germline_model
        File? read_orientation_model
        File mutect_stats
        Int? num_spectrum_iterations
        Float? spectrum_learning_rate
        Int? spectrum_batch_size
        Int batch_size
        Int? num_workers
        Int? gpu_count
        String? permutect_filtering_extra_args

        String permutect_docker
        Int? preemptible
        Int? max_retries
        Int? disk_space
        Int? cpu
        # memory in GB, which overrides the estimate below
        Int? mem
    }

    Int num_cpus = select_first([cpu, 4])
    # In training, 2 DataLoader workers were as fast as more, and without them it was 16% slower.  They and the main process
    # need at least 4 CPUs (2 cores).
    Int num_dataloader_workers = select_first([num_workers, if num_cpus >= 4 then 2 else 0])

    # Without mem, an estimate of the peak memory from measurements in November 2026, plus 1.5 GB of margin:
    # - preprocessing the test dataset, in a spawned process per CPU that imports torch and permutect, takes about 1.8 GB
    #   plus 0.62 GB per process, whatever the dataset's size
    # - afterwards memory peaks at about 3.75 GB plus 0.64 GB per GB of test dataset.  The data loader's chunks are at most
    #   an eighth of the free memory, so this overestimates for a smaller machine.
    # - each DataLoader worker adds up to about 0.5 GB.  The batch size and the number of GPUs don't matter.
    Float preprocessing_gb = 1.8 + 0.62 * num_cpus
    Float filtering_gb = 3.75 + 0.64 * size(test_dataset, "GB")
    Float peak_gb = (if preprocessing_gb > filtering_gb then preprocessing_gb else filtering_gb) + 0.5 * num_dataloader_workers
    # in MB as Cromwell counts them, 2^20 bytes, so a little more than the estimate
    Int machine_mem = if defined(mem) then mem * 1000 else ceil((peak_gb + 1.5) * 1000)
    # the reference, the test dataset and its normalized copy, in temporary memory maps, which is smaller, and the VCF, as
    # given, pre-filtered and filtered
    Int default_disk_space = ceil(size(reference, "GB") + 2 * size(test_dataset, "GB") + 3 * size(mutect2_vcf, "GB")) + 10

    command <<<
        set -euo pipefail
        genomic_span=`grep "callable" ~{mutect_stats} | while read name value; do echo $value; done`

        pre_permutect_filtering --input ~{mutect2_vcf} --reference ~{reference[0]} --output pre-filtered.vcf \
            ~{if hairpin_filter then "" else "--no_hairpin_filter"}

        # one process per CPU reads and normalizes chunks of the test dataset
        filter_variants --input pre-filtered.vcf --test_dataset ~{test_dataset} \
            --artifact_model ~{artifact_model} \
            --contigs_table ~{contigs_table} \
            --output permutect-filtered.vcf \
            --tensorboard_dir tensorboard \
            --batch_size ~{batch_size} --num_workers ~{num_dataloader_workers} --num_processes ~{num_cpus} \
            ~{" --num_spectrum_iterations " + num_spectrum_iterations} \
            ~{" --spectrum_learning_rate " + spectrum_learning_rate} \
            ~{" --spectrum_batch_size " + spectrum_batch_size} \
            ~{" --germline_model " + germline_model} \
            ~{" --read_orientation_model " + read_orientation_model} \
            --genomic_span $genomic_span ~{permutect_filtering_extra_args}

        tar cvf tensorboard.tar tensorboard/

        # compress
        bcftools view permutect-filtered.vcf -Oz -o permutect-filtered.vcf.gz

        bcftools index -t permutect-filtered.vcf.gz
    >>>

    runtime {
        docker: permutect_docker
        bootDiskSizeGb: 4  # Cromwell on GCP Batch adds 30 GB for the OS and the docker image (7.2 GB unpacked)
        memory: machine_mem + " MB"
        disks: "local-disk " + select_first([disk_space, default_disk_space]) + " SSD"
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

task SplitMultiallelics {
    input {
        File input_vcf
        File input_vcf_idx
        # the reference FASTA, then its .fai index, its .dict dictionary and, if it's bgzipped, its .gzi block index
        Array[File] reference
        String gatk_docker
        Int? preemptible
        Int max_retries = 0
        Int? disk_space
        Int cpu = 1
        Int mem = 4
    }

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = mem * 1000
    Int command_mem = machine_mem - 500
    # the reference, and the VCF, its split copy and the output
    Int default_disk_space = ceil(size(reference, "GB") + 3 * size(input_vcf, "GB")) + 10

    command <<<
        set -euo pipefail

        bcftools norm -m -any -f ~{reference[0]} ~{input_vcf} > split.vcf

        gatk --java-options "-Xmx~{command_mem}m" IndexFeatureFile -I split.vcf

        gatk --java-options "-Xmx~{command_mem}m" SelectVariants -V split.vcf -O output.vcf --lenient \
            -DGA DP -DGA AF -DGA FAD -DGA SB \
            -DA AS_FilterStatus -DA AS_SB_TABLE -DA ECNT -DA GERMQ -DA MBQ -DA MFRL -DA MMQ -DA MPOS
    >>>

    runtime {
        docker: gatk_docker
        bootDiskSizeGb: 4  # Cromwell on GCP Batch adds 30 GB for the OS and the docker image (7.2 GB unpacked)
        memory: machine_mem + " MB"
        disks: "local-disk " + select_first([disk_space, default_disk_space]) + " SSD"
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
        # files of genomic intervals, whose union the task evaluates, or none for the whole genome
        Array[File] intervals
    	File? masks
    	File truth_vcf
    	File truth_vcf_idx
    	File eval_vcf
    	File eval_vcf_idx
        String? concordance_header
        # split each multi-nucleotide variant (MNV) of eval_vcf into its constituent SNVs, which needs bcftools
        Boolean split_mnvs = false

    	# runtime
    	String gatk_docker = "us.gcr.io/broad-gatk/gatk"
    	Int preemptible = 0
        Int? disk_space
	}

    # The VCFs and Concordance's outputs and the records extracted from them, about twice the VCFs' uncompressed size, which
    # is up to about five times a bgzipped VCF's
    Int default_disk_space = ceil(10 * (size(truth_vcf, "GB") + size(eval_vcf, "GB"))) + 10

    command <<<
        eval_vcf="~{eval_vcf}"
        # decompose each MNV into its constituent SNVs, each of which keeps the MNV's FILTER, INFO and FORMAT fields.
        # Multiallelic records are split first, because atomizing a multiallelic record with an MNV allele gives star
        # alleles.  Other variants are unchanged.  The rest of this script runs without set -e, so stop here explicitly.
        if [ "~{split_mnvs}" = "true" ]; then
            bcftools norm -m -any --atomize "$eval_vcf" -Oz -o mnvs-split.vcf.gz || exit 1
            eval_vcf=mnvs-split.vcf.gz
        fi

        # this shouldn't be necessary but the Terra hacks on top of Terra hacks to deal with all the bugs recently
        # have reached a breaking point
        gatk IndexFeatureFile -I "$eval_vcf"

        gatk --java-options "-Xmx2g" Concordance \
            ~{if length(intervals) > 0 then "-L " else ""}~{sep=" -L " intervals} \
            ~{"-XL " + masks} \
            -truth ~{truth_vcf} -eval "$eval_vcf" \
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
        bootDiskSizeGb: 4  # Cromwell on GCP Batch adds 30 GB for the OS and the docker image (7.2 GB unpacked)
        docker: "${gatk_docker}"
        disks: "local-disk " + select_first([disk_space, default_disk_space]) + " HDD"
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
        # memory in GB, which overrides the estimate below
        Int? mem
    }

    # numba parallelizes the E-step: in November 2026 a genome took 17% less time with 8 vCPUs than with 4, at about 40%
    # more cost, and 16 vCPUs were no faster than 8
    Int num_cpus = select_first([cpu, 4])

    # Without mem, an estimate of the peak memory from measurements in November 2026, plus 1.5 GB of margin: about 2 GB
    # plus 0.8-1 GB per million sites of the larger sample, whose pileups table holds 32 (hg38) to 41 (b37) million
    # sites per GB.  The runtime value is in MB.
    Float tumor_gb = size(tumor_pileups, "GB")
    Float normal_gb = size(normal_pileups, "GB")
    Float peak_gb = 2.0 + 33 * (if tumor_gb > normal_gb then tumor_gb else normal_gb)
    Int machine_mem = if defined(mem) then mem * 1000 else ceil((peak_gb + 1.5) * 1000)
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
        bootDiskSizeGb: 4  # Cromwell on GCP Batch adds 30 GB for the OS and the docker image (7.2 GB unpacked)
        memory: machine_mem + " MB"
        disks: "local-disk " + disk_space + " SSD"
        preemptible: select_first([preemptible, 0])
        maxRetries: select_first([max_retries, 0])
        cpu: num_cpus
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

# learn the tumor's read orientation artifact model from the F1R2 counts of Mutect2's scatters, given segment_pileups'
# GermlineModel if there is one
task LearnReadOrientationModel {
    input {
        Array[File]+ f1r2_counts
        File? germline_model

        String permutect_docker
        Int? preemptible
        Int? max_retries
        Int? cpu
        # memory in GB, which overrides the estimate below
        Int? mem
    }

    # torch uses a thread per core: in November 2026 4 vCPUs (2 cores) were 1.6-1.8 times as fast as 2, at the same cost
    Int num_cpus = select_first([cpu, 4])

    # Without mem, an estimate of the peak memory from measurements in November 2026, plus 1.5 GB of margin: about 1.35 GB
    # plus, per GB of F1R2 counts, 81 GB with a germline model, whose minor allele fraction mixtures multiply the distinct
    # sites, or 32 GB without.  An FFPE genome's 80 MB of counts peaked at 7.8 GB with its germline model.
    Float peak_gb = 1.35 + (if defined(germline_model) then 81 else 32) * size(f1r2_counts, "GB")
    # in MB as Cromwell counts them, 2^20 bytes, so a little more than the estimate
    Int machine_mem = if defined(mem) then mem * 1000 else ceil((peak_gb + 1.5) * 1000)
    # the counts, which the tool reads without extracting them, and the germline model
    Int disk_space = ceil(size(f1r2_counts, "GB") + size(germline_model, "GB")) + 10

    command <<<
        set -euo pipefail
        learn_read_orientation_model --input ~{sep=" " f1r2_counts} ~{"--germline_model " + germline_model} \
            --output read-orientation-model.pt
    >>>

    runtime {
        docker: permutect_docker
        bootDiskSizeGb: 4  # Cromwell on GCP Batch adds 30 GB for the OS and the docker image (7.2 GB unpacked)
        memory: machine_mem + " MB"
        disks: "local-disk " + disk_space + " SSD"
        preemptible: select_first([preemptible, 0])
        maxRetries: select_first([max_retries, 0])
        cpu: num_cpus
    }

    output {
        File read_orientation_model = "read-orientation-model.pt"
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
        bootDiskSizeGb: 4  # Cromwell on GCP Batch adds 30 GB for the OS and the docker image (7.2 GB unpacked)
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
        bootDiskSizeGb: 4  # Cromwell on GCP Batch adds 30 GB for the OS and the docker image (7.2 GB unpacked)
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
