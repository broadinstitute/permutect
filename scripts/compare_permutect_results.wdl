version 1.0

workflow ComparePermutect {
    input {
        # Files of the genomic intervals to compare, such as .interval_list or .bed files, or [] for the whole genome.
        # Concordance gives GATK each file with its own -L, and GATK takes the union of their intervals.  In a Terra
        # configuration, an attribute expression such as workspace.intervals gives a one-element array.  Elements that
        # are "" are ignored, since Terra may turn unset data table values into "" (see below).  (Earlier versions
        # took a single, optional file.)
        Array[File] intervals
		File? masks
        # Permutect's calls, another tool's calls and the truth, each an array of the VCF and, optionally, its index:
        # [vcf] or [vcf, idx].  Concordance finds each index only by name beside its VCF: the VCF's path plus .tbi for a
        # bgzipped VCF (.vcf.gz or .vcf.bgz), or plus .idx for a plain .vcf.  Given [vcf], the workflow derives that
        # path and passes the index along with the VCF, so [vcf] suffices on every backend, and an index given as the
        # second element must be that same file.  Elements that are "" are ignored, since Terra may turn unset data
        # table values into "" (see below).  (Earlier versions took each index as a separate input, such as
        # permutect_vcf_idx.)
        Array[File]+ permutect_vcf
        Array[File]+ other_vcf
        Array[File]+ truth_vcf

    	# runtime
    	String gatk_docker
    	File? gatk_override
    }

    # WDL version 1.0 does not have an empty Optional literal.  Such a literal is very useful because Terra has a bug
    # where whenever a data table is updated, empty values silently and invisibly get converted to empty strings "".
    # Thus it is useful to recognize empty strings and declare empty Optionals.  A declaration inside a conditional is
    # an Optional outside of it, so NONE is always an empty Optional.
    if (false) {
        File NONE = ""
    }

    # The input arrays without their "" elements, which Terra may have made from unset data table values.  WDL has no
    # filter function, so a scatter without calls keeps the other elements, and select_all collects them.  Keep
    # given_arrays and the indices into nonblank below in the same order.
    Array[Array[File]] given_arrays = [intervals, permutect_vcf, other_vcf, truth_vcf]
    scatter (given_array in given_arrays) {
        scatter (element in given_array) {
            if (element != "") {
                File nonblank_element = element
            }
        }
        Array[File] nonblank = select_all(nonblank_element)
    }
    Array[File] given_intervals = nonblank[0]

    # Each VCF with its index: the array as given if it has the index, and otherwise the VCF and the index where GATK
    # looks for it, the VCF's path plus .tbi for a bgzipped VCF (.vcf.gz or .vcf.bgz) or .idx for a plain .vcf.  Keep
    # the indices into vcf_with_index below in the order of the VCFs.
    scatter (given_vcf in [nonblank[1], nonblank[2], nonblank[3]]) {
        String vcf_path = if length(given_vcf) > 0 then "~{given_vcf[0]}" else ""
        Array[File] vcf_with_index = if length(given_vcf) == 1
            then [vcf_path, vcf_path + (if sub(vcf_path, "\\.b?gz$", "") != vcf_path then ".tbi" else ".idx")]
            else given_vcf
    }
    Array[File] permutect_vcf_with_index = vcf_with_index[0]
    Array[File] other_vcf_with_index = vcf_with_index[1]
    Array[File] truth_vcf_with_index = vcf_with_index[2]

    File? masks_hack = if select_first([masks, ""]) == "" then NONE else masks
    File? gatk_override_hack = if select_first([gatk_override, ""]) == "" then NONE else gatk_override

    call Concordance as PermutectConcordance {
        input:
            intervals = given_intervals,
            masks = masks_hack,
            truth_vcf = truth_vcf_with_index[0],
            truth_vcf_idx = truth_vcf_with_index[1],
            eval_vcf = permutect_vcf_with_index[0],
            eval_vcf_idx = permutect_vcf_with_index[1],
            gatk_override = gatk_override_hack,
            gatk_docker = gatk_docker
    }

    call Concordance as OtherConcordance {
        input:
            intervals = given_intervals,
            masks = masks_hack,
            truth_vcf = truth_vcf_with_index[0],
            truth_vcf_idx = truth_vcf_with_index[1],
            eval_vcf = other_vcf_with_index[0],
            eval_vcf_idx = other_vcf_with_index[1],
            gatk_override = gatk_override_hack,
            gatk_docker = gatk_docker
    }

    call Compare {
        input:
            permutect_ffn = PermutectConcordance.ffn,
            permutect_ffn_idx = PermutectConcordance.ffn_idx,
            other_ffn = OtherConcordance.ffn,
            other_ffn_idx = OtherConcordance.ffn_idx,

            permutect_fp = PermutectConcordance.fp,
            permutect_fp_idx = PermutectConcordance.fp_idx,
            other_fp = OtherConcordance.fp,
            other_fp_idx = OtherConcordance.fp_idx,

            permutect_summary = PermutectConcordance.summary,
            other_summary = OtherConcordance.summary,

            gatk_override = gatk_override_hack,
            gatk_docker = gatk_docker
    }


    output {
        File ffn_created_by_permutect = Compare.ffn_created_by_permutect
        File ffn_created_by_permutect_idx = Compare.ffn_created_by_permutect_idx
        File ffn_fixed_by_permutect = Compare.ffn_fixed_by_permutect
        File ffn_fixed_by_permutect_idx = Compare.ffn_fixed_by_permutect_idx

        File fp_created_by_permutect = Compare.fp_created_by_permutect
        File fp_created_by_permutect_idx = Compare.fp_created_by_permutect_idx
        File fp_fixed_by_permutect = Compare.fp_fixed_by_permutect
        File fp_fixed_by_permutect_idx = Compare.fp_fixed_by_permutect_idx

        File summary = Compare.summary
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

    	File? gatk_override

    	# runtime
    	String gatk_docker
    	Int preemptible = 2
	}

    command <<<
        export GATK_LOCAL_JAR=~{default="/root/gatk.jar" gatk_override}

        gatk --java-options "-Xmx2g" Concordance \
            ~{if length(intervals) > 0 then "-L " else ""}~{sep=" -L " intervals} \
            ~{"-XL " + masks} \
            -truth ~{truth_vcf} -eval ~{eval_vcf} \
            -tpfn "tpfn.vcf" \
            -tpfp "tpfp.vcf" \
            -ftnfn "ftnfn.vcf" \
            -filter-analysis "filter-analysis.txt" \
            -summary "summary.txt"

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

task Compare {
    input {
    	File permutect_ffn
        File permutect_ffn_idx
        File other_ffn
        File other_ffn_idx

        File permutect_fp
        File permutect_fp_idx
        File other_fp
        File other_fp_idx

        File permutect_summary
        File other_summary

    	File? gatk_override

    	# runtime
    	String gatk_docker
    	Int preemptible = 2
	}

    command <<<
        set -e
        export GATK_LOCAL_JAR=~{default="/root/gatk.jar" gatk_override}

        # the records of the first VCF that aren't in the second
        select_discordant() {
            gatk --java-options "-Xmx2g" SelectVariants -V $1 -disc $2 -O $3
        }
        select_discordant ~{permutect_ffn} ~{other_ffn} ffn_created_by_permutect.vcf
        select_discordant ~{other_ffn} ~{permutect_ffn} ffn_fixed_by_permutect.vcf
        select_discordant ~{permutect_fp} ~{other_fp} fp_created_by_permutect.vcf
        select_discordant ~{other_fp} ~{permutect_fp} fp_fixed_by_permutect.vcf

        touch summary.txt
        echo "PERMUTECT SUMMARY" >> summary.txt
        cat ~{permutect_summary} >> summary.txt
        printf "\n" >> summary.txt
        echo "OTHER TOOL SUMMARY" >> summary.txt
        cat ~{other_summary} >> summary.txt
        printf "\n" >> summary.txt

        # append a label and the number of records in a VCF to the summary
        count_records() {
            printf '%s: ' "$1" >> summary.txt
            grep -v '#' $2 | wc -l >> summary.txt
        }
        count_records "Permutect FFN count" ~{permutect_ffn}
        count_records "Other tool FFN count" ~{other_ffn}
        count_records "Permutect FP count" ~{permutect_fp}
        count_records "Other tool FP count" ~{other_fp}
        count_records "Permutect created FFN count" ffn_created_by_permutect.vcf
        count_records "Permutect fixed FFN count" ffn_fixed_by_permutect.vcf
        count_records "Permutect created FP count" fp_created_by_permutect.vcf
        count_records "Permutect fixed FP count" fp_fixed_by_permutect.vcf
    >>>

    runtime {
        memory: "5 GB"
        bootDiskSizeGb: 12
        docker: "${gatk_docker}"
        disks: "local-disk " + 100 + " HDD"
        preemptible: select_first([preemptible, 2])
    }

    output {
        File ffn_created_by_permutect = "ffn_created_by_permutect.vcf"
        File ffn_created_by_permutect_idx = "ffn_created_by_permutect.vcf.idx"
        File ffn_fixed_by_permutect = "ffn_fixed_by_permutect.vcf"
        File ffn_fixed_by_permutect_idx = "ffn_fixed_by_permutect.vcf.idx"

        File fp_created_by_permutect = "fp_created_by_permutect.vcf"
        File fp_created_by_permutect_idx = "fp_created_by_permutect.vcf.idx"
        File fp_fixed_by_permutect = "fp_fixed_by_permutect.vcf"
        File fp_fixed_by_permutect_idx = "fp_fixed_by_permutect.vcf.idx"

        File summary = "summary.txt"
    }
}