version 1.0

workflow ComparePermutect {
    input {
		File? intervals
		File? masks
    	File permutect_vcf
    	File permutect_vcf_idx
        File other_vcf
    	File other_vcf_idx
    	File truth_vcf
    	File truth_vcf_idx

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

    File? intervals_hack = if select_first([intervals, ""]) == "" then NONE else intervals
    File? masks_hack = if select_first([masks, ""]) == "" then NONE else masks
    File? gatk_override_hack = if select_first([gatk_override, ""]) == "" then NONE else gatk_override

    call Concordance as PermutectConcordance {
        input:
            intervals = intervals_hack,
            masks = masks_hack,
            truth_vcf = truth_vcf,
            truth_vcf_idx = truth_vcf_idx,
            eval_vcf = permutect_vcf,
            eval_vcf_idx = permutect_vcf_idx,
            gatk_override = gatk_override_hack,
            gatk_docker = gatk_docker
    }

    call Concordance as OtherConcordance {
        input:
            intervals = intervals_hack,
            masks = masks_hack,
            truth_vcf = truth_vcf,
            truth_vcf_idx = truth_vcf_idx,
            eval_vcf = other_vcf,
            eval_vcf_idx = other_vcf_idx,
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
    	File? intervals
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
            ~{"-L " + intervals} \
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