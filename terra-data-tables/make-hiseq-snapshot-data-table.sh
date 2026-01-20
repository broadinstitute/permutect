#!/bin/bash

gsutil ls gs://broad-dsp-david-benjamin/hiseqx-2018 | grep -v crai | grep cram > crams.tmp

touch sample.tsv
printf "entity:sample_id\tbai\tbam\tparticipant\ttraining_intervals\ttraining_truth\ttraining_truth_idx\n" >> sample.tsv

intervals=gs://gcp-public-data--broad-references/hg38/v0/wgs_calling_regions.hg38.interval_list
participant=hiseqx_2018_snapshot

while read cram; do
    crai=${cram}.crai
    without_folders="${cram##*/}"	# this is the path after the last slash
    sample="${without_folders%.*}"	# also remove the .cram extension
    # echo "Putting cram ${cram} into the data table"
    # echo "Sample name is ${sample}"
    

    # figure out the training truth
    # there are four different GIAB samples used here:
    # NA12878 (HG001)
    # NA24143 (HG004)
    # NA24149 (HG003)
    # NA24385 (HG002)
    # The cram names all start with NA_____, so we can check which truth VCF is appropriate

    if [[ "${sample}" == *"NA12878"* ]]; then
        truth_vcf=gs://broad-dsp-david-benjamin/giab/HG001_GRCh38_1_22_v4.2.1_benchmark.vcf.gz
    elif [[ "${sample}" == *"NA24143"* ]]; then
        truth_vcf=gs://broad-dsp-david-benjamin/giab/HG004_GRCh38_1_22_v4.2.1_benchmark.vcf.gz
    elif [[ "${sample}" == *"NA24149"* ]]; then
        truth_vcf=gs://broad-dsp-david-benjamin/giab/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz
    elif [[ "${sample}" == *"NA24385"* ]]; then
        truth_vcf=gs://broad-dsp-david-benjamin/giab/HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz
    else
        echo "DANGER! UNEXPECTED (possibly non-GIAB) sample ${sample} encountered."
    fi

    truth_vcf_idx=${truth_vcf}.tbi


    printf "${sample}\t${crai}\t${cram}\t${participant}\t${intervals}\t${truth_vcf}\t${truth_vcf_idx}\n" >> sample.tsv
done < crams.tmp

rm crams.tmp

echo "DONE.  Data table written to sample.tsv"
