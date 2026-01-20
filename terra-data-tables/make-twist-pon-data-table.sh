#!/bin/bash

source=gs://fc-f3bda9d5-024a-4524-8123-51dd01eb7cb3/twist-pon/
gsutil ls $source | grep .bam$ > bams.tmp

touch sample.tsv
printf "entity:sample_id\tbai\tbam\tparticipant\n" >> sample.tsv

# intervals=______.interval_list
participant=twist

while read bam; do
    bai=${bam}.bai
    without_folders="${bam##*/}"	# this is the path after the last slash
    sample="${without_folders%.*}"	# also remove the .cram extension
    # echo "Putting bam ${bam} into the data table"
    # echo "Sample name is ${sample}"

    printf "${sample}\t${bai}\t${bam}\t${participant}\n" >> sample.tsv
done < bams.tmp

rm bams.tmp

echo "DONE.  Data table written to sample.tsv"
