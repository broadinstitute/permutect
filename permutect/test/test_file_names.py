import os

# this ought to be .../permutect/permutect/test/, where the first "permutect" is the github repo folder
test_dir = os.path.dirname(__file__)
integration_test_dir = os.path.abspath(os.path.join(test_dir, "..", "..", "integration-tests"))

SMALL_PLAIN_TEXT_DATA = os.path.join(integration_test_dir, "hiseqx-NA12878-8000-data-plain-text.dataset")
PREPROCESSED_DATA = os.path.join(integration_test_dir, "preprocessed-dataset.tar")
SMALL_ARTIFACT_MODEL = os.path.join(integration_test_dir, "small_artifact_model.pt")

HG19_CONTIGS_TABLE = os.path.join(integration_test_dir, "contigs.table")
HG38_CONTIGS_TABLE = os.path.join(integration_test_dir, "hg38-contigs.table")
DREAM_1_CHR20_PLAIN_TEXT_DATA = os.path.join(integration_test_dir, "dream-1-chr-20-plain-text.dataset")
HG19_CHR20_REFERENCE = os.path.join(integration_test_dir, "Homo_sapiens_assembly19_chr20.fasta")
HG38_CHR20_REFERENCE = os.path.join(integration_test_dir, "Homo_sapiens_assembly38_chr20.fasta")
HG38_CHR20_REFERENCE_INDEX = HG38_CHR20_REFERENCE + ".fai"
HG38_CHR20_REFERENCE_DICT = os.path.join(integration_test_dir, "Homo_sapiens_assembly38_chr20.dict")
SEQC2_WES_PLAIN_TEXT_DATA = os.path.join(integration_test_dir, "seqc2-wes-chr20-plain-text.dataset")
# chr20 of the GetPileupSummaries tables of SEQC2's WES_IL_T_1 and WES_IL_N_1, with gnomAD allele frequencies
SEQC2_WES_TUMOR_PILEUPS = os.path.join(integration_test_dir, "seqc2-wes-chr20-tumor.pileups.table")
SEQC2_WES_NORMAL_PILEUPS = os.path.join(integration_test_dir, "seqc2-wes-chr20-normal.pileups.table")

ARTIFACT_MODEL_V_0_4_0 = os.path.join(integration_test_dir, "artifact-model-v0.4.0.pt")
PERMUTECT_V0_9_2_ARTIFACT_MODEL = os.path.join(integration_test_dir, "permutect-v0.9.2-artifact-model.pt")

MUTECT2_CHR20_FILTERED_VCF = os.path.join(integration_test_dir, "mutect2_chr20.vcf")
MUTECT2_SEQC2_WES_VCF = os.path.join(integration_test_dir, "mutect2_seqc2_wes_chr20.vcf")
MUTECT2_SEQC2_WES_VCF_INDEX = MUTECT2_SEQC2_WES_VCF + ".idx"
MUTECT2_SEQC2_WES_STATS = MUTECT2_SEQC2_WES_VCF + ".stats"

# for testing SNV-dependent priors, this VCF only has true _ -> T substitution.
MUTECT2_CHR20_FILTERED_VCF_ONLY_T_SNVS = os.path.join(integration_test_dir, "mutect2_chr20_only_T_snvs.vcf")

# GATK's CalculateContamination test data: GetPileupSummaries tables of NA12891 reads spiked into NA12892 and of small
# gene panels, and in a subdirectory the segments that GATK's CalculateContamination found in them
CONTAMINATION_TEST_DIR = os.path.join(integration_test_dir, "calculate-contamination")
GATK_CONTAMINATION_SEGMENTS_DIR = os.path.join(CONTAMINATION_TEST_DIR, "gatk-segments")

# F1R2 counts from GATK CollectF1R2Counts on GATK test data, in scatters, and GATK LearnReadOrientationModel's priors
# learned from the scatters.  DREAM challenge synthetic tumor 4 (sample
# synthetic.challenge.set4.tumour), b37 chr20 in two halves: every context has data.  The 3 scatters of the small read
# orientation test bam of GATK's LearnReadOrientationModelIntegrationTest: about half the contexts have too few sites
# and keep GATK's flat prior.
READ_ORIENTATION_TEST_DIR = os.path.join(integration_test_dir, "read-orientation")
DREAM_TUMOR_4_F1R2_SCATTERS = [
    os.path.join(READ_ORIENTATION_TEST_DIR, f"dream-tumor-4.f1r2-scatter-{n}.tar.gz") for n in range(2)
]
DREAM_TUMOR_4_GATK_ORIENTATION_PRIORS = os.path.join(
    READ_ORIENTATION_TEST_DIR, "dream-tumor-4.gatk-orientation-priors.tar.gz"
)
HAPMAP_F1R2_SCATTERS = [
    os.path.join(READ_ORIENTATION_TEST_DIR, f"hapmap-read-orientation.f1r2-scatter-{n}.tar.gz") for n in range(3)
]
HAPMAP_GATK_ORIENTATION_PRIORS = os.path.join(
    READ_ORIENTATION_TEST_DIR, "hapmap-read-orientation.gatk-orientation-priors.tar.gz"
)
