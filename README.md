# Overview of Permutect
[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg)](https://anaconda.org/bioconda/permutect)
[![Anaconda-Server Badge](https://anaconda.org/bioconda/permutect/badges/version.svg)](https://anaconda.org/bioconda/permutect)
[![Anaconda-Server Badge](https://anaconda.org/bioconda/permutect/badges/downloads.svg)](https://anaconda.org/bioconda/permutect)

Permutect is a pipeline for calling and filtering variants from genomic sequencing data.  It takes a BAM or CRAM file (and, optionally, a matched normal sample BAM or CRAM) and outputs a filtered VCF.  Roughly speaking it comprises two parts: 1) permissive, unfiltered variant calling with Mutect2 and 2) a new deep learning-based filtering model.

The greatest advantage of Permutect over the Mutect2/FilterMutectCalls pipeline is that the latter performs poorly without a "panel of normals", which is a blacklist VCF of common errors compiled from a very large number of samples.  Since the typical user does not have sufficient data (100+ WGS samples) the best practice is to use one of the Broad Institute's publicly-available panels and hope for the best.  While this usually works for short read Illumina sequencing it fails for novel sequencing platforms and library preparation, non-human data, highly-variable parts of the genome etc.  In contrast, the Permutect model is so efficient that it can be trained on a single WGS sample with no labeled truth data without overfitting.

In addition to working on a variety of sequencing platforms Permutect is designed for both somatic and germline variant calling.  Crucially, a trained Permutect model is specific to a sequencing technology but not to any particular biological scenario.  Thus it is possible (in fact, recommended) to train the model on a germline sample and then deploy it for somatic calling or a different species.  This works due to a division of responsibility between a black box deep learning model that learns technical error modes and a probabilistic model that learns biological features such as somatic mutation rates and subclonal cell fractions.

Unlike the deep learning artifact model, the probabilistic model of biology does not have its own workflow below.  This is because it is sample-specific and rather than being trained ahead of time its parameters are learned during variant calling.

We strongly recommend running the pipeline from the [Permutect Terra workspace](https://app.terra.bio/#workspaces/broad-firecloud-dsde/Permutect).

# Installation
Permutect is available as a [Bioconda](https://bioconda.github.io/) package. We recommend installing it into an isolated environment:

```bash
conda create -n permutect -c conda-forge -c bioconda permutect
conda activate permutect
```

Or, to install into an existing environment:

```bash
conda install -c conda-forge -c bioconda permutect
```

This installs the command line tools (`pre_permutect_filtering`, `filter_variants`, `train_artifact_model`, `preprocess_dataset`, `edit_dataset`) along with all dependencies.

# Pipeline
Here we described the workflows in the Terra workspace, most of which correspond to a WDL script in the /scripts directory of this repository and a command line tool in the /tools directory.

## scripts/make_training_dataset.wdl
This runs the Mutect2 WDL as a subworkflow, invoking a mode that generates a Permutect training data file as a side effect.  Most of this workflow's inputs are straightforward things like a BAM/CRAM file, a GATK docker, reference files, a genomic intervals file, and a gnomAD VCF.  (We reiterate that requiring gnomAD for training does not restrict variant calling to human data; nor does the reference used for generating training data need to match the reference used for variant calling).  Optionally, one may give a training truth VCF of known germline variants for the BAM/CRAM sample.  Without this, Permutect generates weak truth labels based on the observation that artifacts generally occur at lower allele fractions than germline variants.  This weak labeling scheme works well, but if possible it is best to train on one of the NIST Genome in a Bottle samples.

After generating Permutect training data in a plain text format, the workflow calls tools/preprocess_dataset.py to create a binary .tar.gz file for use in the model training workflow.

## scripts/train_artifact_model.wdl
This workflow takes the .tar.gz output from the training dataset workflow and produces a Permutect artifact model in PyTorch .pt format.  Most of the configuration comprises hyperparameters of the model architecture and we recommend copying these from the Terra workspace.  Feel free to contact David Benjamin (davidben@broadinstitute.org) regarding details.

## scripts/permutect.wdl, GATK Mutect2 and tools/filter_variants.py)
This workflow runs Mutect2 in a mode that generates both an unfiltered VCF and a plain text Permutect dataset.  The data format is the same as the training dataset workflow but the dataset includes all unfiltered variant calls.  It then uses the artifact model from the training workflow to compute artifact likelihoods, fits the sample-specific biology model, and generates a filtered VCF.

# Building the Permutect Docker image
First of all, I ought to say right now that all this engineering stuff is total voodoo to me copied from other people's example.  I have done my best to make it professional but I really have no idea what I'm doing.  Here is an overview of all the magical files in this repo that somehow work to make a docker image.

## pyproject.toml
This is a list of required python packages along with the versions that will work.  It also tells pip where to look for command line programs and what names to give them.  Thus, for example, the main() method of permutect.tools.train_artifact_model becomes the command line program train_artifact_model.  If you want to build Permutect locally, instead of within a Docker image, you can run the terminal command "pip install ." from the root of this repo.

## pixi.toml
For development, this defines [pixi](https://pixi.sh) environments with locked dependencies and tasks such as "pixi run test-unit", which runs the tests that CI runs.

## Dockerfile
This installs bcftools and GATK, then uses pyproject.toml to install Permutect in a Docker image with the correct dependencies.  Pre-built public Docker images are listed under each Permutect release.
