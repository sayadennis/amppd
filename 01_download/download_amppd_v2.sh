#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomics
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 48:00:00
#SBATCH --mail-user=sayarenedennis@northwestern.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=amppd_v2_genomics
#SBATCH --output=pd_project/amppd_outputs/210728_amppd_v2_genomics.out

## WGS data 
gcp_wgs='gs://amp-pd-genomics/releases/2021_v2-5release_0510/wgs'
quest_wgs='/projects/b1131/saya/amppd_v2/wgs'

gsutil -u alien-striker-275013 cp -r $gcp_wgs/wgs_samples.csv $quest_wgs/
gsutil -u alien-striker-275013 cp -r $gcp_wgs/wgs_qc_flags.csv $quest_wgs/
gsutil -u alien-striker-275013 cp -r $gcp_wgs/gatk $quest_wgs/ # check whether there are anything other than vcf/ and metrics/ in here
gsutil -u alien-striker-275013 cp -r $gcp_wgs/topmed $quest_wgs/ # this hasn't been downloaded yet 
gsutil -u alien-striker-275013 cp -r $gcp_wgs/plink $quest_wgs/ # this hasn't been downloaded yet 

## RNAseq data
gcp_rna='gs://amp-pd-transcriptomics/releases/2021_v2-5release_0510'
quest_rna='/projects/b1131/saya/amppd_v2/rnaseq'

gsutil -u alien-striker-275013 cp -r $gcp_rna/rnaseq/rna_seq_samples.csv $quest_rna/rnaseq/
gsutil -u alien-striker-275013 cp -r $gcp_rna/rnaseq/picard $quest_rna/rnaseq/
gsutil -u alien-striker-275013 cp -r $gcp_rna/rnaseq/plink $quest_rna/rnaseq/
gsutil -u alien-striker-275013 cp -r $gcp_rna/rnaseq/reports $quest_rna/rnaseq/
gsutil -u alien-striker-275013 cp -r $gcp_rna/rnaseq/salmon $quest_rna/rnaseq/
gsutil -u alien-striker-275013 cp -r $gcp_rna/rnaseq/sequencing $quest_rna/rnaseq/
gsutil -u alien-striker-275013 cp -r $gcp_rna/rnaseq/star $quest_rna/rnaseq/
gsutil -u alien-striker-275013 cp -r $gcp_rna/rnaseq/subread $quest_rna/rnaseq/

gsutil -u alien-striker-275013 cp -r $gcp_rna/rnaseq_pools/rna_pooled_sample_inventory.csv $quest_rna/rnaseq_pools/
gsutil -u alien-striker-275013 cp -r $gcp_rna/rnaseq_pools/rna_seq_pooled_samples.csv $quest_rna/rnaseq_pools/
gsutil -u alien-striker-275013 cp -r $gcp_rna/rnaseq_pools/picard $quest_rna/rnaseq_pools/
gsutil -u alien-striker-275013 cp -r $gcp_rna/rnaseq_pools/reports $quest_rna/rnaseq_pools/
gsutil -u alien-striker-275013 cp -r $gcp_rna/rnaseq_pools/salmon $quest_rna/rnaseq_pools/
gsutil -u alien-striker-275013 cp -r $gcp_rna/rnaseq_pools/star $quest_rna/rnaseq_pools/
gsutil -u alien-striker-275013 cp -r $gcp_rna/rnaseq_pools/subread $quest_rna/rnaseq_pools/

## Proteomics data
gcp_prot='gs://amp-pd-proteomics/releases/2021_v2-5release_0510/preview/targeted'
quest_prot='/projects/b1131/saya/amppd_v2/proteomics'

gsutil -u alien-striker-275013 cp -r $gcp_prot/rnaseq_pools/rna_pooled_sample_inventory.csv $quest_prot/

## Clinical data
gcp_clin='gs://amp-pd-data/releases/2021_v2-5release_0510'
quest_clin='/projects/b1131/saya/amppd_v2/clinical'

gsutil -u alien-striker-275013 cp -r $gcp_clin/clinical $quest_clin/
gsutil -u alien-striker-275013 cp -r $gcp_clin/amp_pd_case_control.csv $quest_clin/
gsutil -u alien-striker-275013 cp -r $gcp_clin/amp_pd_participant_wgs_duplicates.csv $quest_clin/
gsutil -u alien-striker-275013 cp -r $gcp_clin/amp_pd_participants.csv $quest_clin/
gsutil -u alien-striker-275013 cp -r $gcp_clin/rna_sample_inventory.csv $quest_clin/
gsutil -u alien-striker-275013 cp -r $gcp_clin/wgs_gatk_joint_genotyping_samples.csv $quest_clin/
gsutil -u alien-striker-275013 cp -r $gcp_clin/wgs_sample_inventory.csv $quest_clin/
