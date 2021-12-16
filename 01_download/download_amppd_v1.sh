
# Whole genome sequencing data -- includes vcf.gz file and vcf.gz.tbi file
gsutil -u project_id cp gs://amp-pd-genomics/releases/2019_v1release_1015/wgs/gatk/vcf/chr<num>.vcf.gz* /path/to/download/dir/ > gsutil_download_amppd_wgs.out

# sequencing data
gsutil -u project_id cp -r gs://amp-pd-transcriptomics/releases/2019_v1release_1015/rnaseq/sequencing /path/to/download/directory > gsutil_download_amppd_sequencing.out

# for picard data
gsutil -u project_id cp -r gs://amp-pd-transcriptomics/releases/2019_v1release_1015/rnaseq/picard /path/to/download/directory > gsutil_download_picard.out

# for subread data
gsutil -u project_id cp -r gs://amp-pd-transcriptomics/releases/2019_v1release_1015/rnaseq/subread /path/to/download/directory > gsutil_download_subread.out

# for reports data
gsutil -u project_id cp -r gs://amp-pd-transcriptomics/releases/2019_v1release_1015/rnaseq/salmon /path/to/download/directory > gsutil_download_reports.out
