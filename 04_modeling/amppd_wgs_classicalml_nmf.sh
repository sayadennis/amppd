#!/bin/bash
#SBATCH -A b1042
#SBATCH -p genomicslong
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mail-user=sayarenedennis@northwestern.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name="pd_ml_nmf"
#SBATCH --output=pd_project/out/amppd_wgs_classicalml_nmf.out

. ~/anaconda3/etc/profile.d/conda.sh
conda activate pdenv

inputdir='/projects/b1131/saya/amppd_v2/wgs/06_cleaned_input_target'
labeldir='/projects/b1131/saya/amppd_v2/wgs/06_cleaned_input_target'
outdir='/home/srd6051/pd_project/model_performance/v2'
ixdir='/projects/b1131/saya/amppd_v2/wgs/indices'

############################
#### Main modeling code ####
############################

# python classical_ml/ClassicalML/run_classical_ml.py \
#     --input $inputdir/vc_mx.csv \
#     --label $labeldir/label.csv \
#     --outfn $outdir/amppd_v2_classicalml_nmf.csv \
#     --indexdir $ixdir \
#     --scoring roc_auc \
#     --nmf 500 \
#     --savemodel "true"
# #

#############################################
#### modeling with batch-corrected input ####
#############################################

inputdir='/projects/b1131/saya/amppd_v2/wgs/07_batch_rm'

python classical_ml/ClassicalML/run_classical_ml.py \
    --input $inputdir/vc_mx_bcor_pca_reverse_tf.csv \
    --label $labeldir/label.csv \
    --outfn $outdir/amppd_v2_classicalml_nmf_bcor_pca_reverse_tf.csv \
    --indexdir $ixdir \
    --scoring roc_auc \
    --nmf 500 \
    --savemodel "true"
#
