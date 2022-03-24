
dn='/projects/b1131/saya/amppd_v2/wgs/samplesets'
fn=f'{dn}/amppd_v2_samples_all.txt'

## Read in whole sample set 
with open(fn, 'r') as f:
    lines = f.readlines()

# Remove newlines at the end of each item 
sampleids=[]
for line in lines:
    sampleids.append(line.strip())

## Create sample subsets that are size <=250 
for i in range(len(sampleids)//250+1): # each subset has 250 samples in it 
    subset=sampleids[i*250:(i+1)*250]
    with open(f'{dn}/amppd_v2_sampleset_{i}.txt', 'w') as f:
        for sampleid in subset:
            f.write('%s\n' % sampleid)

