# Run once

from preprocess.split_by_date import splitByDate
from preprocess.gen_itemdict import gen_itemdict
from preprocess.gen_feats import gen_uci_feats
from preprocess.gen_ic_ind_feats import gen_ic_feats

# Split the data by date
splitByDate(part=1)

# Generate itemdict
gen_itemdict(part=1)

# Extract uci features
gen_uci_feats(part=1)

# Extract item/cat independent features
gen_ic_feats(part=1)

# Done!
# Then you can run main_single_model.py step by step to tune the model.
