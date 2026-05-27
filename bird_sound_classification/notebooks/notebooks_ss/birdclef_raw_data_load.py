
import pandas as pd
import pyprojroot
from pyprojroot.here import here




def load_data():
    """ This method reads in the raw data file and retruns the cleaned raw data"""
    # reading in the data
    base_path = pyprojroot.find_root(pyprojroot.has_dir("notebooks"))
    data = pd.read_csv(here("data/raw/train.csv"))

    # mapping primary - scientific - common labels
    mapped = data[["primary_label", "scientific_name", "common_name"]]
    mapped = mapped.drop_duplicates()

    #species reference to obtain class label
    spec_dat = pd.read_csv(here("data/raw/taxonomy.csv"))
    # taking out the taxon_id since it's duplicated
    spec_dat = spec_dat.drop("inat_taxon_id", axis = 1)

    # merging class_name
    full_data = pd.merge(data, spec_dat, on = ["primary_label", "scientific_name", "common_name"], how = "left")
    full_data.head()


    full_data = full_data.replace(to_replace=r"\[''\]", value=pd.NA, regex=True)


    return full_data

