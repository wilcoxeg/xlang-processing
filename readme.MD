
## To Set up the Environment

`$ conda env create -f environment.yml`
`$ source activate xlang-processing`

Activate the environment and install the appropriate ML libraries

`$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -c pytorch`
`$ pip install transformers`

You may have to pip install a couple of other dependencies

## Running the Code

`$ sh scripts/pull_data.sh`
`$ python get_predictors.py`

I was having a hard time getting the model downloaded in the script. So just download manually in python:

`from transformers import GPT2LMHeadModel, GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/mGPT")
model = GPT2LMHeadModel.from_pretrained("sberbank-ai/mGPT")`

You can run the code on a SLURM cluster with `euler.sh`

## Merging the Data

* Need to have `joint_data_trimmed.rda` downloaded in the `/data/eye_tracking_data/` directory
* Run the script `/scripts/MECO_L1_fix.Rmd` to fix the various bugs in the Meco data (specific bugs are documented in the script). This produces `joint_data.rda`.
* Run `/scripts/separate_langs.Rmd` to separate out the individual langs. This saves the language data in the `/data/rt_data_l1/` directory
* Run `/scripts/clean_data.Rmd`. This merges the data with the model outputs and saves the by-langauge merged files in the `/data/merged_data/` directory
	- Note: This script has been changed since the original published version of the paper so that `freq` is now the negative log2 of frequency, i.e., it is now unigram surprisal.


## Analysis 

`clean_data.Rmd` takes the outputs from above and cleans them up for analysis.

`analysis.Rmd` does all the plotting and various statistical analyses.