
import os
import re
import torch
import pandas as pd
import numpy as np
from wordfreq import word_frequency
from string import punctuation

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

LANG_CODE = { "du":"nl", "en":"en", "fi":"fi", "ge":"de", "gr":"el",
			  "he":"he", "it":"it", "ko":"ko", "no":"nb", "ru":"ru",
			  "sp":"es", "tr":"tr" }
MODEL_TO_SPECIAL_TOKENS = {
	"mgpt" : "Ġ" }
MODEL_NAME = {
	"mgpt" : "sberbank-ai/mGPT"
}

#LANG_DATA_PATH = "../data/langs_l2"
LANG_DATA_PATH = "../data/rt_data_l1"

def ordered_string_join(x, j=''):
	s = sorted(x, key=lambda y: y[0])
	a, b = list(zip(*s))
	return j.join(b)

def get_predictions(sentence, model):
	model_name = MODEL_NAME[model]
	model = GPT2LMHeadModel.from_pretrained(model_name)
	tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

	sent_tokens = tokenizer.tokenize(sentence, add_special_tokens=True)
	indexed_tokens = tokenizer.convert_tokens_to_ids(sent_tokens)

	tokens_tensor = torch.tensor(indexed_tokens).unsqueeze(0)

	with torch.no_grad():
		probs = model(tokens_tensor)[0].softmax(dim=2).squeeze()

	return list(zip(sent_tokens, indexed_tokens, (None,) + probs.unbind()))


def get_surprisals(predictions):
	result = []
	for j, (word, word_idx, preds) in enumerate(predictions):
		if preds is None:
			surprisal = 0.0
		else:
			surprisal = -1 * np.log2(preds[word_idx].item())
		result.append( (j+1, word, surprisal) )
	return result

def get_entropies(predictions):
	result = []
	for j, (word, word_idx, preds) in enumerate(predictions):
		if preds is None:
			entropy = 0.0
		else:
			surprisals = -1 * np.log2(preds)
			probs = preds
			entropy = float((probs * surprisals).sum(-1))
		result.append((j, word, entropy))
	return result

def retokenize(tokens, reference, model):

	token_symbol = MODEL_TO_SPECIAL_TOKENS[model]
	sent = [x[1] for x in tokens]
	stats = [x[2] for x in tokens]
	retokenized_sent, retokenized_stat = [], []

	i, j, = 0, 0
	while True:
		if i >= len(sent) - 1:
			break
		while True:
			j += 1
			if j >= len(sent):
				break
			if sent[j].startswith(token_symbol):
				break
		retokenized_sent.append("".join(sent[i:j]))
		retokenized_stat.append(sum(stats[i:j]))
		i = j

	if len(reference) == len(retokenized_sent):
		return retokenized_stat
	else:
		print("Tokenization error:")
		print(retokenized_sent)
		print(reference)
		print(retokenized_stat)
		print("\n")
		return [0.0 for _ in range(len(reference))]

def get_stats(sent, trialid, lang, model):
	tokenized_sent = sent.split(" ")
	predictions = get_predictions(sent, model)

	frequencies = [word_frequency(w.strip().strip(punctuation), LANG_CODE[lang], wordlist='best', minimum=0.0) for w in tokenized_sent]
	surprisals = retokenize(get_surprisals(predictions), tokenized_sent, model)
	entropies = retokenize(get_entropies(predictions), tokenized_sent, model)

	return [[trialid, i, tokenized_sent[i], frequencies[i], surprisals[i], entropies[i]] for i in range(len(tokenized_sent))]

def get_predictors(lang, model):

	print("Processing data for %s" % lang)

	rt_df = pd.read_csv(LANG_DATA_PATH + "/" + lang + ".csv")
	#sents = rt_df[['trialid', 'sentnum', 'ianum', 'ia']].drop_duplicates().dropna().groupby(
		#by=['trialid', 'sentnum']).apply(lambda x: ordered_string_join(zip(x['ianum'], x['ia']), ' ')).to_dict()

	sents = rt_df[['trialid', 'ianum', 'ia']].drop_duplicates().dropna().groupby(
		by=['trialid']).apply(lambda x: ordered_string_join(zip(x.index, x['ia']), ' ')).to_dict()

	stats = []
	for k, v in sents.items():
		stats = stats + get_stats(v, k, lang, model)

	df_export = pd.DataFrame(stats, columns = ["trialid", "ianum", "ia", "freq", "surp", "ent"])

	df_export.to_csv("../data/lm_results_raw/"+model+"_"+lang+"_long_preds.csv", sep="\t")


def main():
	model = "mgpt"
	langs = [re.sub(".csv", "", x) for x in os.listdir(LANG_DATA_PATH)]
	# Dropping Estonian and Norwegian. Not supported by mGPT
	filter_langs = ["ee", "no"]
	langs = filter(lambda x: x not in filter_langs, langs)
	langs = ["ru"]

	for lang in langs:
		get_predictors(lang, model)


if __name__ == '__main__':
	main()