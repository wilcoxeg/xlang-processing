#!/usr/bin/env Rscript
# The pooled mixed-effects test and predictors that are constant within a text.
#
# Clark (1973) points out that words, sentences and texts in an experiment are
# a sample of the language, in the same way that participants are a sample of
# readers. A test that treats the materials as fixed supports a claim about
# those particular materials only. When the claim is about the language, the
# materials have to be treated as a random effect, or the test overstates the
# evidence and rejects too often.
#
# The pooled analysis in analysis_surp.Rmd (L318-L397) stacks the eleven
# languages, shuffles the rows, fits the baseline and target regressions with
# by-language random effects over ten folds, and tests the per-word held-out
# delta log-likelihood with jmuOutlier::perm.test. Neither regression has a
# term for the text (trialid), and the test counts every word as an
# independent observation. Because the rows are shuffled before the folds are
# cut, every text also has words on both the training and the test side of
# every fold.
#
# This script keeps the released pipeline and asks two questions of it.
#
#   1. Give every word the mean surprisal of its text, so that the predictor
#      says nothing about the word and only something about the text. Does the
#      released test call that predictor significant?
#
#   2. Give every word K random numbers drawn once per text (language x
#      trialid), which carry no information at all. How often does the
#      released test call them significant, as K grows?
#
# Two variants can be run alongside, each changing one thing:
#   text_intercept : both regressions get a random intercept for the text,
#                    (1 | text), so the materials are treated as random;
#   text_folds     : the folds hold out whole texts, so no text is on both
#                    sides of a fold.
#
# Run from analysis/ (reads ../data/merged_data/{lang}.csv):
#
#   Rscript text_level_false_positives.R
#   Rscript text_level_false_positives.R --draws 0          # question 1 only
#   Rscript text_level_false_positives.R --draws 30 --ks 1,5,10,20,50
#   Rscript text_level_false_positives.R --variants released,text_intercept,text_folds
#
# Reference:
#   Clark, H. H. (1973). The language-as-fixed-effect fallacy: A critique of
#   language statistics in psychological research. Journal of Verbal Learning
#   and Verbal Behavior, 12(4), 335-359.
#   https://psycnet.apa.org/record/1974-05129-001

suppressPackageStartupMessages({
  library(lme4)
  library(jmuOutlier)
  library(parallel)
})

args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(flag, default) {
  i <- match(flag, args)
  if (is.na(i)) default else args[i + 1]
}
in_dir <- parse_arg("--in", "../data/merged_data")
lm_name <- parse_arg("--model", "mgpt_lc")
measure <- parse_arg("--measure", "gaze_rt")
ks <- as.integer(strsplit(parse_arg("--ks", "1,5,10,20,50"), ",")[[1]])
draws <- as.integer(parse_arg("--draws", "30"))
variants <- strsplit(parse_arg("--variants", "released"), ",")[[1]]
cores <- as.integer(parse_arg("--cores", "10"))
seed <- as.integer(parse_arg("--seed", "444"))
num_sim <- as.integer(parse_arg("--num-sim", "1000"))
out_tsv <- parse_arg("--out", "text_level_false_positives.tsv")
runs_tsv <- sub("\\.tsv$", "_runs.tsv", out_tsv)

langs <- c("du", "en", "fi", "ge", "gr", "he", "it", "ko", "sp", "tr", "ru")

# Released data preparation: stack, filter, scale within language model, shuffle.
df <- do.call(rbind, lapply(langs, function(l) {
  d <- read.csv(file.path(in_dir, paste0(l, ".csv")))
  d$lang <- l
  d
}))
df <- df[df$freq > 0 & df$prev_freq > 0 & df$prev2_freq > 0 &
           is.finite(df$freq) & is.finite(df$prev_freq) & is.finite(df$prev2_freq), ]
to_scale <- unique(c("surp", "len", "freq", measure, "prev_surp", "prev_len", "prev_freq"))
for (m in unique(df$model)) {
  k <- df$model == m
  for (v in to_scale) df[[v]][k] <- as.numeric(scale(df[[v]][k]))
}
set.seed(seed)
df <- df[sample(seq_len(nrow(df))), ]
df <- df[df$model == lm_name, ]
df <- df[complete.cases(df), ]
df$lang <- factor(df$lang)
df$text <- factor(paste(df$lang, df$trialid))
df$y <- df[[measure]]
n <- nrow(df)
text_id <- as.integer(df$text)
n_texts <- nlevels(df$text)

# Surprisal split into the mean of the word's text and the remainder.
df$text_mean_surp <- ave(df$surp, df$text)
df$within_surp <- df$surp - df$text_mean_surp
df$within_prev_surp <- df$prev_surp - ave(df$prev_surp, df$text)

# Released baseline (surprisal block), with the response renamed to y. Each
# target adds its predictors to the fixed part and to the by-language slopes,
# as the released target does for surp and prev_surp.
controls <- "freq + len + prev_freq + prev_len"
make_form <- function(predictors, extra, slopes = TRUE) {
  fixed <- paste(c(predictors, controls), collapse = " + ")
  random <- paste(c(if (slopes) predictors, controls), collapse = " + ")
  paste0("y ~ ", fixed, " + (", random, " | lang)", extra)
}
targets <- list(
  "surprisal" = c("surp", "prev_surp"),
  "text mean of surprisal" = "text_mean_surp",
  "surprisal minus its text mean" = c("within_surp", "within_prev_surp")
)

fold_sets <- list(
  rows = cut(seq_len(n), breaks = 10, labels = FALSE),
  texts = {
    set.seed(seed + 1)
    sample(rep(1:10, length.out = n_texts))[text_id]
  }
)
ctrl <- lmerControl(calc.derivs = FALSE, optimizer = "bobyqa",
                    optCtrl = list(maxfun = 1e5))

# Released cross-validation: fit on nine folds, score the tenth with the
# conditional prediction and the residual standard deviation.
cross_val <- function(form, data, folds) {
  parts <- mclapply(1:10, function(i) {
    te <- folds == i
    model <- suppressMessages(suppressWarnings(
      lmer(as.formula(form), data = data[!te, ], control = ctrl)))
    list(rows = which(te),
         ll = log(dnorm(data$y[te],
                        mean = predict(model, newdata = data[te, ], allow.new.levels = TRUE),
                        sd = sigma(model))))
  }, mc.cores = cores)
  out <- numeric(nrow(data))
  for (p in parts) out[p$rows] <- p$ll
  out
}
released_test <- function(dll) {
  set.seed(seed)
  perm.test(dll, num.sim = num_sim)$p.value
}
record <- function(variant, predictor, k, draw, dll) {
  by_text <- as.numeric(tapply(dll, df$text, mean))
  data.frame(variant = variant, predictor = predictor, k = k, draw = draw,
             mean_dll = mean(dll), share_words_better = mean(dll > 0),
             p = released_test(dll), texts_better = sum(by_text > 0),
             p_over_texts = t.test(by_text)$p.value, stringsAsFactors = FALSE)
}

settings <- list(
  released = list(extra = "", folds = "rows"),
  text_intercept = list(extra = " + (1 | text)", folds = "rows"),
  text_folds = list(extra = "", folds = "texts")
)

runs <- list()
cat(sprintf("%s, %s: %d words, %d texts, %d languages\n",
            lm_name, measure, n, n_texts, nlevels(df$lang)))
for (vn in variants) {
  st <- settings[[vn]]
  folds <- fold_sets[[st$folds]]
  base_ll <- cross_val(make_form(character(0), st$extra), df, folds)

  for (tn in names(targets)) {
    dll <- cross_val(make_form(targets[[tn]], st$extra), df, folds) - base_ll
    runs[[length(runs) + 1]] <- record(vn, tn, length(targets[[tn]]), 0L, dll)
    cat(sprintf("[%s] %s: mean dLL %.5f, p = %.4f\n",
                vn, tn, mean(dll), runs[[length(runs)]]$p))
  }

  for (k in ks) {
    if (draws < 1) break
    noise_names <- paste0("z", seq_len(k))
    noise_form <- make_form(noise_names, st$extra, slopes = FALSE)
    for (d in seq_len(draws)) {
      set.seed(seed + 100000L * k + d)
      z <- matrix(rnorm(n_texts * k), n_texts, k)[text_id, , drop = FALSE]
      colnames(z) <- noise_names
      dll <- cross_val(noise_form, cbind(df, z), folds) - base_ll
      runs[[length(runs) + 1]] <- record(vn, "random numbers per text", k, d, dll)
      write.table(do.call(rbind, runs), runs_tsv, sep = "\t",
                  quote = FALSE, row.names = FALSE)
    }
    r <- do.call(rbind, runs)
    r <- r[r$variant == vn & r$k == k & r$draw > 0, ]
    cat(sprintf("[%s] K = %d: declared helpful in %d of %d draws, harmful in %d\n",
                vn, k, sum(r$p < 0.05 & r$mean_dll > 0), nrow(r),
                sum(r$p < 0.05 & r$mean_dll < 0)))
  }
}

runs <- do.call(rbind, runs)
write.table(runs, runs_tsv, sep = "\t", quote = FALSE, row.names = FALSE)

cat("\n1. Surprisal and its two parts\n\n")
shown <- runs[runs$draw == 0, c("variant", "predictor", "mean_dll", "share_words_better",
                                "p", "texts_better", "p_over_texts")]
shown$mean_dll <- signif(shown$mean_dll, 3)
shown$share_words_better <- round(shown$share_words_better, 3)
shown$p_over_texts <- signif(shown$p_over_texts, 2)
print(shown, row.names = FALSE)

noise <- runs[runs$draw > 0, ]
if (nrow(noise) > 0) {
  res <- do.call(rbind, lapply(split(noise, list(noise$k, noise$variant), drop = TRUE), function(s) {
    data.frame(
      variant = s$variant[1], k = s$k[1], draws = nrow(s),
      false_positive_rate = mean(s$p < 0.05 & s$mean_dll > 0),
      declared_harmful_rate = mean(s$p < 0.05 & s$mean_dll < 0),
      median_dll = median(s$mean_dll), max_dll = max(s$mean_dll),
      stringsAsFactors = FALSE)
  }))
  res <- res[order(match(res$variant, variants), res$k), ]
  rownames(res) <- NULL
  cat("\n2. Random numbers per text: share of draws the test declares significant at .05\n")
  cat("   (nominal rate: 0.025 in each direction)\n\n")
  shown <- res
  shown$false_positive_rate <- round(shown$false_positive_rate, 3)
  shown$declared_harmful_rate <- round(shown$declared_harmful_rate, 3)
  shown$median_dll <- signif(shown$median_dll, 2)
  shown$max_dll <- signif(shown$max_dll, 2)
  print(shown, row.names = FALSE)
  write.table(res, out_tsv, sep = "\t", quote = FALSE, row.names = FALSE)
  cat(sprintf("\nwrote %s and %s\n", out_tsv, runs_tsv))
} else {
  cat(sprintf("\nwrote %s\n", runs_tsv))
}
