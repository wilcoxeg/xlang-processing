#!/usr/bin/env Rscript
# Item-level reanalysis of the surprisal delta-log-likelihood test.
#
# The surprisal analysis in analysis_surp.Rmd (L48-L150) averages reading time
# across participants per word, fits the target and baseline regressions with lm
# over ten row-order folds, and tests the per-word held-out delta log-likelihood
# with a sign-flip permutation test (jmuOutlier::perm.test). That test treats
# every averaged word as an independent observation, so its effective sample
# size is the ~2000 words rather than the reading passages the words are nested
# in (one text per trialid, 12 per language in MECO L1).
#
# This script keeps the same delta-log-likelihood contrast and makes three
# minimal changes so the test respects the items:
#   1. an item random intercept (1 | trialid) is added to both regressions;
#   2. the folds are leave-one-text-out, so no text is split across train and
#      test, and the held-out density uses the mixed-model predictive spread
#      sqrt(sigma^2 + tau^2);
#   3. the sign-flip permutation runs over the per-text mean delta
#      log-likelihoods, so its independent units are the texts, not the words.
# Benjamini-Hochberg correction is applied across the reported cells.
#
# The released per-word test is computed alongside so both decisions sit in one
# table. The sign-flip permutation is spelled out here (mean statistic, random
# sign flips); it is the one-sample paired test jmuOutlier::perm.test performs,
# without the dependency.
#
# Run from analysis/ (reads ../data/merged_data/{lang}.csv by default):
#
#   Rscript item_level_reanalysis.R --lang en
#   Rscript item_level_reanalysis.R --lang en --out item_reanalysis_en.tsv
#
# Model labels (as in the released data):
#   mgpt_sc   : multilingual GPT (mGPT), surprisal from sentence context only
#   mgpt_lc   : multilingual GPT (mGPT), surprisal from the whole preceding passage
#   monot_30m : monolingual transformer trained from scratch on 30M words
#   monot_all : monolingual transformer trained on the full corpus

suppressPackageStartupMessages(library(lme4))

args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(flag, default) {
  i <- match(flag, args)
  if (is.na(i)) default else args[i + 1]
}
in_dir <- parse_arg("--in", "../data/merged_data")
lang <- parse_arg("--lang", "en")
out_tsv <- parse_arg("--out", NA)
seed <- as.integer(parse_arg("--seed", "444"))
nsim_word <- as.integer(parse_arg("--nsim-word", "2000"))
nsim_item <- as.integer(parse_arg("--nsim-item", "4096"))
models <- strsplit(parse_arg("--models", "mgpt_sc,mgpt_lc,monot_30m,monot_all"), ",")[[1]]
measures <- strsplit(parse_arg("--measures", "gaze_rt,total_rt,firstfix_rt"), ",")[[1]]
set.seed(seed)

pred <- c("surp", "prev_surp", "prev2_surp", "freq", "len",
          "prev_freq", "prev_len", "prev2_freq", "prev2_len")
base <- setdiff(pred, c("surp", "prev_surp", "prev2_surp"))

# One-sample sign-flip permutation p-value on the mean of x.
signflip_p <- function(x, nsim) {
  x <- x[is.finite(x)]
  n <- length(x)
  if (n < 2) return(NA_real_)
  obs <- abs(mean(x))
  hits <- 0L
  for (i in seq_len(nsim)) {
    s <- sample(c(-1, 1), n, replace = TRUE)
    if (abs(mean(s * x)) >= obs - 1e-12) hits <- hits + 1L
  }
  (hits + 1) / (nsim + 1)
}

# Released path: lm, ten row-order folds, per-word held-out log density.
cv_lm <- function(form, df, dvar, nfolds = 10) {
  folds <- cut(seq_len(nrow(df)), breaks = nfolds, labels = FALSE)
  est <- numeric(0)
  for (k in seq_len(nfolds)) {
    te <- which(folds == k)
    m <- lm(as.formula(form), data = df[-te, ])
    est <- c(est, log(dnorm(df[te, ][[dvar]],
                            mean = predict(m, newdata = df[te, ]), sd = sigma(m))))
  }
  est
}

# Reanalysis path: lmer + (1 | trialid), leave-one-text-out folds, marginal
# prediction for the held-out text with predictive spread sqrt(sigma^2 + tau^2).
cv_lmer <- function(form, df, dvar) {
  texts <- sort(unique(df$trialid))
  est <- numeric(0)
  tid <- numeric(0)
  prd <- numeric(0)
  act <- numeric(0)
  for (t in texts) {
    tr <- df[df$trialid != t, ]
    te <- df[df$trialid == t, ]
    m <- lmer(as.formula(paste(form, "+ (1 | trialid)")), data = tr,
              REML = FALSE, control = lmerControl(calc.derivs = FALSE))
    vc <- as.data.frame(VarCorr(m))
    tau2 <- sum(vc$vcov[vc$grp == "trialid"], na.rm = TRUE)
    pr <- predict(m, newdata = te, re.form = NA, allow.new.levels = TRUE)
    est <- c(est, log(dnorm(te[[dvar]], mean = pr, sd = sqrt(sigma(m)^2 + tau2))))
    tid <- c(tid, te$trialid)
    prd <- c(prd, pr)
    act <- c(act, te[[dvar]])
  }
  list(ll = est, tid = tid, pred = prd, y = act)
}

df <- read.csv(file.path(in_dir, paste0(lang, ".csv")))
df <- df[df$freq > 0 & df$prev_freq > 0 & df$prev2_freq > 0 &
           is.finite(df$freq) & is.finite(df$prev_freq) & is.finite(df$prev2_freq), ]

rows <- list()
for (mm in models) {
  de0 <- df[df$model == mm, ]
  for (ps in measures) {
    de <- de0[complete.cases(de0[, c(pred, ps, "trialid")]), ]
    tf <- paste0(ps, " ~ ", paste(pred, collapse = " + "))
    bf <- paste0(ps, " ~ ", paste(base, collapse = " + "))

    dll_word <- cv_lm(tf, de, ps) - cv_lm(bf, de, ps)
    dll_word <- dll_word[is.finite(dll_word)]
    p_word <- signflip_p(dll_word, nsim_word)

    tg <- cv_lmer(tf, de, ps)
    bl <- cv_lmer(bf, de, ps)
    dll_r <- tg$ll - bl$ll
    ok <- is.finite(dll_r)
    per_item <- tapply(dll_r[ok], tg$tid[ok], mean)
    p_item <- signflip_p(as.numeric(per_item), nsim_item)

    # Held-out variance explained (leave-one-text-out, marginal prediction):
    # the full (surprisal) model, the baseline, and the gain from surprisal.
    sst <- sum((tg$y - mean(tg$y))^2)
    r2_full <- 1 - sum((tg$pred - tg$y)^2) / sst
    r2_base <- 1 - sum((bl$pred - bl$y)^2) / sst
    dr2 <- r2_full - r2_base

    rows[[length(rows) + 1]] <- data.frame(
      lang = lang, model = mm, measure = ps,
      mean_dll = mean(dll_word), r2_base = r2_base, r2_full = r2_full, dr2 = dr2,
      n_words = length(dll_word), n_items = length(per_item),
      p_word = p_word, p_item = p_item, stringsAsFactors = FALSE)
  }
}
res <- do.call(rbind, rows)
res$p_word_BH <- p.adjust(res$p_word, method = "BH")
res$p_item_BH <- p.adjust(res$p_item, method = "BH")

res$mean_dll <- round(res$mean_dll, 5)
for (c in c("r2_base", "r2_full", "dr2")) res[[c]] <- round(res[[c]], 4)
for (c in c("p_word", "p_item", "p_word_BH", "p_item_BH")) res[[c]] <- round(res[[c]], 4)

cat(sprintf("\nSurprisal test, %s: released per-word vs item-level reanalysis\n", lang))
cat(sprintf("(%d texts held out one at a time; BH across the %d cells)\n\n", res$n_items[1], nrow(res)))
print(res, row.names = FALSE)
cat(sprintf("\nsignificant at .05 after BH: per-word %d/%d, item-level %d/%d\n",
            sum(res$p_word_BH < 0.05), nrow(res), sum(res$p_item_BH < 0.05), nrow(res)))

if (!is.na(out_tsv)) {
  write.table(res, out_tsv, sep = "\t", quote = FALSE, row.names = FALSE)
  cat(sprintf("wrote %s\n", out_tsv))
}
