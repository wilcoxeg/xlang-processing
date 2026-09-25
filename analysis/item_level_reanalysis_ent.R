#!/usr/bin/env Rscript
# Item-level reanalysis of the entropy contrasts (companion to
# item_level_reanalysis.R and analysis_ent.Rmd).
#
# analysis_ent.Rmd tests two contrasts against the surprisal model:
#   add     : does entropy add over surprisal?  (surp + ent) minus (surp)
#   replace : entropy in place of surprisal.     (ent)        minus (surp)
# Both are tested with the same released per-word sign-flip permutation test that
# counts each averaged word as independent, though the words are nested in 12
# texts (trialid). This script keeps the contrasts and applies the same three
# minimal changes so the test respects the items: item random intercept
# (1 | trialid), leave-one-text-out folds with predictive spread
# sqrt(sigma^2 + tau^2), and a sign-flip permutation over the per-text means, with
# Benjamini-Hochberg within each contrast. Held-out R^2 and dR2 (relative to the
# surprisal model) are reported. The released per-word test is computed alongside.
#
# Model labels (as in the released data):
#   mgpt_sc   : multilingual GPT (mGPT), surprisal from sentence context only
#   mgpt_lc   : multilingual GPT (mGPT), surprisal from the whole preceding passage
#   monot_30m : monolingual transformer trained from scratch on 30M words
#   monot_all : monolingual transformer trained on the full corpus
#
# Run from analysis/:
#   Rscript item_level_reanalysis_ent.R --lang en

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

lex <- c("freq", "len", "prev_freq", "prev_len", "prev2_freq", "prev2_len")
surp_terms <- c("surp", "prev_surp", "prev2_surp")
ent_terms <- c("ent", "prev_ent", "prev2_ent")
allcols <- c(lex, surp_terms, ent_terms)

signflip_p <- function(x, nsim) {
  x <- x[is.finite(x)]; n <- length(x); if (n < 2) return(NA_real_)
  obs <- abs(mean(x)); hits <- 0L
  for (i in seq_len(nsim)) {
    s <- sample(c(-1, 1), n, replace = TRUE)
    if (abs(mean(s * x)) >= obs - 1e-12) hits <- hits + 1L
  }
  (hits + 1) / (nsim + 1)
}
cv_lm <- function(form, df, dvar, nfolds = 10) {
  folds <- cut(seq_len(nrow(df)), breaks = nfolds, labels = FALSE); est <- numeric(0)
  for (k in seq_len(nfolds)) {
    te <- which(folds == k); m <- lm(as.formula(form), data = df[-te, ])
    est <- c(est, log(dnorm(df[te, ][[dvar]], mean = predict(m, newdata = df[te, ]), sd = sigma(m))))
  }
  est
}
cv_lmer <- function(form, df, dvar) {
  texts <- sort(unique(df$trialid)); est <- numeric(0); tid <- numeric(0); prd <- numeric(0); act <- numeric(0)
  for (t in texts) {
    tr <- df[df$trialid != t, ]; te <- df[df$trialid == t, ]
    m <- lmer(as.formula(paste(form, "+ (1 | trialid)")), data = tr, REML = FALSE,
              control = lmerControl(calc.derivs = FALSE))
    vc <- as.data.frame(VarCorr(m)); tau2 <- sum(vc$vcov[vc$grp == "trialid"], na.rm = TRUE)
    pr <- predict(m, newdata = te, re.form = NA, allow.new.levels = TRUE)
    est <- c(est, log(dnorm(te[[dvar]], mean = pr, sd = sqrt(sigma(m)^2 + tau2))))
    tid <- c(tid, te$trialid); prd <- c(prd, pr); act <- c(act, te[[dvar]])
  }
  list(ll = est, tid = tid, pred = prd, y = act)
}
# One contrast: target model vs baseline model on the same rows.
contrast_row <- function(tf, bf, de, ps) {
  dll_word <- cv_lm(tf, de, ps) - cv_lm(bf, de, ps)
  dll_word <- dll_word[is.finite(dll_word)]
  p_word <- signflip_p(dll_word, nsim_word)
  tg <- cv_lmer(tf, de, ps); bl <- cv_lmer(bf, de, ps)
  dll_r <- tg$ll - bl$ll; ok <- is.finite(dll_r)
  per_item <- tapply(dll_r[ok], tg$tid[ok], mean)
  p_item <- signflip_p(as.numeric(per_item), nsim_item)
  sst <- sum((tg$y - mean(tg$y))^2)
  r2_full <- 1 - sum((tg$pred - tg$y)^2) / sst
  r2_base <- 1 - sum((bl$pred - bl$y)^2) / sst
  list(mean_dll = mean(dll_word), dr2 = r2_full - r2_base, r2_full = r2_full,
       p_word = p_word, p_item = p_item)
}

df <- read.csv(file.path(in_dir, paste0(lang, ".csv")))
df <- df[df$freq > 0 & df$prev_freq > 0 & df$prev2_freq > 0 &
           is.finite(df$freq) & is.finite(df$prev_freq) & is.finite(df$prev2_freq), ]

fm <- function(terms) paste(terms, collapse = " + ")
rows <- list()
for (mm in models) {
  de0 <- df[df$model == mm, ]
  for (ps in measures) {
    de <- de0[complete.cases(de0[, c(allcols, ps, "trialid")]), ]
    f_surp <- paste0(ps, " ~ ", fm(c(surp_terms, lex)))
    f_add <- paste0(ps, " ~ ", fm(c(ent_terms, surp_terms, lex)))
    f_repl <- paste0(ps, " ~ ", fm(c(ent_terms, lex)))
    for (con in c("add", "replace")) {
      tf <- if (con == "add") f_add else f_repl
      r <- contrast_row(tf, f_surp, de, ps)
      rows[[length(rows) + 1]] <- data.frame(
        contrast = con, model = mm, measure = ps,
        mean_dll = r$mean_dll, dr2 = r$dr2, r2_full = r$r2_full,
        p_word = r$p_word, p_item = r$p_item, stringsAsFactors = FALSE)
    }
  }
}
res <- do.call(rbind, rows)
# Benjamini-Hochberg within each contrast (12 cells each).
res$p_word_BH <- NA_real_; res$p_item_BH <- NA_real_
for (con in unique(res$contrast)) {
  ix <- res$contrast == con
  res$p_word_BH[ix] <- p.adjust(res$p_word[ix], method = "BH")
  res$p_item_BH[ix] <- p.adjust(res$p_item[ix], method = "BH")
}
res$mean_dll <- round(res$mean_dll, 5)
for (c in c("dr2", "r2_full", "p_word", "p_item", "p_word_BH", "p_item_BH")) res[[c]] <- round(res[[c]], 4)

for (con in c("add", "replace")) {
  cat(sprintf("\nEntropy %s contrast (vs the surprisal model), %s\n", con, lang))
  sub <- res[res$contrast == con, setdiff(names(res), "contrast")]
  print(sub, row.names = FALSE)
  cat(sprintf("significant at .05 after BH: per-word %d/%d, item-level %d/%d\n",
              sum(sub$p_word_BH < 0.05), nrow(sub), sum(sub$p_item_BH < 0.05), nrow(sub)))
}
if (!is.na(out_tsv)) {
  write.table(res, out_tsv, sep = "\t", quote = FALSE, row.names = FALSE)
  cat(sprintf("wrote %s\n", out_tsv))
}
