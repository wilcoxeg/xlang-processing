
library(tidyverse)

load("joint_data.rda")
langs = unique(joint.data$lang)
langs = langs[!is.na(langs)]

for ( x in langs) {
  df = joint.data %>% filter(lang == x)
  write.csv(df, paste("./langs_l1/", x, ".csv", sep = ""))
}


load("joint_data_l2_trimmed.rda")
langs = unique(joint.data$lang)
langs = langs[!is.na(langs)]

for ( x in langs) {
  df = joint.data %>% filter(lang == x)
  write.csv(df, paste("./langs_l2/", x, ".csv", sep = ""))
}
