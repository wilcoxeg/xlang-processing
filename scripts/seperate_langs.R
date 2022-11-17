
library(tidyverse)

load("joint_data.rda")

langs = unique(joint.data$lang)
langs = langs[!is.na(langs)]

for ( x in langs) {
  df = joint.data %>% filter(lang == x)
  write.csv(df, paste("./langs/", x, ".csv", sep = ""))
}