# compile a random dataset
# balancing the different sheets / handwritten styles
# the characters represented, the uncertainty of the
# citizen science transcriptions and empty values
#
# ideally a subset of the full dataset (>350K) values
# is made and restricted to a couple of thousand values
# to fit within the repo for demo purposes, only to train
# the full model on a separate data directory later

library(dplyr)
library(tidyr)
library(readr)

# read in the majority votes for the Jungle Weather
# citizen science programme
df <- readr::read_csv("data-raw/climate_data_majority_vote.csv")

# only retain values with absolute concensus between
# volunteers, can be relaxed to grow the dataset
df <- df |>
  filter(
    consensus == 1
  ) |>
  select(
    filename,
    final_value,
    folder,
    col
  )

# how many images are retained
message(sprintf("number of retained images: %s", nrow(df)))

# too much snot in my head just do a simple
# random sampling for now with 10k samples
df <- df |>
  sample_n(size = 10000) |>
  mutate(
    filename = gsub(".png","_dl.png", filename)
  )

readr::write_csv(df, "data/tmp/labels.csv")

files <- list.files("/data/scratch/zooniverse/format_1_batch_1_dl/","*.png")

# copy all files over to tmp directory
file.copy(
  file.path("/data/scratch/zooniverse/format_1_batch_1_dl/",df$filename),
  "data/tmp/"
)

# split out individual characters
# run some statistics on the prevalence of the characters
# present in the dataset
char_stats <- df |>
  rowwise() |>
  mutate(
    character = strsplit(as.character(final_value[1]),"")
  ) |>
  unnest(cols = "character") |>
  ungroup() |>
  group_by(character) |>
  summarize(
    n = n()
  ) |>
  arrange(
    n
  )
