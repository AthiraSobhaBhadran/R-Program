library(tidyverse)
library(tidymodels)
library(textrecipes)
library(afinn)

# Read input data
df <- read.csv("C:/Professional/MS/SHU/Athira_dissertation/preprocessed_kindle_review.csv", header = TRUE)

reviews_parsed <- df %>%
  mutate(total_ratings = case_when(
    rating > 3 ~ "good",
    TRUE ~ "bad"
  ))

# Create recipe
review_rec <- recipe(total_ratings ~ reviewText, data = reviews_parsed) %>%
  step_tokenize(reviewText) %>%
  step_stopwords(reviewText) %>%
  step_tokenfilter(reviewText) %>%
  step_word_embeddings(reviewText, embeddings = afinn)

# Split the data
trees_split <- initial_split(reviews_parsed, strata = total_ratings)
trees_train <- training(trees_split)
trees_test <- testing(trees_split)

# Create Random Forest model without tuning
rf_spec <- rand_forest(
  mtry = 10,  # Set your desired value
  trees = 1000,
  min_n = 15  # Set your desired value
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

# Create workflow
rf_wf <- workflow() %>%
  add_recipe(review_rec) %>%
  add_model(rf_spec)

# Fit Random Forest model
rf_fit <- rf_wf %>%
  last_fit(trees_split)

# Collect metrics
rf_fit %>%
  collect_metrics()

# Confusion matrix
rf_fit %>%
  collect_predictions() %>%
  conf_mat(total_ratings, .pred_class)

