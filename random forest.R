packages_to_install <- c("textdata", "tidytext", "dplyr", "tidyr", "stringr", "scales", "wordcloud2", "htmlwidgets","tidymodels","tidyverse","vip","afinn")
install.packages(packages_to_install, dependencies = TRUE)

library(tidyverse)
# Read input data
df <- read.csv("C:/Professional/MS/SHU/Athira_dissertation/preprocessed_kindle_review.csv", header = TRUE)

reviews_parsed <- df %>%
  mutate(total_ratings = case_when(
    rating > 3 ~ "good",
    TRUE ~ "bad"
  ))
library(tidymodels)
library(textrecipes)
library(afinn)
afinn <- get_sentiments("afinn")


review_rec <- recipe(total_ratings ~ reviewText, data = reviews_parsed) %>%
  step_tokenize(reviewText) %>%
  step_stopwords(reviewText) %>%
  step_tokenfilter(reviewText) %>%
  step_word_embeddings(reviewText, embeddings = afinn)


set.seed(123)
trees_split <- initial_split(reviews_parsed, strata = total_ratings)
trees_train <- training(trees_split)
trees_test <- testing(trees_split)


tune_spec <- rand_forest(
  mtry = tune(),
  trees = 1000,
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("ranger")


tune_wf <- workflow() %>%
  add_recipe(review_rec) %>%
  add_model(tune_spec)

set.seed(123)
trees_folds <- vfold_cv(trees_train)

doParallel::registerDoParallel()

set.seed(345)
tune_res <- tune_grid(
  tune_wf,
  resamples = trees_folds,
  grid = 20
)

tune_res

tune_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, min_n, mtry) %>%
  pivot_longer(min_n:mtry,
               values_to = "value",
               names_to = "parameter"
  ) %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "AUC")

rf_grid <- grid_regular(
  mtry(range = c(1,500)),
  min_n(range = c(15,25)),
  levels = 5
)

rf_grid

set.seed(456)
regular_res <- tune_grid(
  tune_wf,
  resamples = trees_folds,
  grid = rf_grid
)

regular_res

regular_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(aes(mtry, mean, color = min_n)) +
  geom_line(alpha = 0.5, size = 1.5) +
  geom_point() +
  labs(y = "AUC")

best_auc <- select_best(regular_res, "roc_auc")

final_rf <- finalize_model(
  tune_spec,
  best_auc
)

final_rf

final_wf <- workflow() %>%
  add_recipe(review_rec) %>%
  add_model(final_rf)

final_res <- final_wf %>%
  last_fit(trees_split)

final_res %>%
  collect_metrics()


final_res %>%
  collect_predictions() %>%
  conf_mat(total_ratings, .pred_class)

