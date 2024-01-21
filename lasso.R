packages_to_install <- c("textdata", "tidytext", "dplyr", "tidyr", "stringr", "scales", "wordcloud2", "htmlwidgets","tidymodels","tidyverse","vip")
install.packages(packages_to_install, dependencies = TRUE)

library(tidyverse)
# Read input data
df <- read.csv("C:/Professional/MS/SHU/Athira_dissertation/preprocessed_kindle_review.csv", header = TRUE)

print(sapply(df,class))


# Check for missing values
missing_values <- df %>%
  summarise_all(~sum(is.na(.)))
print(missing_values)
df %>%
  count(rating) %>%
  ggplot(aes(rating, n)) +
  geom_col(fill = "midnightblue", alpha = 0.7)


reviews_parsed <- df %>%
  mutate(total_ratings = case_when(
    rating > 3 ~ "good",
    TRUE ~ "bad"
  ))

library(tidytext)

words_per_review <- reviews_parsed %>%
  unnest_tokens(word, reviewText) %>%
  count(X, name = "total_words")

words_per_review %>%
  ggplot(aes(total_words)) +
  geom_histogram(fill = "midnightblue", alpha = 0.8)


library(tidymodels)

set.seed(123)
review_split <- initial_split(reviews_parsed, strata = total_ratings)
review_train <- training(review_split)
review_test <- testing(review_split)

library(textrecipes)

review_rec <- recipe(total_ratings ~ reviewText, data = review_train) %>%
  step_tokenize(reviewText) %>%
  step_stopwords(reviewText) %>%
  step_tokenfilter(reviewText, max_tokens = 500) %>%
  step_tfidf(reviewText) %>%
  step_normalize(all_predictors())

review_prep <- prep(review_rec)

review_prep

lasso_spec <- logistic_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet")

lasso_wf <- workflow() %>%
  add_recipe(review_rec) %>%
  add_model(lasso_spec)
lasso_wf

lambda_grid <- grid_regular(penalty(), levels = 40)
lambda_grid

set.seed(123)
review_folds <- bootstraps(review_train, strata = total_ratings)
review_folds

doParallel::registerDoParallel()

set.seed(2020)
lasso_grid <- tune_grid(
  lasso_wf,
  resamples = review_folds,
  grid = lambda_grid,
  metrics = metric_set(roc_auc, ppv, npv)
)

lasso_grid %>%
  collect_metrics()

lasso_grid %>%
  collect_metrics() %>%
  ggplot(aes(penalty, mean, color = .metric)) +
  geom_line(size = 1.5, show.legend = FALSE) +
  facet_wrap(~.metric) +
  scale_x_log10()

best_auc <- lasso_grid %>%
  select_best("roc_auc")

best_auc

final_lasso <- finalize_workflow(lasso_wf, best_auc)

final_lasso

library(vip)

final_lasso %>%
  fit(review_train) %>%
  pull_workflow_fit() %>%
  vi(lambda = best_auc$penalty) %>%
  group_by(Sign) %>%
  top_n(20, wt = abs(Importance)) %>%
  ungroup() %>%
  mutate(
    Importance = abs(Importance),
    Variable = fct_reorder(Variable, Importance)
  ) %>%
  ggplot(aes(x = Importance, y = Variable, fill = Sign)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~Sign, scales = "free_y") +
  labs(y = NULL)


review_final <- last_fit(final_lasso, review_split)

review_final %>%
  collect_metrics()

review_final %>%
  collect_predictions() %>%
  conf_mat(total_ratings, .pred_class)
