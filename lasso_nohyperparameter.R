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

library(tidymodels)

set.seed(123)
review_split <- initial_split(reviews_parsed, strata = total_ratings)
review_train <- training(review_split)
review_test <- testing(review_split)

library(textrecipes)

review_rec <- recipe(total_ratings ~ reviewText, data = review_train) %>%
  step_tokenize(reviewText) %>%
  step_stopwords(reviewText) %>%
  step_tokenfilter(reviewText) %>%
  step_tfidf(reviewText) %>%
  step_normalize(all_predictors())

lasso_spec <- logistic_reg(penalty = 1, mixture = 1) %>%
  set_engine("glmnet")

lasso_wf <- workflow() %>%
  add_recipe(review_rec) %>%
  add_model(lasso_spec)

review_final <- last_fit(lasso_wf, review_split)

review_final %>%
  collect_metrics()

review_final %>%
  collect_predictions() %>%
  conf_mat(total_ratings, .pred_class)
