l
# Install and load required packages
packages_to_install <- c("textdata", "tidytext", "dplyr", "tidyr", "stringr", "scales", "wordcloud2", "htmlwidgets","tidymodels","tidyverse","vip")
install.packages(packages_to_install, dependencies = TRUE)

# Load the installed packages
library(textdata)
library(tidytext)
library(dplyr)
library(tidyr)
library(stringr)
library(scales)
library(wordcloud2)
library(htmlwidgets)

# Load sentiment lexicons
afinn <- get_sentiments("afinn")

# Read input data
df <- read.csv("C:/Professional/MS/SHU/Athira_dissertation/preprocessed_kindle_review.csv", header = TRUE)

# Tokenize the text and remove stopwords
tokens <- df %>%
  unnest_tokens(word, reviewText) %>%
  anti_join(stop_words)%>%
  mutate(word = str_to_lower(word))

# Sentiment wordcloud using AFINN lexicon
sentiment_wordcloud <- tokens %>%
  inner_join(afinn, by = "word",) %>%
  group_by(word) %>%
  summarize(sentiment_score_afinn = sum(value)) 

# Assuming sentiment_score_afinn > 0 as positive and < 0 as negative
positive_afinn <- sentiment_wordcloud %>%
  filter(sentiment_score_afinn > 0)

negative_afinn <- sentiment_wordcloud %>%
  filter(sentiment_score_afinn < 0) %>%
  mutate(sentiment_score_afinn = -sentiment_score_afinn)

# Create a function to generate word clouds
generate_word_cloud <- function(data, title) {
  wordcloud2(data, size = 0.8, shape = "circle", backgroundColor = "white") %>%
    htmlwidgets::onRender(sprintf('function(el, x) {el.innerHTML = "<h4>%s</h4>";}', title))
}

# Generate word clouds for positive and negative sentiments
generate_word_cloud(positive_afinn, "Positive Emotions Word Cloud")
generate_word_cloud(negative_afinn, "Negative Emotions Word Cloud")


# Sentiment analysis using AFINN lexicon
sentiment_afinn <- tokens %>%
  inner_join(afinn, by = "word",) %>%
  group_by(X) %>%
  summarize(sentiment_score_afinn = sum(value)) 

# Combine sentiment scores with the initial dataframe
df_with_sentiment <- df %>%
  left_join(sentiment_afinn, by = "X")

write.csv(df_with_sentiment,"C:/Professional/MS/SHU/Athira_dissertation/combined_data.csv")
# View the combined dataframe
View(df_with_sentiment)




