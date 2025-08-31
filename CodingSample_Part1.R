### Coding Sample.
### 1. DATA EXTRACTION FROM SQL DATABASE
### 2. BASIC NLP ANALYSIS OF REDDIT DATA
### 3. See Python file for advanced NLP models
###

## Set working directory
setwd("XXXXXXXXXXXXX")

## Loading required libraries for SQL extraction

library(DBI)
library(RPostgres)

## Establishing database connection

con <- dbConnect(
    RPostgres::Postgres(),
    dbname   = "xxxx",
    host     = "xxxx",
    port     = xxxx,
    user     = "xxxx",
    password = "xxxx"
)


## Exploring the SQL database; List tables in SQL

dbListTables(con)
# [1] "comments"    "submissions" ## available tables

## List fields in "submissions"

dbListFields(con, "submissions")

## Confirming which fields have been indexed

dbGetQuery(con, "
  SELECT
    indexname,
    indexdef
  FROM pg_indexes
  WHERE tablename = 'submissions';
")


## Assessing available date range

dbGetQuery(con, "
  SELECT
    MIN(created_utc) AS earliest,
    MAX(created_utc) AS latest
  FROM submissions;
")

## Assessing number of rows

dbGetQuery(con, "SELECT COUNT(*) AS total_rows FROM submissions;")

## Note for query building: data is between 2/1/2025 and 5/31/2025. 155,156,164 total rows.

## Choosing subreddits

subreddits <- c(
    'relationships', 'relationship_advice', 'dating_advice',
    'BreakUps', 'Divorce', 'exnocontact'
)

## Building query

subs_sql <- paste(DBI::dbQuoteString(con, subreddits), collapse = ", ")


query <- paste0("
  SELECT subreddit, COUNT(*) AS post_count
  FROM submissions
  WHERE subreddit IN (", subs_sql, ")
  GROUP BY subreddit
  ORDER BY post_count DESC;
")

## Checking number of posts/subreddit available to see if batching required.

relationship_counts <- dbGetQuery(con, query)

## Pull data

query2 <- paste0("
  SELECT *
  FROM submissions
  WHERE subreddit IN (", subs_sql, ")
")

relationship_df <- dbGetQuery(con, query2)

## Confirming all submissions have been extracted by comparing df and counts.
table(relationship_df$subreddit)
relationship_counts

library(readr)

# Save dataframe as CSV
write_csv(relationship_df, "relationships_df.csv")

dbDisconnect(con) ## End SQL Connection

#################################################
#################################################
########### Text Analysis

## Examine the dataset
str(relationship_df)

## Load necessary libraries
library(dplyr)
library(stringr)

## Checking to see if "id" is a unique identifier

n_distinct(relationship_df$id) ## no duplicate IDs

# Checking for unique number of users
n_distinct(relationship_df$author) ## 198,449 unique users and 334,297 posts


# Combine title and selftext into a single text field
# Remove extra whitespace
# Drop unnecessary columns
relationship_df_selected <- relationship_df %>%
    mutate(
        text = str_trim(paste(title, selftext))
    ) %>%
    select(id, subreddit, created_utc, text, score, num_comments, author)


## Assessing average length of posts

relationship_df_selected <- relationship_df_selected %>%
    mutate(text_length = nchar(text))

summary(relationship_df_selected$text_length)


## Prepare Corpus

library(quanteda)

# Create a corpus
corp <- corpus(relationship_df_selected, text_field = "text", docid_field = "id")

# Tokenize, remove punctuation, numbers, and stopwords
toks <- tokens(
    corp,
    remove_punct = TRUE,
    remove_numbers = TRUE
) %>%
    tokens_tolower() %>%
    tokens_remove(stopwords("en"))

# Build DFM
dfmat <- dfm(toks)


# Top 200 features by frequency
top_words <- topfeatures(dfmat, 200)

# Convert to data frame for plotting
top_words_df <- data.frame(
    word = names(top_words),
    frequency = as.numeric(top_words)
)

# Save to CSV
write_csv(top_words_df, "top_words.csv")

## Graphing Top Words (over all submissions)

library(ggplot2)

## Top 20
top_words_df %>%
    slice_max(frequency, n = 20) %>%
    ggplot(aes(x = reorder(word, frequency), y = frequency)) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    labs(title = "Top 20 Words", x = "Word", y = "Frequency") +
    theme_minimal()

## Word Cloud of Top 200

library(quanteda.textplots)
library(wordcloud2)
library(quanteda.textstats)

# Trim low-frequency words to reduce clutter
dfmat_trimmed <- dfm_trim(dfmat, min_termfreq = 10)

# Use top 200 features
word_freq <- textstat_frequency(dfmat_trimmed, n = 200) %>%
    select(feature, frequency)

# Plot with wordcloud2
wordcloud2(word_freq)

### Examining bigrams

# Tokenize into bigrams
toks_bigrams <- tokens(
    corpus(relationship_df_selected, text_field = "text", docid_field = "id"),
    remove_punct = TRUE,
    remove_numbers = TRUE
) %>%
    tokens_tolower() %>%
    tokens_remove(stopwords("en")) %>%
    tokens_ngrams(n = 2)

dfmat_bigrams <- dfm(toks_bigrams)

top_bigrams <- topfeatures(dfmat_bigrams, 200)
print(top_bigrams)

## Making a word cloud of bigrams

# Get top 100 bigrams
bigram_freq <- textstat_frequency(dfmat_bigrams, n = 100) %>%
    select(feature, frequency)

# Plot
wordcloud2(bigram_freq)


## Sentiment Analysis

library(dplyr)

## Ensure correctly formatted date column
## NOTE: If created_utc is Unix seconds, convert with lubridate::as_datetime() then as.Date()

relationship_df_selected <- relationship_df_selected %>%
    mutate(date = as.Date(created_utc))

library(sentimentr)

## Extract sentiment scores and save to df
sentiment_scores <- sentiment(relationship_df_selected$text)

write.csv(sentiment_scores, "sentiment_scores.csv", row.names = FALSE)

saveRDS(sentiment_scores, "sentiment_scores.rds")


## Post level sentiment

library(dplyr)

# Aggregate sentiment scores to the post level
post_level_sentiment <- sentiment_scores %>%
    group_by(element_id) %>%
    summarise(
        mean_sentiment = mean(sentiment, na.rm = TRUE),
        sd_sentiment = sd(sentiment, na.rm = TRUE),
        n_sentences = n()
    )


# Connecting back to original dataframe for covariates

relationship_df_sentiment <- relationship_df_selected %>%
    mutate(element_id = row_number()) %>%
    left_join(post_level_sentiment, by = "element_id")


## Plotting

library(dplyr)
library(ggplot2)

# Make sure created_utc is a Date object
# NOTE: If created_utc is Unix seconds, convert with lubridate::as_datetime() then as.Date()

relationship_df_sentiment <- relationship_df_sentiment %>%
    mutate(created_day = as.Date(created_utc))

# Group by day and calculate average sentiment
daily_sentiment <- relationship_df_sentiment %>%
    group_by(created_day) %>%
    summarise(
        avg_sentiment = mean(mean_sentiment, na.rm = TRUE),
        n_posts = n()
    )


ggplot(daily_sentiment, aes(x = created_day, y = avg_sentiment)) +
    geom_line(color = "steelblue") +
    geom_smooth(method = "loess", se = FALSE, color = "darkred", linetype = "dashed") +
    labs(
        title = "Average Sentiment Over Time",
        x = "Date",
        y = "Average Sentiment"
    ) +
    theme_minimal()


### Adding post volume to plot

library(ggplot2)
library(scales)

ggplot(daily_sentiment, aes(x = created_day)) +
    # Sentiment line (left y-axis)
    geom_line(aes(y = avg_sentiment), color = "steelblue", size = 1) +

    # Post volume line (right y-axis), rescaled
    geom_line(aes(y = rescale(n_posts, to = range(avg_sentiment))), color = "orange", linetype = "dashed", size = 1) +

    scale_y_continuous(
        name = "Average Sentiment",
        sec.axis = sec_axis(~ rescale(., from = range(daily_sentiment$avg_sentiment), to = range(daily_sentiment$n_posts)),
                            name = "Number of Posts")
    ) +

    labs(
        title = "Daily Sentiment and Post Volume",
        x = "Date"
    ) +
    theme_minimal() +
    theme(
        axis.title.y.right = element_text(color = "orange"),
        axis.title.y.left = element_text(color = "steelblue")
    )



### 7 day rolling averages for smoothing

library(dplyr)
library(zoo)

daily_sentiment_smoothed <- daily_sentiment %>%
    arrange(created_day) %>%
    mutate(
        avg_sentiment_7d = rollmean(avg_sentiment, k = 7, fill = NA, align = "right"),
        n_posts_7d = rollmean(n_posts, k = 7, fill = NA, align = "right")
    )


library(ggplot2)
library(scales)

ggplot(daily_sentiment_smoothed, aes(x = created_day)) +
    # Smoothed Sentiment (left y-axis)
    geom_line(aes(y = avg_sentiment_7d), color = "steelblue", size = 1) +

    # Smoothed Post Volume (right y-axis), rescaled
    geom_line(aes(y = rescale(n_posts_7d, to = range(avg_sentiment_7d, na.rm = TRUE))),
              color = "orange", linetype = "dashed", size = 1) +

    scale_y_continuous(
        name = "7-Day Avg Sentiment",
        sec.axis = sec_axis(
            ~ rescale(., from = range(daily_sentiment_smoothed$avg_sentiment_7d, na.rm = TRUE),
                      to = range(daily_sentiment_smoothed$n_posts_7d, na.rm = TRUE)),
            name = "7-Day Avg Post Volume"
        )
    ) +
    labs(
        title = "7-Day Smoothed Sentiment and Post Volume Over Time",
        x = "Date"
    ) +
    theme_minimal() +
    theme(
        axis.title.y.left = element_text(color = "steelblue"),
        axis.title.y.right = element_text(color = "orange")
    )



### Plotting by Subreddit

library(dplyr)
library(ggplot2)

# Check the input data has subreddit and created_day columns
# Create daily average sentiment per subreddit
daily_sentiment <- relationship_df_sentiment %>%
    filter(!is.na(mean_sentiment)) %>%
    group_by(subreddit, created_day) %>%
    summarise(mean_sentiment = mean(mean_sentiment, na.rm = TRUE), .groups = "drop")

# Confirm structure
str(daily_sentiment)

# Plot with facets
ggplot(daily_sentiment, aes(x = created_day, y = mean_sentiment)) +
    geom_line(color = "steelblue") +
    facet_wrap(~ subreddit, ncol = 2, scales = "fixed") +
    labs(
        title = "Daily Average Sentiment by Subreddit",
        x = "Date",
        y = "Mean Sentiment"
    ) +
    theme_minimal()


## Running 7 day average

library(dplyr)
library(ggplot2)
library(zoo)

# Ensure data is sorted by subreddit and date
daily_sentiment_smoothed <- relationship_df_sentiment %>%
    filter(!is.na(mean_sentiment)) %>%
    group_by(subreddit, created_day) %>%
    summarise(mean_sentiment = mean(mean_sentiment, na.rm = TRUE), .groups = "drop") %>%
    arrange(subreddit, created_day) %>%
    group_by(subreddit) %>%
    mutate(sentiment_7day = rollmean(mean_sentiment, k = 7, fill = NA, align = "right")) %>%
    ungroup()

# Plot
ggplot(daily_sentiment_smoothed, aes(x = created_day, y = sentiment_7day)) +
    geom_line(color = "steelblue") +
    facet_wrap(~ subreddit, ncol = 2, scales = "fixed") +
    labs(
        title = "7-Day Smoothed Sentiment by Subreddit",
        x = "Date",
        y = "7-Day Average Sentiment"
    ) +
    theme_minimal()


### STM Topic Model -- subreddit and time covariate; First, grab a sample

library(dplyr)

set.seed(123)

# Add a weekly bin
relationship_df_sentiment <- relationship_df_sentiment %>%
    mutate(week = cut(created_day, breaks = "1 week"))

# Count how many combinations exist
cell_counts <- relationship_df_sentiment %>%
    count(subreddit, week)

# Calculate number to sample per group
# Choosing equal allocation for now
target_per_cell <- ceiling(10000 / nrow(cell_counts))

# Stratified sample
relationship_df_sample <- relationship_df_sentiment %>%
    group_by(subreddit, week) %>%
    slice_sample(n = target_per_cell, replace = FALSE) %>%
    ungroup()

# Trim to exactly 10,000
relationship_df_sample <- relationship_df_sample %>%
    slice_sample(n = 10000)


### checking balance

table(relationship_df_sample$subreddit)


### Run STM on Sample

library(stm)
library(quanteda)
library(dplyr)
library(tidytext)


# Create corpus
corp <- corpus(relationship_df_sample, text_field = "text")

# Tokenize and clean
toks <- tokens(corp,
               remove_punct = TRUE,
               remove_symbols = TRUE,
               remove_numbers = TRUE) %>%
    tokens_tolower() %>%
    tokens_remove(stopwords("en")) %>%
    tokens_wordstem()

# Create dfm and trim rare words
dfmat <- dfm(toks) %>%
    dfm_trim(min_termfreq = 10, min_docfreq = 5)


# Convert dfm to stm input
out <- convert(dfmat, to = "stm")

# Add metadata
meta <- relationship_df_sample %>%
    select(subreddit, week) %>%
    mutate(week = as.numeric(as.factor(week)))  # Convert week to numeric

relationship_df_sample %>%
    summarise(
        subreddit_missing = sum(is.na(subreddit)),
        week_missing = sum(is.na(week)),
        total_rows = n()
    )

str(relationship_df_sample$subreddit)
str(relationship_df_sample$week)


str(out$meta)
length(out$documents)
length(out$meta$docnames)


out$meta$week_num <- as.numeric(out$meta$week)
stm_model <- stm(
    documents = out$documents,
    vocab = out$vocab,
    data = out$meta,
    K = 30,
    prevalence = ~ subreddit + s(week_num),
    max.em.its = 75,
    verbose = TRUE,
    seed = 123
)

labelTopics(stm_model, n = 10)
plot(stm_model, type = "summary", n = 10)

prep <- estimateEffect(
    1:30 ~ subreddit + s(week_num),
    stmobj = stm_model,
    metadata = out$meta,
    uncertainty = "Global"
)

#Plotting by subreddit:
plot(prep, covariate = "subreddit", topics = c(1, 2, 3),
         model = stm_model, method = "difference",
         cov.value1 = "relationship_advice", cov.value2 = "BreakUps",
         xlab = "More BreakUps ←→ More relationship_advice",
         main = "Topic prevalence by subreddit", labeltype = "custom")


#Plotting over time:
    plot(prep, covariate = "week_num", topics = c(1, 2, 3),
         model = stm_model, method = "continuous",
         xlab = "Week", ylab = "Expected topic proportion",
         main = "Topic trends over time")


## Save model

saveRDS(stm_model, file = "stm_model_30topics.rds")


### Popular topics by subreddit

library(dplyr)
library(tidyr)

# Combine topic proportions with metadata
topic_props <- as.data.frame(stm_model$theta)
colnames(topic_props) <- paste0("Topic", 1:30)
topic_props$subreddit <- out$meta$subreddit

# Average topic proportions by subreddit
topic_summary <- topic_props %>%
    group_by(subreddit) %>%
    summarise(across(starts_with("Topic"), mean)) %>%
    pivot_longer(cols = starts_with("Topic"),
                 names_to = "Topic",
                 values_to = "Avg_Proportion") %>%
    group_by(subreddit) %>%
    slice_max(order_by = Avg_Proportion, n = 5)

topic_summary

## Label Topics

labelTopics(stm_model, n = 10)  # Top 10 words per topic

## Find Thoughts

top_n_topics <- c(1, 5, 12)

texts_matched <- quanteda::texts(corp)[quanteda::docnames(dfmat)]

# Finding representative docs
thoughts <- findThoughts(
    model = stm_model,
    texts = texts_matched,
    topics = top_n_topics,
    n = 5
)

# View the results
thoughts$docs


