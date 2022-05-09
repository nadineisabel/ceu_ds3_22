#Cleaning environment and loading relevant packages
rm(list = ls())
library(tidyverse)
library(tidytext)
library(glmnet)
library(dplyr)
library(stringr)
library(ggplot2)
library(stopwords)
library(rtweet)
library(wordcloud)

#Authenticating via a browser
twitter_token <- create_token(
  app = app_name,
  consumer_key = api_key,
  consumer_secret = api_secret_key)

#Saving token
my_authorization <- rtweet::create_token(
  app = app_name,
  consumer_key = api_key,
  consumer_secret = api_secret_key,
  access_token = access_token,
  access_secret = access_token_secret)

#List of twitter usernames: women, men, and all
women_tech_leaders <- c("GinniRometty",
                        "SusanWojcicki",
                        "sherylsandberg",
                        "aileenlee",
                        "mer__edith",
                        "annewoj23",
                        "kirstenagreen",
                        "6Gems",
                        "xeeliz",
                        "ekp")

men_tech_leaders <- c("JeffBezos",
                      "elonmusk",
                      "tim_cook",
                      "satyanadella",
                      "JackMa",
                      "PalmerLuckey",
                      "Benioff",
                      "BobSwan",
                      "stewart",
                      "ajassy")

all_tech_leaders <- c(women_tech_leaders, men_tech_leaders)

#Calling Twitter API with get_timeline function
all_tweets <- get_timeline(all_tech_leaders, n=10000, max_id = NULL,
                           check = FALSE, parse = T, token = my_authorization, 
                           include_rts = TRUE)

#Applying gender label
all_tweets$gender <- with(all_tweets, 
                          ifelse(screen_name %in% women_tech_leaders, 
                                 "woman", "man"))

#Now let's save the precious API calls
all_tweets_export <- apply(all_tweets, 2, as.character)
write.csv(all_tweets_export,"ds3/all_tweets_export.csv")

#Load data back to avoid rerunning API
all_tweets <- read_csv("https://raw.githubusercontent.com/nadineisabel/ceu_ds3_22/main/data/all_tweets_export.csv")

#Now we can move onto cleaning the text for sentiment analysis. Let's clean the text column first with a function.
clean_tweets <- function(x) {
  x %>%
    str_remove_all(" ?(f|ht)(tp)(s?)(://)(.*)[.|/](.*)") %>%
    str_remove_all("@[[:alnum:]_]{4,}") %>%
    str_remove_all("#[[:alnum:]_]+") %>%
    str_replace_all("&amp;", "and") %>%
    str_remove_all("[[:punct:]]") %>%
    str_remove_all("^RT:? ") %>%
    str_replace_all("\\\n", " ") %>%
    str_to_lower() %>%
    str_trim("both") 
}
all_tweets$text <- all_tweets$text %>% clean_tweets

#Let's only select the columns we need for speed of processing.
all_tweets_final <- all_tweets[, c("text", "screen_name", "gender", 
                                   "status_id")]

#Now let's create a new data frame with words as tokens for analysis
words <- all_tweets_final %>% unnest_tokens(word, text) %>% 
  anti_join(stop_words, by = "word") %>% anti_join((data.frame(word = c("im"))), by = "word")
  mutate(tweet_by_woman = as.integer(gender == "woman")) %>% distinct()

#Finally, let's create a wide data set with these words to do some fun analysis
tweets <- words %>%
  group_by(status_id, word, tweet_by_woman) %>%
  summarise(contains = 1) %>%
  ungroup() %>%
  spread(key = word, value = contains, fill = 0) %>% 
  select(-status_id)

#Before we move onto detecting sentiment, let's see if we can model authorship based on gender
fit <- cv.glmnet(x = tweets %>% select(-tweet_by_woman) %>% as.matrix(),
                 y = tweets$tweet_by_woman, family = "binomial")

temp <- coef(fit, s = exp(-3.5)) %>% as.matrix()
coefficients <- data.frame(word = row.names(temp), beta = temp[, 1])
coef_data <- coefficients %>%
  filter(beta != 0) %>%
  filter(word != "(Intercept)") %>%
  arrange(desc(beta)) %>%
  mutate(i = row_number())

#Let's save this data so we don't have to rerun...
write.csv(coef_data,"ds3/coef_data_export.csv")
coef_data <- read_csv("https://raw.githubusercontent.com/nadineisabel/ceu_ds3_22/main/data/coef_data_export.csv")

#Let's visualize it.
coefficent_plot <- ggplot(coef_data, aes(x = i, y = beta, 
                                         fill = ifelse(beta > 0, "Women Tech Leaders", "Men Tech Leaders"))) +
  geom_bar(stat = "identity", alpha = 0.75) +
  scale_x_continuous(breaks = coef_data$i, labels = coef_data$word, 
                     minor_breaks = NULL) +
  xlab("") +
  ylab("Coefficient Estimate") +
  coord_flip() +
  scale_fill_manual(
    guide = guide_legend(title = "Word Typically Used By:"),
    values = c("indianred1", "turquoise3") ) +
  theme_bw() +
  theme(legend.position = "top")
coefficent_plot

#What about word clouds for the genders?
#Top 15 Tweet Words from Men Tech Leaders
words %>%
  filter(gender == "man") %>%
  count(word) %>% 
  with(wordcloud(word, n, max.words = 15, random.order=FALSE, rot.per=0.25,
                 colors=brewer.pal(8, "Paired")))

#Top 15 Tweet Words from Women Tech Leaders
words %>%
  filter(gender == "woman") %>%
  count(word) %>% 
  with(wordcloud(word, n, max.words = 15, random.order=FALSE, rot.per=0.25,
                 colors=brewer.pal(8, "Paired")))

#Let's move onto sentiment analysis. I will use the AFINN dictionary.
afinn <- words %>% 
  inner_join(get_sentiments("afinn")) %>% 
  mutate(method = "AFINN")

#Visualizing the sentiment
afinn_to_plot1 <- afinn %>% 
  group_by(gender, value) %>% 
  summarize(Count = n()) %>%
  group_by(gender) %>%
  summarise(average_afinn = weighted.mean(value, Count))

afinn_plot_gender <- ggplot(afinn_to_plot1, aes(gender, average_afinn, 
                                                fill = gender)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~gender, ncol = 2, scales = "free_x")+
  xlab("Gender") + 
  ylab("Weighted Average AFINN Lexicon Score") + 
  ggtitle("Weighted Average AFINN Lexicon Score \n by Gender") +
  theme(plot.title = element_text(hjust = 0.5), axis.ticks.x=element_blank(),
        axis.text.x=element_blank())
afinn_plot_gender

afinn_to_plot2 <- afinn %>% 
  group_by(screen_name, gender) %>% 
  summarise(average_afinn = mean(value))

afinn_plot_screen_names <- ggplot(afinn_to_plot2, aes(screen_name, 
                                                      average_afinn, 
                                                      fill = gender)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~gender, ncol = 2, scales = "free_x") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Twitter Users") + 
  ylab("Average AFINN Lexicon Score") +
  ggtitle("Average AFINN Lexicon Score by \n Twitter Screen Names") +
  theme(plot.title = element_text(hjust = 0.5))
afinn_plot_screen_names

###
