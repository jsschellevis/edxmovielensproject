# EDX DATA SCIENCE PROGRAM CAPSTONE COURSE
# MovieLens project
# Code by Jebbe Schellevis - Sept 2021

# INTRODUCTION
#
# The code consists of 5 sections:
# - 0 installation and loading of required libraries
# - 1 downloading, loading and cleansing of dataset
# - 2 splitting dataset into validation, train and test sets
# - 3 development and evaluation of different machine learning models
# - 4 final assessment of selected model on validation set and calculating resulting RMSE
#
# The full repository including code (this file), R-markup file (Rmd), resulting PDF report, 
# and intermediate data file can also be found on GitHub: https://github.com/jsschellevis/edxmovielens 


#### 0 Install and load libraries ####

# Install libraries as needed
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(anytime)) install.packages("anytime", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

# Load libraries once installed
library(tidyverse)
library(caret)
library(data.table)
library(knitr)
library(anytime)
library(lubridate)

#### 1 Download and load main data sets ####

# MovieLens 10M dataset origin:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Create temp file and download file into it
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Read ratings file and create data table with file contents
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

# Read movies file and create matrix with file contents
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)

# Set column names for movie matrix
colnames(movies) <- c("movieId", "title", "genres")

# Convert movie matrix to data frame and convert columns to correct types
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

# Join ratings and movies sets into single data frame
movielens <- left_join(ratings, movies, by = "movieId")

# Create final validation set with 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Save data.frames into rdata file
save(edx, validation, file = "data.RData")

# Remove temp variables for cleanup
rm(dl, ratings, movies, test_index, temp, movielens, removed)


### 2 Split edx working set into train and test sets ####

# Create train and test sets 
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.5, list = FALSE)
train <- edx[-test_index,]
test <- edx[test_index,]

# Remove edx set
rm(edx)

# Show dimensions for train and test sets
dim(train)
dim(test)

# Remove index variable
rm(test_index)

### 3 Initial training of different algorithms ####

# Define results table to show results
model_results <- data.frame()

# Train M1 - Movie as predictor

  # Calculate overall average for all movies
  overall_avg_rating <- mean(train$rating)

  # Calculate average difference per movie from overall average rating
  movie_avgs <- train %>% 
    group_by(movieId) %>% 
    summarize(movie_effect = mean(rating - overall_avg_rating))
  
  # Predict ratings for test set
  m1_predicted_ratings <- overall_avg_rating + test %>% 
    left_join(movie_avgs, by='movieId') %>%
    pull(movie_effect)
  
  # Calculate RMSE for M1
  m1_rmse <- RMSE(m1_predicted_ratings, test$rating, na.rm = TRUE)
  
  # Store RMSE in results table
  model_results <- bind_rows(model_results, data_frame(Model="M1 - Movie as predictor", RMSE = m1_rmse))
  
  # Remove predictions
  rm(m1_predicted_ratings)
  
# Train M2 - Movie and user as predictors
  
  # Calculate average difference per user from average rating per movie
  user_avgs <- train %>% 
    left_join(movie_avgs, by='movieId') %>%
    group_by(userId) %>%
    summarize(user_effect = mean(rating - overall_avg_rating - movie_effect))
  
  # Predict ratings for test set
  m2_predicted_ratings <- test %>% 
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    mutate(pred = overall_avg_rating + movie_effect + user_effect) %>%
    pull(pred)
  
  # Calculate RMSE for M2
  m2_rmse <- RMSE(m2_predicted_ratings, test$rating, na.rm = TRUE)
  
  # Store RMSE in results table
  model_results <- bind_rows(model_results, data_frame(Model="M2 - Movie and user as predictors", RMSE = m2_rmse))
  
  # Remove predictions
  rm(m2_predicted_ratings)
    
# Train M3 - Movie, user and age as predictors
  
  # Derive movie age from timestamp and movie year (for train and test sets and validation set)
  # Regex pattern looks for opening parenthesis '(', then 4 digits, then closing parenthesis ')'
  train <- train %>% mutate(movie_year = as.numeric(str_extract(title, "(?<=\\()\\d{4}(?=\\))")), movie_age = str_c(movieId, '-', year(anytime(train$timestamp))-movie_year))
  test <- test %>% mutate(movie_year = as.numeric(str_extract(title, "(?<=\\()\\d{4}(?=\\))")), movie_age = str_c(movieId, '-', year(anytime(timestamp))-movie_year))
  validation <- validation %>% mutate(movie_year = as.numeric(str_extract(title, "(?<=\\()\\d{4}(?=\\))")), movie_age = str_c(movieId, '-', year(anytime(timestamp))-movie_year))
  
  # Calculate average difference per user from average rating per movie
  age_avgs <- train %>% 
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    group_by(movie_age) %>%
    summarize(age_effect = mean(rating - overall_avg_rating - movie_effect - user_effect))
  
  # Visualize age effect for example movie
    # Get age effect factors for The Net from set (filter using movieId followed by dash)
    example_agefactors <- age_avgs %>% filter(substr(movie_age, 1, 4) == str_c("185-"))
    
    # Extract age from combined movie-age column
    example_agefactors <- separate(example_agefactors, movie_age, c(NA, "age"), remove=TRUE)
    
    # Set as numeric and sort
    example_agefactors <- example_agefactors %>% mutate(age = as.numeric(age)) %>% arrange(age)
    
    # Plot age effect
    example_agefactors %>% ggplot(aes(age, age_effect)) + geom_line() + ylim(-0.5,0.5) + scale_x_continuous(breaks = 1:14) + geom_hline(yintercept = 0, col = "grey") + labs(x = "Age (years)", y = "Age effect on predicted rating")
    
    # Plot age effect
    example_agefactors %>% ggplot(aes(age, age_effect)) + geom_line() + ylim(-0.5,0.5) + scale_x_continuous(breaks = 1:14) + geom_hline(yintercept = 0, col = "grey") + labs(x = "Age (years)", y = "Age effect on predicted rating", title = "Age effect for movie The Net (1995)")
  
  # Predict ratings for test set
  m3_predicted_ratings <- test %>% 
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    left_join(age_avgs, by='movie_age') %>%
    mutate(pred = overall_avg_rating + movie_effect + user_effect + age_effect) %>%
    pull(pred)
  
  # Calculate RMSE for M3
  m3_rmse <- RMSE(m3_predicted_ratings, test$rating, na.rm = TRUE)
  
  # Store RMSE in results table
  model_results <- bind_rows(model_results, data_frame(Model="M3 - Movie, user and age as predictors", RMSE = m3_rmse))
  
  # Remove predictions
  rm(m3_predicted_ratings)
  
# Train M4 - Movie, user and age as predictors (regularized)

  # Define function to calculate M4 (M3 regularized with given lambda value)
  m4_model <- function(l) {
  
    # Calculate average difference per movie from overall average rating, but now regularized
    movie_avgs <- train %>% 
      group_by(movieId) %>% 
      summarize(movie_effect = sum(rating - overall_avg_rating)/(n()+l))
  
    # Calculate average difference per user from average rating per movie
    user_avgs <- train %>% 
      left_join(movie_avgs, by='movieId') %>%
      group_by(userId) %>%
      summarize(user_effect = sum(rating - overall_avg_rating - movie_effect)/(n()+l))
    
    # Calculate average difference per user from average rating per movie
    age_avgs <- train %>% 
      left_join(movie_avgs, by='movieId') %>%
      left_join(user_avgs, by='userId') %>%
      group_by(movie_age) %>%
      summarize(age_effect = sum(rating - overall_avg_rating - movie_effect - user_effect)/(n()+l))
  
    # Predict ratings for test set
    m4_predicted_ratings <- test %>% 
      left_join(movie_avgs, by='movieId') %>%
      left_join(user_avgs, by='userId') %>%
      left_join(age_avgs, by='movie_age') %>%
      mutate(pred = overall_avg_rating + movie_effect + user_effect + age_effect) %>%
      pull(pred)
    
    # Calculate and return RMSE
    return(RMSE(m4_predicted_ratings, test$rating, na.rm = TRUE))
  
  }
  
  # Set lambda values for tuning between 0 and 10 with 0.25 steps
  lambda_tuning <- seq(0,10,0.25)
  
  # Tune M4 model using lambda values
  m4_tuning_rmse <- sapply(lambda_tuning, m4_model)

  # Show relation between RMSE and lambda
  qplot(lambda_tuning, m4_tuning_rmse)
  
  # Return lowest RMSE with M4
  min(m4_tuning_rmse)
  
  # Return lambda value that yields lowest RMSE with M4
  l = lambda_tuning[which.min(m4_tuning_rmse)]
  
  # Run M4 model again with tuned lambda value
  m4_rmse <- m4_model(l)
  
  # Store RMSE in results table
  model_results <- bind_rows(model_results, data_frame(Model="M4 - Movie, user and age as predictors (regularized)", RMSE = m4_rmse))
  

### 4 Final assessment on validation set ####

  # Calculate average difference per movie from overall average rating, but now regularized
  movie_avgs <- train %>% 
    group_by(movieId) %>% 
    summarize(movie_effect = sum(rating - overall_avg_rating)/(n()+l))
  
  # Calculate average difference per user from average rating per movie
  user_avgs <- train %>% 
    left_join(movie_avgs, by='movieId') %>%
    group_by(userId) %>%
    summarize(user_effect = sum(rating - overall_avg_rating - movie_effect)/(n()+l))
  
  # Calculate average difference per user from average rating per movie
  age_avgs <- train %>% 
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    group_by(movie_age) %>%
    summarize(age_effect = sum(rating - overall_avg_rating - movie_effect - user_effect)/(n()+l))
  
  # Predict ratings for validation set
  m4_predicted_ratings_validation <- validation %>% 
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    left_join(age_avgs, by='movie_age') %>%
    mutate(pred = overall_avg_rating + movie_effect + user_effect + age_effect) %>%
    pull(pred)
  
  # Calculate and return RMSE
  m4_rmse_validation <- RMSE(m4_predicted_ratings_validation, validation$rating, na.rm = TRUE)
