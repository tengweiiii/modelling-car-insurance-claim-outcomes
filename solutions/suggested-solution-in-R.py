# Import required libraries
library(readr)
library(dplyr)

# Read in dataset
cars = read_csv('car_insurance.csv')

# View data types
str(cars)

# Create a dataframe to store features
# Tip: A faster way of achieving the same result here is to use a combination of names(), subset() and select()
# features_df <- data.frame(features = c(names(subset(cars, select = -c(id, outcome)))))
features_df <- data.frame(features = c('age', 'driving_experience', 'education', 'income', 'credit_score', 'vehicle_ownership', 'vehicle_year', 'married', 'children', 'postal_code', 'annual_mileage', 'vehicle_type', 'speeding_violations', 'duis', 'past_accidents'))

# Repeat this step for all features in features_df
# Create a model
model <- glm('outcome ~ driving_experience', data = cars, family = 'binomial')
# Get prediction values for the model
predictions <- round(fitted(model))
# Calculate accuracy
accuracy <- length(which(predictions == cars$outcome)) / length(cars$outcome)
# Add accuracy to features_df
features_df[which(features_df$features == 'driving_experience'), "accuracy"] = accuracy

# Alternatively, use a for loop to make this training more efficient, looping through all of the features at once with the use of glue()
# library(glue)
# for (col in features_df$features) {
#     # Create a model
#     model <- glm(glue('outcome ~ {col}'), data = cars, family = 'binomial')
#     # Get prediction values for the model
#     predictions <- round(fitted(model))
#     # Calculate accuracy
#     accuracy <- length(which(predictions == cars$outcome)) / length(cars$outcome)
# 	# Add accuracy to features_df
# 	features_df[which(features_df$feature == col), "accuracy"] = accuracy
# }

# Find the feature with the largest accuracy
best_feature_df <- features_df %>% slice_max(order_by = accuracy, n = 1)

# Run in a new cell to check your solution
best_feature_df
