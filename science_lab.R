# Enterprise Data Science Lab
# Analysis of Farm Health Trends

data <- read.csv("database/surveys.csv")

# Calculate Correlation Matrix
correlation <- cor(data$health_score, data$water_index)
print(paste("Correlation Coeff:", correlation))

# Linear Regression Model
model <- lm(yield ~ health_score + age, data=data)
summary(model)

# Plot Heatmap (Requires ggplot2)
library(ggplot2)
ggplot(data, aes(x=lon, y=lat, fill=health_score)) + geom_tile()
