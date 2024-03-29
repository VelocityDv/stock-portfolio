library(ggplot2)

# Read the CSV file
data <- read.csv("data/alibaba.csv")
data$Date <- as.Date(data$Date)

BABA_HK = data$X9988.HK
BABA_NYSE = data$BABA

combined_data <- data.frame(Date = data$Date, BABA_HK = data$X9988.HK, BABA_NYSE = data$BABA)

library(tidyr)
combined_data_long <- pivot_longer(combined_data, cols = c(BABA_HK, BABA_NYSE), names_to = "Ticker", values_to = "Price")

ggplot(combined_data_long, aes(x = Date, y = Price, color = Ticker)) +
  geom_line() +
  labs(title = "Time Series Comparison", x = "Date", y = "Price") +
  theme_minimal()

x = BABA_HK - BABA_NYSE
plot(x, type = "l", xlab = "Index", ylab = "Difference", main = "Difference between BABA_HK and BABA_NYSE")

cov(BABA_HK, BABA_NYSE)

missing_rows <- is.na(BABA_HK) | is.na(BABA_NYSE)
data_cleaned_BABA <- data[!missing_rows, ]


cleaned_BABA_HK = data_cleaned_BABA$X9988.HK
cleaned_BABA_NYSE = data_cleaned_BABA$BABA

cov(cleaned_BABA_HK, cleaned_BABA_NYSE)

