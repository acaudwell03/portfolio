# ===========================
# Load Libraries
# ===========================
library(tidyverse)   # data manipulation and plotting
library(afex)        # ANOVA with corrections
library(emmeans)     # pairwise comparisons
library(ggsignif)    # significance annotations

# ===========================
# Load and Reshape Data
# ===========================
f1_data <- read.csv('comparison_table.csv')

f1_long <- f1_data %>%
  pivot_longer(
    cols = -X,
    names_to = "Model",
    values_to = "F1"
  ) %>%
  rename(Emotion = X)

# ===========================
# Repeated-Measures ANOVA
# ===========================
aov_res <- aov_ez(
  id = "Emotion",
  dv = "F1",
  within = "Model",
  data = f1_long,
  type = 3,
  return = "afex_aov"
)

# View ANOVA summary
print(aov_res)

# Pairwise comparisons with Holm adjustment
pairwise_res <- emmeans(aov_res, pairwise ~ Model, adjust = "holm")
print(pairwise_res$contrasts)

# ===========================
# Bar Plot with Significance
# ===========================
signif_labels <- c("p = .055", "p = .055", "p = .096")

ggplot(f1_long, aes(x = Model, y = F1, fill = Model)) +
  geom_bar(stat = "summary", fun = mean, width = 0.6, color = "black") +
  geom_signif(
    comparisons = list(
      c("Model.1", "Model.2"),
      c("Model.1", "Model.3"),
      c("Model.2", "Model.3")
    ),
    annotations = signif_labels,
    y_position = c(0.5, 0.55, 0.53),
    tip_length = 0.01,
    textsize = 4
  ) +
  coord_cartesian(ylim = c(0.3, 0.7)) +
  labs(
    y = "Mean F1 Score",
    x = NULL
  ) +
  theme_minimal() +
  theme(legend.position = "none")

# ===========================
# Example Data for Correlation
# ===========================
df <- data.frame(
  Label = c('Admiration', 'Amusement', 'Anger', 'Annoyance', 'Approval', 'Caring',
            'Confusion', 'Curiosity', 'Desire', 'Disappointment', 'Disapproval', 'Disgust', 
            'Embarrassment', 'Excitement', 'Fear', 'Gratitude', 'Grief', 'Joy', 'Love',
            'Nervousness', 'Optimism', 'Pride', 'Realization', 'Relief', 'Remorse', 
            'Sadness', 'Surprise'),
  F1 = c(0.70, 0.84, 0.52, 0.40, 0.41, 0.42, 0.41, 0.43, 0.55, 0.31, 0.39, 0.49, 
         0.59, 0.37, 0.67, 0.92, 0.14, 0.58, 0.81, 0.36, 0.60, 0.63, 0.32, 0.23, 
         0.73, 0.53, 0.57),
  Samples = c(5120, 2894, 1959, 3092, 3678, 1372, 1658, 2690, 801, 1583, 2563, 1012, 
              374, 1049, 764, 3372, 96, 1785, 2574, 208, 1975, 140, 1382, 182, 669, 
              1623, 1328)
)

# ===========================
# Spearman Correlation
# ===========================
spearman_corr <- cor(df$F1, df$Samples, method = "spearman")
cat("Spearman correlation:", round(spearman_corr, 2), "\n")

# ===========================
# Scatter Plot with Regression
# ===========================
ggplot(df, aes(x = Samples, y = F1)) +
  geom_point(color = "steelblue", size = 3) +
  geom_smooth(method = "lm", se = TRUE, color = "darkred") +
  labs(
    x = "Number of Samples",
    y = "F1 Score",
    title = "F1 Score vs Sample Size"
  ) +
  theme_minimal(base_size = 12)
