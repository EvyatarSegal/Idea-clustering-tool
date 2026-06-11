df <- read.csv("data/processed/test_idea_clustered.csv")



# Add length metrics
df <- df %>%
  mutate(
    char_count = nchar(idea_text),
    word_count = sapply(strsplit(idea_text, "\\s+"), length),
    sentence_count = sapply(strsplit(idea_text, "[.!?]+"), length)
  )

# Summary statistics
cat("\n========== Text Length Statistics ==========\n")
cat(paste("Total rows:", nrow(df), "\n"))
cat(paste("Mean characters:", round(mean(df$char_count), 0), "\n"))
cat(paste("Median characters:", median(df$char_count), "\n"))
cat(paste("Max characters:", max(df$char_count), "\n"))
cat(paste("Min characters:", min(df$char_count), "\n"))
cat("\nWord counts:\n")
cat(paste("Mean words:", round(mean(df$word_count), 1), "\n"))
cat(paste("Max words:", max(df$word_count), "\n"))

# Distribution by bins
cat("\n========== Character Count Distribution ==========\n")
breaks <- c(0, 500, 1000, 1500, 2000, 3000, 5000, Inf)
labels <- c("<500", "500-999", "1000-1499", "1500-1999", "2000-2999", "3000-4999", "≥5000")
df$char_bin <- cut(df$char_count, breaks = breaks, labels = labels, include.lowest = TRUE)
print(table(df$char_bin))

# Longest 5 rows (full text)
cat("\n========== Longest 5 Rows ==========\n")
longest <- df %>% arrange(desc(char_count)) %>% head(5)
for (i in 1:nrow(longest)) {
  cat(paste("\n--- Row ID", longest$id[i], "|", longest$char_count[i], "chars,", longest$word_count[i], "words ---\n"))
  cat(substr(longest$idea_text[i], 1, 500))
  if (nchar(longest$idea_text[i]) > 500) cat("... [truncated]")
  cat("\n")
}

# Optional: export a shortened version for summarization
# Create a new column with first 1000 characters
df$idea_short <- substr(df$idea_text, 1, 1000)
short_path <- file.path("..", "..", "data", "intermediate", "test_ideas_short.csv")
dir.create(dirname(short_path), recursive = TRUE, showWarnings = FALSE)
write.csv(df[, c("id", "idea_short")], short_path, row.names = FALSE)
cat(paste("\nSaved shortened version (first 1000 chars) to:", short_path, "\n"))




df_short <- read.csv("data/intermediate/test_ideas_with_summaries_and_embeddings.csv")
