library(readr)
library(caret)
library(dplyr)
library(FactoMineR)

data <- read_csv("data.csv")

data$X1 <- NULL
head(data)

hist(data$num_packets0_1)
hist(data$bw0_1)
hist(data$delay0_1_average)

plot(data$bw0_1, data$delay0_1_average)

df <- select(data, bw0_1, num_packets0_1, drop0_1, delay0_1_average)
df$num_packets0_1 <- scale(df$num_packets0_1)

cor(df)
summary(df)

res.pca = PCA(df, ncp=5, graph=T)
summary(res.pca)


data2 <- read_csv("50nodes_ap.csv")

data2$X1 <- NULL
head(data2)

hist(data2$num_packets0_1, main="Distribution of number of packets")
hist(data2$bw0_1, main="Bandwidth distribution")
hist(data2$delay0_1_average, 20, main="Delay distribution")
hist(data2$drop0_1, 20, main="Drop distribution")


plot(data2$bw0_1, data2$delay0_1_average, main = "Delay vs Bandwidth")
plot(data2$drop0_1, data2$delay0_1_average, main = "Delay vs Drops")


df <- select(data2, bw0_1, num_packets0_1, drop0_1, delay0_1_average)
df$num_packets0_1 <- scale(df$num_packets0_1)

cor(df)
summary(df)

data3 <- read_csv("14_nodes_15_SP_1.csv")

data3$X1 <- NULL
head(data3)

hist(data3$bw0_1, main="Bandwidth distribution")
hist(data3$drop0_1, 20, main="Drop distribution")
hist(data3$delay0_1_average, 20, main="Delay distribution")


plot(data3$bw0_1, data3$delay0_1_average, main = "Delay vs Bandwidth")
plot(data3$drop0_1, data3$delay0_1_average, main = "Delay vs Drops")
df3 <- select(data3, bw0_1, drop0_1, delay0_1_average)
cor(df3)
summary(df3)

res.pca = PCA(df3, ncp=5, graph=T)
summary(res.pca)

data14 <- read_csv("14_nodes_AL_1_K_1.csv")

data14$X1 <- NULL
head(data14)

hist(data14$bw0_1, main="Bandwidth distribution")
hist(data14$delay0_1_average, 20, main="Delay distribution")
hist(data14$drop0_1, 20, main="Drop distribution")


plot(data14$bw0_1, data14$delay0_1_average, main = "Delay vs Bandwidth")
plot(data14$drop0_1, data14$delay0_1_average, main = "Delay vs Drops")
df14 <- select(data14, bw0_1, drop0_1, delay0_1_average)
cor(df14)
summary(df14)

data24 <- read_csv("24_nodes.csv")

data24$X1 <- NULL
head(data24)

hist(data24$bw0_1, main="Bandwidth distribution")
hist(data24$delay0_1_average, 20, main="Delay distribution")
hist(data24$drop0_1, 20, main="Drop distribution")


plot(data24$bw0_1, data24$delay0_1_average, main = "Delay vs Bandwidth")
plot(data24$drop0_1, data24$delay0_1_average, main = "Delay vs Drops")


df24 <- select(data24, bw0_1, drop0_1, delay0_1_average)
cor(df24)
summary(df24)

res.pca = PCA(df24, ncp=5, graph=T)
summary(res.pca)


data_ap_8 <- read_csv("50_nodes_ap_8.csv")

data_ap_8$X1 <- NULL

data_ap_58 <- read_csv("50_nodes_ap_58.csv")
data_ap_58$X1 <- NULL
ds <- rbind(data_ap_8, data_ap_58)

data_ap_11 <- read_csv("50_nodes_ap_2.csv")
data_ap_11$X1 <- NULL
ds <- rbind(ds, data_ap_11)

data_ap_138 <- read_csv("50_nodes_ap_138.csv")

data_ap_138$X1 <- NULL

ds <- rbind(ds, data_ap_138)

data_ap_15 <- read_csv("50_nodes_ap_15.csv")

data_ap_15$X1 <- NULL

ds <- rbind(ds, data_ap_15)

hist(ds$bw0_1, main="Bandwidth distribution ")
hist(ds$delay0_1_average, 40, main="Delay distribution ")
hist(ds$drop0_1, 20, main="Drop distribution ")
hist(ds$num_packets0_1, main="Distribution of number of packets ")


plot(ds$bw0_1, ds$delay0_1_average, main = "Delay vs Bandwidth ")
plot(ds$drop0_1, ds$delay0_1_average, main = "Delay vs Drops")
plot(ds$drop0_1, ds$bw0_1, main = "Delay vs Drops")

ds <- select(ds,  bw0_1, num_packets0_1, drop0_1, delay0_1_average)
cor(ds)
summary(ds)

res.pca = PCA(ds, ncp=5, graph=T)
summary(res.pca)
