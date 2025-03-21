# === LIBRAIRIES ===
library(data.table)
library(ggplot2)
library(corrplot)
library(pls)
library(caret)
library(factoextra)
library(FactoMineR)



# === CHARGEMENT DES DONNÉES ===
UPENN <- read.table("data/UPENN.txt")
GT <- read.table("data/GT.txt")

# === PRÉTRAITEMENT ===
# On enlève ID (colonne 1) et on applique log sur la variable réponse
X_train <- UPENN[, -1]
X_test <- GT[, -1]

# Variable réponse (target)
Y_train <- log(UPENN$totallight)
Y_test <- log(GT$totallight)

# On centre-réduit
X_train_scaled <- scale(X_train)
X_test_scaled <- scale(X_test, center = attr(X_train_scaled, "scaled:center"), 
                       scale = attr(X_train_scaled, "scaled:scale"))

# === DIMENSIONS ===
n <- nrow(X_train_scaled)
p <- ncol(X_train_scaled)

# === PCR ===
model_pcr <- pcr(Y_train ~ ., data = as.data.frame(X_train_scaled), ncomp = 15, validation = "CV")
validationplot(model_pcr, val.type = "RMSEP", type = "b", main = "PCR - RMSEP CV")
RMSEP_pcr <- RMSEP(model_pcr, estimate = "CV")$val[1,,]
opt_pcr <- which.min(RMSEP_pcr) - 1
cat("Nombre optimal de composantes PCR:", opt_pcr, "\n")

# === PLS ===
model_pls <- plsr(Y_train ~ ., data = as.data.frame(X_train_scaled), ncomp = 15, validation = "CV")
validationplot(model_pls, val.type = "RMSEP", type = "b", main = "PLS - RMSEP CV")
RMSEP_pls <- RMSEP(model_pls, estimate = "CV")$val[1,,]
opt_pls <- which.min(RMSEP_pls) - 1
cat("Nombre optimal de composantes PLS:", opt_pls, "\n")

# === COMPARAISON TEST ===
pred_pcr <- predict(model_pcr, newdata = as.data.frame(X_test_scaled), ncomp = opt_pcr)
pred_pls <- predict(model_pls, newdata = as.data.frame(X_test_scaled), ncomp = opt_pls)

rmsep_test_pcr <- sqrt(mean((Y_test - pred_pcr)^2))
rmsep_test_pls <- sqrt(mean((Y_test - pred_pls)^2))

r2_pcr <- cor(Y_test, pred_pcr)^2
r2_pls <- cor(Y_test, pred_pls)^2

cat("PCR - RMSEP test:", round(rmsep_test_pcr, 4), "- R²:", round(r2_pcr, 4), "\n")
cat("PLS - RMSEP test:", round(rmsep_test_pls, 4), "- R²:", round(r2_pls, 4), "\n")

# === INTERPRÉTATION DU MODÈLE PLS ===
A <- opt_pls
model_final <- plsr(Y_train ~ ., data = as.data.frame(X_train_scaled), ncomp = A, method = "oscorespls")

# Scores
scoreplot(model_final, comp = 1:2, main = "Scores PLS", pch = 19)
abline(h=0, v=0)

# Corrélation variables
pls::corrplot(model_final, comp = 1:2, plotx = TRUE, ploty = TRUE, 
              labels = c(colnames(X_train), "log(totallight)"))

# VIP
vip <- VIP(model_final)[A, ]
barplot(vip, las = 2, cex.names = 0.7, ylab = "VIP")
abline(h = 1, col = "red", lty = 2)

# Coefficients β
beta <- coef(model_final)[, 1, 1]
barplot(beta, las = 2, cex.names = 0.7, ylab = "Coefficients β")

# Jackknife
model_jack <- plsr(Y_train ~ ., data = as.data.frame(X_train_scaled), ncomp = A, 
                   method = "oscorespls", validation = "LOO", jackknife = TRUE)
jack_res <- jack.test(model_jack, ncomp = A)
barplot(jack_res$tvalues, names.arg = colnames(X_train), las = 2,
        ylab = "t-values (jackknife)", cex.names = 0.7)
abline(h = c(-2, 2), col = "red", lty = 2)

