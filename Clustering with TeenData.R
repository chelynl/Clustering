# Segmentation analysis of teenagers
library(rpart)
library(psych,ggplot2) #pairs.panels
library(cluster) # Gap statistic of Tibshirani et al
library(tidyverse)
library(magrittr)
library(flashClust)
library(NbClust)
library(clValid)
library(ggfortify)
library(clustree)
library(dendextend)
library(factoextra)
library(FactoMineR)
library(rpart) # for building decision trees
library(lattice) # for better plotting defaults of multivariate relationships
library(rattle) # fancy tree plot
library(rpart.plot) # enhanced tree plots
library(RColorBrewer) # color selection for fancy tree plot
library(party) # alternative decision tree algorithm
library(partykit) # convert rpart object to BinaryTree

# Import Data
load('C:/Users/chely/Documents/Fall 2/Data Mining/Data/TeenSNS.RData')

#----------------------- EDA: see distributions and if normalization is needed ------------------------------#

# Remove cluster column from data
teens$cluster <- NULL

# See summary statistics of data
summary(teens)

# Vector of 37 terms used in status updates by teens
terms <- c("friends", "basketball", "football", "soccer", "softball", "volleyball", 
           "swimming","cheerleading", "baseball", "tennis", "sports", "cute", "sex", 
           "sexy", "hot", "kissed", "dance", "band", "marching", "music", "rock", "god", 
           "church", "jesus", "bible", "hair", "dress", "blonde", "mall", "shopping", 
           "clothes", "hollister", "abercrombie", "die", "death", "drunk", "drugs")

# Demographic attributes
demographics <- c("gradyear","gender","age", "female","no_gender")

# Only include observations where the sum of all the terms per teen is greater than 0 (remove obs with no terms)
teens <- teens[rowSums(teens[,terms])>0,]

# Observe basic summary stats for the terms
summary(teens[,terms])

# Get 5000 random indices from 1 to 29074 to downsample obs to 5000 because 30K is a lot to draw
samplePoints <- sample(1:29074, 5000, replace=F)

# See scatter plots, histograms, and correlation values for all terms (in small chunks)
pairs.panels(teens[samplePoints,terms[1:5]], method = 'pearson', density=T, ellipses = F)
pairs.panels(teens[samplePoints,terms[6:10]], method = 'pearson', density=T, ellipses = F)
pairs.panels(teens[samplePoints,terms[11:15]], method = 'pearson', density=T, ellipses = F)
pairs.panels(teens[samplePoints,terms[16:20]], method = 'pearson', density=T, ellipses = F)
pairs.panels(teens[samplePoints,terms[21:25]], method = 'pearson', density=T, ellipses = F)
pairs.panels(teens[samplePoints,terms[26:30]], method = 'pearson', density=T, ellipses = F)
pairs.panels(teens[samplePoints,terms[31:37]], method = 'pearson', density=T, ellipses = F)

#----------------- Process the Data: transform, standardize, or normalize input features ------------------#

# Option 1: do nothing, use raw data
teens.norm <- teens[,terms]

# Option 2: standardize by subtracting the mean of each column and dividing by the std of each column
teens.norm <- scale(teens[,terms], center=T, scale=T)

# Option 3: range standardization, putting each variable on a scale from 0 to 1 using fx below
teens.norm <- apply(teens[,terms], 2, function(x)(x-min(x))/(max(x)-min(x)))

# Option 4: standardize each obs so that the rows sum to 1 (represents individual's usage of each word as a proportion of the total number of words they used from this list)
teens.norm <- as.data.frame(t(apply(teens[,terms], 1, function(x)(x/sum(x)))))

# Option 5: do log transformation by transforming each column, x, into log(x + 1) where 1 is added to the 
# argument to prevent the problem that would be associated with computing log(0) which is undefined.
teens.norm <- log(teens[,terms]+1)
                                    
# Option 2 was used for this project                                   

#----------------------------------------- Visualize Processed Data ----------------------------------------#

# After transforming your data, you should repeat EDA to make sure nothing changed
# Consider projection onto principal components (correlation or covariance)
# We want to better understand the structure of the data cloud in space and to inform any decisions regarding the 
# number of clusters or outliers that could affect the progress of clustering algorithms

cov.pca <- prcomp(teens.norm, scale=F) # covariance PCA
cor.pca <- prcomp(teens.norm, scale=T) # correlation PCA

par(mfrow=c(2,1)) # set grid parameters

plot(cov.pca$sdev^2, main = 'Covariance PCA',ylab='Eigenvalue') # scree plot
plot(cov.pca$x) # plot first 2 PCs

plot(cor.pca$sdev^2, main = 'Correlation PCA',ylab='Eigenvalue') # scree plot
plot(cor.pca$x) # plot first 2 PCs

# Explore FURTHER principal components for visualization
par(mfrow=c(3,3),mar=c(4,4,2,1))
plot(cov.pca$x[samplePoints,c(1,3)])
plot(cov.pca$x[samplePoints,c(1,4)])
plot(cov.pca$x[samplePoints,c(1,5)])
plot(cov.pca$x[samplePoints,c(2,3)])
plot(cov.pca$x[samplePoints,c(2,4)])
plot(cov.pca$x[samplePoints,c(2,5)])
plot(cov.pca$x[samplePoints,c(3,4)])
plot(cov.pca$x[samplePoints,c(3,5)])
plot(cov.pca$x[samplePoints,c(4,5)])
mtext("COVARIANCE PCA", side = 3, line = -1.5, outer = TRUE)

par(mfrow=c(3,3),mar=c(4,4,2,1))
plot(cor.pca$x[samplePoints,c(1,3)])
plot(cor.pca$x[samplePoints,c(1,4)])
plot(cor.pca$x[samplePoints,c(1,5)])
plot(cor.pca$x[samplePoints,c(2,3)])
plot(cor.pca$x[samplePoints,c(2,4)])
plot(cor.pca$x[samplePoints,c(2,5)])
plot(cor.pca$x[samplePoints,c(3,4)])
plot(cor.pca$x[samplePoints,c(3,5)])
plot(cor.pca$x[samplePoints,c(4,5)])
mtext("CORRELATION PCA", side = 3, line = -1.5, outer = TRUE)
                                    
#----------------------- Determine Input Data: Reduce Dimensionality Before Clustering ---------------------#

# Choose specific matrix factorization and rank (dimensionality/number of PCs)
# We have 9 different options of matrix factorizations using PCA (corr or cov)
# Number of PCs can be chosen based upon screeplot or cumulative % variance explained

input <- teens.norm # raw normalized data
input <- cov.pca$x[,1:4] # first 4 PCs from covariance PCA on normalized data
input <- cor.pca$x[,1:7] # first 7 PCs from correlation PCA on normalized data-- first 7 PCs seem useful
                                    
# The first 7 PCs from the correlation PCA was used as reduced input data                                    

pca_var <- cor.pca$sdev^2 # convert std to variance to get eigenvalues
pca_var_per <- round(pca_var/sum(pca_var) * 100, 1) # get variance percentages for each PC
barplot(pca_var_per, main="Scree Plot", xlab = "Principle Component", ylab = "Percent Variation") #scree plot

#------------------------------------------- Choose number of clusters -------------------------------------#

# Option 1: look at screeplot (there are 9 different options for that screeplot)

# Option 2: Examine the SSE value for clustering with k = 1, 2, 3,... and look for an elbow in that graph
# The code below shows 10 clusterings for each value of k, and then displays a boxplot of the SSE for each value of k.
obj <- matrix(NA, nrow=10, ncol = 19)
for(k in 2:20){
  iter=1
  while(iter<11){
    obj[iter,(k-1)] = kmeans(input,k)$tot.withinss
    iter=iter+1
    }
}
colnames(obj) <- paste('k',2:20,sep='')
rownames(obj) <- paste('iter',1:10,sep='')
# Use output to create data frame for boxplot visual
obj <- data.frame(obj)
obj2 <- gather(obj,key = 'K',value = 'SSE', k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15,k16,k17,k18,k19,k20 )
obj2$K <- gsub("k",'',obj2$K)
obj2$K <- as.numeric(obj2$K)
par(mfrow=c(1,1))
boxplot(SSE~K,data=obj2, ylab = 'SSE Objective Function', xlab='Number of clusters, k', col='violet', main = 'Box plots of SSE for 10 runs of k-means with each k')

# Option 3: compute the trace(Wq) for various numbers of clusters
# This should decrease monotonically as the number of clusters grows, and we use the maximum of the second differences to determine the number of clusters 
# (i.e. where the slope of the curve is increasing the fastest)

res <- NbClust(input, distance = "euclidean", min.nc=2, max.nc=8, method = "kmeans", index = "tracew")
(k <- res$Best.nc)
clusters <- res$Best.partition # gives me 5 clusters

# Option 4: compute Marriot Index. Let Wq be the within-group dispersion matrix for data clustered into q clusters, as defined in equation 1.
# Look for the maximum difference between successive levels to determine the optimal k
res <- NbClust(input, distance = "euclidean", min.nc=2, max.nc=8, method = "kmeans", index = "marriot")
(k <- res$Best.nc)
clusters <- res$Best.partition

# Option 5: Method of Friedman and Rubin
# Look for the maximum difference in values of this criterion to determine the optimal k
res <- NbClust(input, distance = "euclidean", min.nc=2, max.nc=10, method = "kmeans", index = "friedman")
(k <- res$Best.nc)
clusters <- res$Best.partition

# Option 6: use the Gap Statistic (too slow for 30K+ observations) 
# It's not really a great idea to downsample to employ these methods, but if you'd like to see this type of analysis in action, the cluster package has a nice implementation
samplePoints <- sample(1:29074, 2000, replace=F)

out.gap <- clusGap(input[samplePoints,], kmeans, 10, B = 50, d.power = 2, 
                 spaceH0 = c("scaledPCA"), verbose = interactive())

fviz_gap_stat(out.gap)+
  theme_minimal()+
  ggtitle("fviz_gap_stat: Gap Statistic")

# Option 7: Use the average Silhouette value
fviz_nbclust(input[samplePoints,], kmeans, method = "silhouette", k.max = 10) + 
  theme_minimal() + 
  ggtitle("The Silhouette Plot")

# Option 8: explore 24 metrics that have been proposed in the literature via the Nbclust package. 
# too slow for 30K+ observations, requires storage of pairwise distance matrix
samplePoints <- sample(1:29074, 2000, replace=F)
res <- NbClust(input[samplePoints,], distance = "euclidean", min.nc=2, max.nc=10, 
               method = "kmeans", index = "all")
                                    
# Option 2 was used to determine 7 clusters for k-means                                    

#------------------------------------- Determine Final Clustering Algorithm ----------------------------------#

# K-means
set.seed(11117)
final.clusters <- kmeans(input,7)
teens$cluster <- final.clusters$cluster


# Hierarchical clustering: not recommended for users of R, as it will require in-memory storage of a similarity matrix that is roughly 10gb
tree <- flashClust(dist(input),method='centroid')
teens$cluster <- cutree(tree,k=3)

# I ended up using k-means as my clustering algorithm                                    

#---------------------------------------------- Profile the clusters -----------------------------------------#

# This is where the actionable insights are generated - describe each cluster. 
# Create a profile for each cluster that describes key attributes which differentiate it from the other clusters

# Option 1: Visualize variable distributions, cluster vs. all
clusterProfile = function(df, clusterVar, varsToProfile){
  k = max(df[,clusterVar])
  for(j in varsToProfile){
    if(is.numeric(df[,j])){
      for(i in 1:k){
        hist(as.numeric(df[df[,clusterVar]==i ,j ]), breaks=50, freq=F, col=rgb(1,0,0,0.5), 
             xlab=paste(j), ylab="Density", main=paste("Cluster",i, 'vs all data, variable:',j))
        hist(as.numeric(df[,j ]), breaks=50,freq=F, col=rgb(0,0,1,0.5), xlab="", ylab="Density", add=T)
        
        legend("topright", bty='n',legend=c(paste("cluster",i),'all observations'), 
               col=c(rgb(1,0,0,0.5),rgb(0,0,1,0.5)), pt.cex=2, pch=15 )}
      }
    if(is.factor(df[,j])&length(levels(df[,j]))<5){
      (counts = table( df[,j],df[,clusterVar]))
      (counts = counts%*%diag(colSums(counts)^(-1)))
      barplot(counts, main=paste(j, 'by cluster'), xlab=paste(j),legend = rownames(counts), beside=TRUE)
      }
    }
}

par(mfrow=c(2,2))
clusterProfile(df=teens, clusterVar='cluster', demographics)

# prop.table(table(teens$gender, teens$cluster))


# Option 2: Decision Trees to Predict Cluster
teens$cluster <- as.factor(teens$cluster)
dtree <- rpart(cluster~.,data=teens)
plot(dtree)
text(dtree)

  
fancyRpartPlot(dtree)
prp(dtree, type =3, extra=8) # label branches, label nodes with pred prob of class

