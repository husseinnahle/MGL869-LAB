# Inclure les dependences necessaires,
library(rms)
library(pROC)
library(ROCR)
library(ScottKnottESD)
library(boot)

library(car)

# Charger les donnees:
data <- read.csv("ant-1.7.csv")
head(data)

# Pre-processing 
data$is_class<-ifelse(data$class_or_interface=="C",1,0)
data$class_or_interface<-NULL
data$has_bug<-as.factor(data$has_bug)
head(data)

# Fonction pour calculer la performance AUC (Area Under Curve), Precision et recall
get_auc<-function(actuals,predicted){
  predictions<-prediction(predicted,actuals)
  auc<-ROCR::performance(predictions,'auc')
  auc<-unlist(slot(auc,'y.values'))
  result_auc<-min(round(auc,digits=2))
  result_auc<-ifelse(result_auc<0.50,1-result_auc,result_auc)
  return(result_auc)
}

#Precision et recall
error_metric<-function(actuals,predicted){ 
  CM <- table(predicted, actuals)
  TN =CM[1,1]
  TP =CM[2,2]
  FN =CM[1,2]
  FP =CM[2,1]
  precision =(TP)/(TP+FP)
  recall=(TP)/(TP+FN)
  my_list <- list("precision" = precision, "recall" = recall)
  return(my_list)    
}

# Analyse de correlation
## Calculer la correlation: 
correlations<-varclus(~ nr_of_getters+nr_of_getters+nr_of_setters+
          nr_of_methods+nr_incoming_calls+
          nr_outgoing_calls+cohesion+is_class, data=data, similarity = "spearman",
          trans = "abs")

plot(correlations)
abline (h = 1 - 0.7 , col = "grey")

## Enlever la correlation: 
data$nr_of_methods<-NULL

# Redondance:
redun_obj = redun(~ nr_of_getters+nr_of_getters+nr_of_setters+
                    nr_incoming_calls+
                    nr_outgoing_calls+cohesion+is_class,
                     data = data, nk = 0)
paste (redun_obj$Out , collapse =",") # Aucune variable redondante


# Generer 100 bootstrap samples

## Fonction qui retounrne les indexes des bootstrap samples 
getBoostrapSamples<-function(dat, idx) {
  return(idx)
}

boot_indices<-boot(data, statistic = getBoostrapSamples, R=100)$t

## Declarer des listes dont lesquelles on va stocker les resultats du modele
AUC<-list()
Precision<-list()
Recall<-list()
interpretation<-list()

# Entrainer et tester 100 modeles. Un modele par sample
for (i in 1:100) {
  # Donnees pour entrainer le modele
  train<-data[boot_indices[i,],]
  # Ce qui n'est pas dans train serait utiliser pour tester le modele
  test<-data[-boot_indices[i,],]
  
  # Entrainer le modele
  logistic_regression_model<-lrm(has_bug~.,data=train)
  
  actuals<-test$has_bug
  test$has_bug<-NULL
  
  # Utiliser le modele pour predire les donnees du test
  predicted<-predict(logistic_regression_model,newdata=test,type='fitted.ind')
  

  # Calculer la performance du modele
  AUC[[i]]<-get_auc(actuals, predicted)
  predicted_values<-ifelse(predicted>0.5,1,0)
  Precision[[i]]<-error_metric(actuals,predicted_values)[[1]]
  Recall[[i]]<-error_metric(actuals,predicted_values)[[2]]
  
  # Les variables les plus importantes
  interpretation[[i]]<-anova(logistic_regression_model)[,1][-length(anova(logistic_regression_model)[,1])]
  
}

AUC<-do.call(rbind,lapply(AUC,function(x) x))
Precision<-do.call(rbind,lapply(Precision,function(x) x))
Recall<-do.call(rbind,lapply(Recall,function(x) x))
interpretation<-do.call(rbind,lapply(interpretation,function(x) x))

# Resume des resultats
summary(AUC)
summary(Precision)
summary(Recall)

# Scott-knott : identifier les variables les plus importantes
sk_esd(interpretation)$groups

# Nomogram : identifier l'impacte de chaque variable
dd = datadist(data) 
options(datadist='dd')

model<-lrm(has_bug~.,data=data,x=TRUE,y=TRUE)
boot_model<-bootcov(model, B=100,pr=TRUE,maxit=1000)


eclipse_nomogram <- nomogram(boot_model, fun=function(x)1/(1+exp(-x)),  # or fun=plogis
                             fun.at=c(.001,seq(.1,.9,by=.5),.999),
                             lp = FALSE,
                             funlabel="DefectProbability",
                             abbrev = TRUE)

plot(eclipse_nomogram, xfrac=.30, cex.var = 1.0, cex.axis = 0.7,main="What makes a file defective?")

# Comparer le modele avec et sans analyse de correlation!

