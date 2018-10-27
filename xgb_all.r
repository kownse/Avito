library(tidyverse)
library(lubridate)
library(magrittr)
library(text2vec)
library(tokenizers)
library(stopwords)
library(xgboost)
library(Matrix)
library(stringr)
library(stringi)
library(forcats)
library(dplyr)
set.seed(0)

#---------------------------


cat("Loading data...\n")
num = Inf

tr <- read_csv("../input/att_train.csv") 
tri <- 1:nrow(tr)
rm(tr);gc()

tr_te <- read_csv("/media/extend/cache/total_0_1503424.csv") %>%
    select(-X1, -deal_probability)
y <- read_csv("/media/extend/cache/total_y_0_1503424.csv")$deal_probability
cat("End Loading data...\n")

       
catcols = c('user_id', 'city', 'region', 
            'parent_category_name', 'category_name',
            'param_1','param_2','param_3',
            'user_type', 'Weekday','img_label', 'reg_Time_zone')
for (col in catcols){
    tr_te[[col]] <- ifelse(is.na(tr_te[[col]]), 'none', tr_te[[col]])
}

tr_te <- tr_te %>% 
  mutate(no_img = is.na(image) %>% as.integer(),
         no_dsc = is.na(description) %>% as.integer(),
         no_p1 = is.na(param_1) %>% as.integer(), 
         no_p2 = is.na(param_2) %>% as.integer(), 
         no_p3 = is.na(param_3) %>% as.integer(),
         titl_capE = str_count(title, "[A-Z]"),
         titl_capR = str_count(title, "[А-Я]"),
         desc_capE = str_count(description, "[A-Z]"),
         desc_capR = str_count(description, "[А-Я]"),
         user_type = factor(user_type),
         category_name = factor(category_name),
         parent_category_name = factor(parent_category_name), 
         region = factor(region),
         img_label = factor(img_label),
         param_1 = factor(param_1),
         param_2 = factor(param_2),
         param_3 = factor(param_3) %>% fct_lump(prop = 0.00005),
         city =  factor(city) %>% fct_lump(prop = 0.0003),
         user_id = factor(user_id) %>% fct_lump(prop = 0.000025),
         Weekday = factor(Weekday),
         reg_Time_zone = factor(reg_Time_zone),
         n_user_items = log1p(n_user_items),
         avg_days_up_user = log1p(avg_days_up_user),
         avg_times_up_user = log1p(avg_times_up_user),
         txt = paste(title, description, sep = " "))%>% 
  select(-item_id, -title, -description) %>% 
  replace_na(list(image_top_1 = -1, price = -1, 
                  param_1   = -1, param_2 = -1, param_3 = -1, 
                  titl_len  = 0,  desc_len  = 0,
                  titl_capE = 0,  titl_capR = 0, 
                  titl_lowE = 0,  titl_lowR = 0, 
                  desc_cap  = 0,
                  titl_pun  = 0,  desc_pun = 0,
                  titl_dig  = 0,  desc_dig = 0,
                  desc_capE = 0,
                  desc_capR = 0)) %T>% 
  glimpse()

for (col in colnames(tr_te)){
    cnt_zero = sum(is.na(tr_te[col]))
    if(cnt_zero > 0){
        #cat(col, cnt_zero, "\n")
        tr_te[[col]] <- ifelse(is.na(tr_te[[col]]), 0, tr_te[[col]])
    }  
}


#---------------------------
cat("Parsing text...\n")
it <- tr_te %$%
  str_to_lower(txt) %>%
  str_replace_all("[^[:alpha:]]", " ") %>%
  str_replace_all("\\s+", " ") %>%
  tokenize_word_stems(language = "russian") %>% 
  itoken()

vect <- create_vocabulary(it, ngram = c(1, 1), stopwords = stopwords("ru")) %>%
  prune_vocabulary(term_count_min = 3, doc_proportion_max = 0.4, vocab_term_max = 12500) %>% 
  vocab_vectorizer()

m_tfidf <- TfIdf$new(norm = "l2", sublinear_tf = T)
tfidf <-  create_dtm(it, vect) %>% 
  fit_transform(m_tfidf)

rm(it, vect, m_tfidf); gc()

#---------------------------
cat("Preparing data...\n")

colnames(tr_te)[colSums(is.na(tr_te)) > 0]

if(T){
    
    catcols <- c(catcols)
    n <- nrow(tr_te)
    nlevels <- sapply(tr_te[catcols], nlevels)
    i <- rep(seq_len(n), ncol(tr_te[catcols]))
    j <- unlist(lapply(tr_te[catcols], as.integer)) +
         rep(cumsum(c(0, head(nlevels, -1))), each = n)
    x <- 1
    sparse_cats <- sparseMatrix(i = i, j = j, x = x)
    
    X <- tr_te %>% 
      select(-txt, -user_id, -city, 
             -region, -parent_category_name, 
             -category_name, 
             -param_1, -param_2, -param_3, 
             -user_type, 
             -Weekday,
             -reg_Time_zone,
             -img_label
            ) %>% 
      sparse.model.matrix(~ . - 1, .) %>%
      cbind(sparse_cats) %>%
      cbind(tfidf)


    rm(tr_te, sparse_cats, tfidf); gc()
}

if(F){    
    X <- tr_te %>% 
      select(-txt) %>% 
      sparse.model.matrix(~ . - 1, .) %>% 
      cbind(tfidf)

    rm(tr_te, tfidf); gc()
}

dtest <- xgb.DMatrix(data = X[-tri, ])
X <- X[tri, ]; gc()

cat(dim(X))
cat("\n")

p <- list(objective = "reg:logistic",
          booster = "gbtree",
          eval_metric = "rmse",
          nthread = 11,
          eta = 0.05,
          max_depth = 18,
          #min_child_weight = 11,
          gamma = 0,
          subsample = 0.80,
          colsample_bytree = 0.7,
          alpha = 2.5,
          lambda = 0.5,
          nrounds = 28000)
cols <- colnames(X)

# X<-X[sample(nrow(X)),]
idx = seq(1, length(y))
idx = sample(idx)
fold = 5
folds <- cut(idx,breaks=fold,labels=FALSE)
for(i in 1:(fold+1)){
  #Segement your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  
  testX <- X[testIndexes, ]
  testY <- y[testIndexes]
  
  trainX <- X[-testIndexes, ]
  trainY <- y[-testIndexes]
  #Use the test and train data partitions however you desire...
  
  dtrain <- xgb.DMatrix(data = trainX, label = trainY)
  dval <- xgb.DMatrix(data = testX, label = testY)
  m_xgb <- xgb.train(p, dtrain, p$nrounds, 
                     list(train = dtrain, val = dval), 
                     print_every_n = 10, 
                     early_stopping_rounds = 50)
  
  cat("Creating submission file...\n")
  read_csv("../input/sample_submission.csv")  %>%
    mutate(deal_probability = predict(m_xgb, dtest)) %>%
    write_csv(paste0("../result/xgb_tfidf_", i, ".csv"))
}