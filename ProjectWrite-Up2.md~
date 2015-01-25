#Weight Lifting Exercise Experiment
#Analysis for Practical Machine Learning Project

##Data Preprocessing

The analysis starts with a preprocessing step which included a summary of each of the 160 variables.  From this the researcher found that only 60 variables contained substantial number of values (other than blanks and NAs).  Of these the first six elements were discarded as idiosyncratic to the sample or invariant.  This left 54 variables, 53 potential predictors and one dependent or classification variable “classe”.  Examination of  a set of “pairs” diagrams for selected variables along with a correlation matrix suggested two observations 1) that there were some highly correlated variables  (leaving out  classe, there were 37 variable pairs with abs(r) >0.71 => 50% linear variation in common), and 2) that the successful classe separating function was unlikely to be from a statistical learning perspective to be inflexible model. The variables removed from the training set were also removed from the test data set.

Because of the substantial correlation present and the large number of potential predictors, a dimensionality reduction step for the predictors using Principal Components Analysis (PCA) was undertaken.  The cumulative variation explained for the first 20 components was as follows:

Cumulative Proportion of Variation Explained
        PC_1     PC_2      PC_3       PC_4        PC_5 
      0.1577   0.3116    0.3999     0.4778      0.5467 
        PC_6     PC_7      PC_8       PC_9       PC_10 
      0.6040   0.6464    0.6856     0.7181      0.7466
       PC_11    PC_12     PC_13      PC_14       PC_15 
      0.7730   0.7947    0.8144     0.8322      0.8489 
       PC_16    PC_17     PC_18      PC_19       PC_20 
      0.8641   0.8779    0.8906     0.9020      0.9120 

Examination of a scree plot of the components along with the somewhat arbitrary decision to limit variation explained to 80% yielded 13 as the number of components to be used.  The preprocessing  function was applied to the training set of 53 predictors yielding a matrix decomposition.  A processing step to create the PC values was performed.  The identical matrices  created from the  decomposition of the training data and training PC score generation were applied to test data to create a set of PC values for the test data.

As is often the case with the use of PCs, interpretability suffers.  To aid this factor analytic loadings were computed for both unrotated and varimax rotated axes. I am sure someone can make-up a story from these, but that facility is beyond my domain knowledge.  The loadings with an absolute value >= 0.75 for unrotated axes are given in the next table. Please fill-in the interpretation which most suits your mental map.  The rotated loadings are available upon request but are not provided here for reasons of time and space.


           UnRot Factor 1
     roll_belt         -0.886
     total_accel_belt  -0.877
     accel_belt_y	   -0.914
     accel_belt_z	    0.9129
     accel_arm_y	    0.7747


          UnRot Factor 2
     pitch_belt	        0.8271
     accel_belt_x	   -0.835
     magnet_belt_x	   -0.821
     yaw_dumbbell	    0.7534


         UnRot Factor 3
     magnet_arm_y	    0.7989


         UnRot Factor 4
     gyros_dumbbell_x  -0.915
     gyros_dumbbell_z	0.9185
     gyros_forearm_y	0.831
     gyros_forearm_z	0.9206


        UnRot Factor 7
     gyros_arm_x	    0.767


##Model Selection 

A series of pairs of scatter-grams focused on the relationship between classe and the training components were examined.  As observed earlier no simple relationship is immediately apparent. So a flexible model is suggested.  Given time constrains an examination over a variety of flexible models was 

not performed and the author made a somewhat arbitrary decision and went directly to a Random Forest (RF) model.  This selection was not without some merit as RF is a flexible statistical learning model and  the instructors recommend it as a model which is commonly used in classification modeling.

##Model Fitting and Cross Validation

The RF models and cross-validation were computed using the following set of R functions:

Random Forest fitting:

    modFit500<-randomForest(x=trainNoClassPC, y=as.factor(classe),  xtest=testNoClassPC, ytest=NULL, ntree=500)
    modFit1000<-randomForest(x=trainNoClassPC, y=as.factor(classe),  xtest=testNoClassPC, ytest=NULL, ntree=1000)
    modFit1500<-randomForest(x=trainNoClassPC, y=as.factor(classe),  xtest=testNoClassPC, ytest=NULL, ntree=1500)


Cross-Validation:

    resultCV5<- rfcv(trainNoClassPC, as.factor(classe), cv.fold=5, scale="log", step=0.5,
                mtry=function(p) max(1, floor(sqrt(p))), recursive=FALSE)
    resultCV10<-rfcv(trainNoClassPC, as.factor(classe), cv.fold=10, scale="log", step=0.5,
                 mtry=function(p) max(1, floor(sqrt(p))), recursive=FALSE)

In these commands: 

trainNoClassPC is a 19,622 X 13 data matrix of training samples (rows) by the 13 principal components values (columns) computed as previously described, with the classification value classe removed.

classe is a length 19,622  factor array of actual classification values from the experiment; it takes on one of the five values in the set {A, B, C, D, E} and is part of the training set.

testNoClassPC is a 20 X 13 data matrix of test samples (rows) by the 13 principal components values (columns) computed as previously described with the identify problem_id removed.

Three runs of the randomForest function were performed with differing numbers of trees to examine the stability of the result. The confusion matrix for the 1500 tree model is 


              A    B    C    D    E class.error       
         A 5512   19   30   16    3  0.0121
         B   51 3687   51    3    5  0.0290
         C    9   39 3339   30    5  0.0242
         D   12   12  109 3081    2  0.0420
         E    1   15   10   13 3568  0.0108

This yields for the training set an accuracy of 0.978 and an overall error rate of 0.0222.  The differences from the runs using smaller number of trees was not greater, although the number of correctly classified classes (diagonally elements) increase slightly and monotonically from 500 to 1000 to 1500 trees. 

The classification of the test samples was in all three runs the same, namely:

          problem_id: 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
                      B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 

All these values were identified as correct when submitted as part of the exercise.

For the cross validation analysis as stated above the rfcv function was run using all default parameters with the exception of the cvfold value set at 5 and 10 alternatively.

 

                                  Number of Predictors Use
             cvfold          13          6          3          1 
                  5    0.02996636 0.09514830 0.39909285 0.73957802 
                 10    0.02594027 0.08928753 0.39761492 0.73672409 

The results are consist with theory in at least two ways.  Firstly there is the expected increase in bias as we increase the number of folds, although this increase is not large at about 0.4%. Secondly, the cross-validation analysis of the sort implemented would tend to have an error rate which greater than that from the training set with essentially a number of folds = 1, which we previously estimates as 0.0222.  Lastly the number of components used seems reasonable  as there is a significant increase in the error rate as we decrease the number predictors.









