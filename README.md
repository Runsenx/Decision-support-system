# Decision-support-system
This is a decision support system that evaluates the risk of Home Equity Line of Credit applications.

# 1. Introduction 

The Home Equity Line of Credit (HELOC) Risk Evaluation System is a powerful and user-friendly tool designed to
assist sales representatives in banks and credit card companies in evaluating the credit risk of applicants. 
This app interactive interface uses Python and the Streamlit library and allows users to input applicant information, receive a credit risk assessment, 
visualize the applicant's risk profile, and obtain personalized advice to help the applicant improve their credit standing.



# 2. Data Cleaning

## 2.1. Data Extraction
We first analyzed the dataset and identified three types of missing values (-7, -8, and -9). 
For the 488 rows with all values as -9, further research is needed to obtain records for these users.

## 2.2. Data Imputation
For the -7 and -8 missing values, we generated separate indicators for each feature column and imputed them with their respective column means. 
This resulted in a cleaned dataset with 69 columns, including the original 23 features and the corresponding -7 and -8 indicators.

## 2.3. Dataset Splitting
We divided the cleaned dataset into 80% training and 20% testing subsets, enabling us to evaluate our models on a representative sample not used during training.


# 3. Model Selection 

1. Train a LDA model using all features.
2. Perform individual t-tests for each feature, calculate their p-values, and remove the feature with the highest p-value (i.e., the feature with the largest p-value, least significant).
3. Retrain the LDA model using the remaining features.
4. Evaluate the new model and check if its performance has declined.
5. If the model's performance declines, repeat steps 2-4, removing the next highest p-value feature until the stopping criteria are met or the desired performance level is reached.

# 4. Cross-validation and Hyperparameter Adjustment

## 4.1 Variable Selection
When the model selection reached more than five variables, the model's accuracy increased rapidly, and the cv_score rose to over 0.73. Consequently, we chose an LDA model with eight variables, including ExternalRiskEstimate,NumSatisfactoryTrades,MaxDelq2PublicRecLast12M, MSinceMostRecentInqexcl7days,NumInqLast6M,NumInqLast6Mexcl7days,NumRevolvingTradesWBalance, and NumBank2NatlTradesWHighUtilization.

## 4.2 Model Performance
By setting the solver to 'lsqr' and shrinkage to 'auto,' we performed K-fold testing on the test set and obtained the following results:The overall accuracy is around 72.6%.


# 5. Principal Component Analysis (PCA)
Another essential reason for choosing LDA is that it can reduce data dimensions and extract features through Principal Component Analysis (PCA), providing suggestions for customers classified as 'Bad' in risk assessment. We selected four principal components.

## 5.1 Interpretation of Principal Components
Each principal component is a linear combination of the original features, with the weights (coefficients) indicating the contribution of each original feature to the new principal component. 

PCA1: Includes the number of credit inquiries (NumInqLast6M and NumInqLast6Mexcl7days) and the credit activity level (NumRevolvingTradesWBalance and NumBank2NatlTradesWHighUtilization). A higher PCA1 value may indicate more credit inquiries and a higher credit activity level.

PCA2: Includes the number of satisfactory trades (NumSatisfactoryTrades), credit inquiries (NumInqLast6M and NumInqLast6Mexcl7days), and credit utilization (NumRevolvingTradesWBalance and NumBank2NatlTradesWHighUtilization). A higher PCA2 value may indicate poorer credit behavior and higher credit utilization.

PCA3: Includes external risk estimates (ExternalRiskEstimate), the number of satisfactory trades (NumSatisfactoryTrades), and public record delinquency (MaxDelq2PublicRecLast12M). A higher PCA3 value may indicate higher credit risk and more public record delinquency.

PCA4: Mainly focuses on recent credit inquiry situations (MSinceMostRecentInqexcl7days) while also involving other variables. A higher PCA4 value indicates more recent credit inquiries.

# 6.Interactive Interface Design
The user interface is designed to be simple and intuitive, with a clear separation between the input form and he results section. Streamlit's built-in widgets, such as sliders and number inputs, are used to collect the applicant's information, making it easy for sales representatives to input data quickly and accurately. The interface also includes a welcome screen for first-time users, providing an overview of the system and its usage.

## 6.1 Visualization of Radar Chart and Bar Chart Based on PAC
To visualize the client's risk profile, we create a radar chart that plots the principal components along radial axes. We also reverse the signs of the last two principal components to ensure that higher values indicate better performance. The axes are labeled with the corresponding aspects of the risk profile, and the client's performance is represented by a polygon connecting the points on the radar chart.

To help sales representative to quickly grasp the overall credit risk, a bar chart visually representing the credit risk evaluation result (GOOD or BAD) is also applied.



## Interface prototype

![image](https://github.com/Runsenx/Decision-support-system/blob/main/bad1.png)

















