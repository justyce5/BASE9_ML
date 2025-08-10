import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import ascii
import os
import arviz as az
from diptest import diptest
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pickle
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

##sklearn is data analysis library

##train test split - splits arrays or matrices into random train and test subsets
##random forest classifier -  meta estimator that fits a number of decision tree classifiers on various
                            #sub-samples of the dataset and uses averaging to improve the predictive 
                            #accuracy and control over-fitting 

##accuracy score - computes subset accuracy
##select kbest - select features according to the highest scores
##f_classif - computes ANOVA (statistical test) F-value for sample
##standard scaler- standardize features by removing the mean and scaling to unit variance

def create_stats (directory, column = 0, files = 1e5): #function that will calculate all the features
    # Function to simulate MCMC samples around each mean age
    def simulate_samples(mean_age,std_dev, num_samples=10000):
        # Simulate normal distribution around the mean age
        return np.random.normal(loc=mean_age, scale=std_dev, size=num_samples)  # Adjust scale as needed

    # function to calcualate effective sample size for each star's mean age
    def calculate_ess_for_mean_ages(mean_age,std_dev_age):
        ess_values = []
        for m, s in zip(mean_age, std_dev_age):
            # Simulate MCMC samples for each mean age
            samples = simulate_samples(m, s)
            samples_reshaped = samples[np.newaxis, :]  # Reshape for ArviZ
            ess = az.ess(samples_reshaped)  # Calculate ESS
            ess_values.append(ess)
        return ess_values
    
    Median = []
    Mean = []
    Percent16 = []
    Percent84 = []
    Width=[]
    Source_id=[]
    SnR = []
    Stdev = []
    dip_val=[]
    dip_p=[]
    ks_val = []
    ks_p = []
    upper = []
    lower = []
    #creating empty arrays for each feature

    file_count = 0

    for filename in os.listdir(directory):
        if file_count >= files: 
            break 
        
        if filename.endswith(".res"):
            file_path = os.path.join(directory, filename)

            try:

                data = np.genfromtxt(file_path, skip_header=1, usecols= column)
        
        
                Age_16th = np.percentile(data, 16)
                Age_84th = np.percentile(data, 84)
                Age_med = np.median(data)
                Age_mean = np.mean(data)
                Age_wid = Age_84th-Age_16th
                Age_std = np.std(data)
                Age_snr = Age_mean / Age_std if Age_std != 0 else np.nan #inputting nan wherever dividing by zero
                Upper_bound = Age_84th - Age_med
                Lower_bound = Age_med - Age_16th

                #calculating all the features 
                

                Percent16.append(Age_16th)
                Percent84.append(Age_84th)
                Median.append(Age_med)
                Width.append(Age_wid)
                Stdev.append(Age_std)
                Mean.append(Age_mean)
                SnR.append(Age_snr)
                upper.append(Upper_bound)
                lower.append(Lower_bound)
                #appending empty arrays with calculated values
        
                source_id = filename.replace('.res','').replace('NGC_2682_','')
                Source_id.append(source_id)
                #appending the source id array with sourceid value from file name

                dip_value, p_value = diptest(data) #calculating dip value and p value
                dip_val.append(dip_value)
                dip_p.append(p_value)
                #diptest tests for unimodality .. whether a data set has a single peak or multiple
                # p value indicates whether the data is likely from a unimodial distribution (~ >.05 likely )

                normal_dist = np.random.normal(Age_mean, Age_std, len(data)) #creating an example data set with the same mean age and stdev as dataset
                ks_statistic, ks_p_value = stats.kstest(data, normal_dist)
                ks_val.append(ks_statistic)
                ks_p.append(ks_p_value)
                #ks test is comparing our dataset to the example of what a normal distribution would look like with our dataset's mean age and stdev
                #p value indicates the probablity of observing the measured difference between two distributions 

                file_count += 1

            except Exception as e:
                print(f"Error processing file {filename}: {e}")



    ESS = calculate_ess_for_mean_ages(Mean, Stdev) #adding column in for calculated ESS

    statistic = pd.DataFrame({ 
        #creating the datatable using all the calculated features
        'source_id' : Source_id,
        'Width': Width,
        'Upper_bound': Upper_bound,
        'Lower_bound': Lower_bound,
        'Stdev': Stdev,
        'SnR': SnR,
        'Dip_p': dip_p,
        'Dip_value': dip_val,
        'KS_value': ks_val,
        'KS_p': ks_p,
        'ESS': ESS



    })
    
    statistic = statistic.dropna(subset=['SnR']) #removing rows with NaN in snr column

    return statistic #output of the function will be the datatable


def create_model(model_cluster_statistic, sampling_df):
  
    #making sure source_id in both dataframes are the same datatype
    model_cluster_statistic['source_id'] = model_cluster_statistic['source_id'].astype(str)
    sampling_df['source_id'] = sampling_df['source_id'].astype(str)

    #merging dataframes based on source ids to match ids to  known sampling labels
    merged_df = model_cluster_statistic.merge(
        sampling_df[['source_id', 'Single Sampling']],
        on='source_id',
        how='left'
    )
    
    merged_df = merged_df.dropna(subset=['Single Sampling']) #dropping any rows with NaN values

    X = merged_df.drop(columns=['source_id', 'Single Sampling']) #dropping sourceid and sampling columns
    y = merged_df['Single Sampling'] #target variable .. we want the model to predict these labels


    # Splitting the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    ##random state - sets a seed in random number generator, ensures the splits generated are reproducible
    ##test size - proportion of dataset to include in test split ~ 30% of data is in test split
    ##allows to train a model on training set and test its accuracy on testing set

    # Standardize the features
    scaler = StandardScaler()
    ##remove mean and scale to unit variance
    ##performs best if all features are on the same scale, (0-1, or whatever scale)


    X_train = scaler.fit_transform(X_train)

    X_test_save = X_test
    X_test = scaler.transform(X_test)
    ##fit - computes mean and stdevs to be used for scaling
    ##transform - perform scaling using mean and stdev from .fit

    #Can use selector if wanting to run with different k 
    # # Feature selection to identify the most relevant features
    # selector = SelectKBest(f_classif, k='all')  # Adjust 'k' to select fewer features if needed
    # X_train_selected = selector.fit_transform(X_train, y_train)
    # X_test_selected = selector.transform(X_test)
    # ##select best features according to k highest scores ?
    # ##k is the amount of features



    # Train a Random Forest Classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    ## training a model by feeding it a dataset and rfc learns patterns and relationships within data to make predictions on new unseen data

    return X, y, clf, X_test, X_train, y_test

def make_preds(X, clf, y_test = None, X_columns = ['Width', 'Upper_bound', 'Lower_bound', 'Stdev', 'SnR', 'Dip_p',
       'Dip_value', 'KS_value', 'KS_p', 'ESS']):
    # Make predictions
    y_pred = clf.predict(X)
    ##making predictions from rfc 
    #predicting what label would be


    #Print evaluation metrics
    #Giving option to print important features
    if y_test is not None:
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
    ##computing subset accuracy, label predictions in a sample must exactly corresponding labels in y
    ##classification report - build a text report showing main classification metrics

        # Feature importance
        feature_importances = clf.feature_importances_
        important_features = pd.Series(feature_importances, index=X_columns).sort_values(ascending=False)
        ##measuring how much a feature contributes to a model's prediction 



        print("Feature Importance Ranking:")
        print(important_features)

    return y_pred
    

    

def save_model(ngc2682_model, filename = 'my_model.pkl'):
    pickle.dump(ngc2682_model, open(filename, 'wb'))

#creating a function that will save the model to a pickle file to be used later
    
def prepare_df_formodel(df, columns = ['Width', 'Upper_bound', 'Lower_bound', 'Stdev', 'SnR', 'Dip_p',
       'Dip_value', 'KS_value', 'KS_p', 'ESS']):
    df_clean = df[columns]
    df_array = df_clean.to_numpy()

    return df_clean, df_array