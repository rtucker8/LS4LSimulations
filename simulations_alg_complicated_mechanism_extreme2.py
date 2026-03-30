#!/usr/bin/env python
# coding: utf-8

# # LS4L simulations
# 
# **Goal:** to assess the performance of our LS4L algorithm when the relationship between the outcome, the treatment, and the context covariates is simple vs. complicated.  We consider two different settings:
# 
# * **Extreme 1:** assume that the treatment effect is constant (i.e., no interactions with context covarites)
# 
# * **Extreme 2:** assume that the treatment interacts with all context covariates
# 
# In both settings, the following characteristics will be informed by the LS4L study data: (i) enrollment of participants and scheduling of updates, (ii) distribution of context covariates, and (iii) magnitude of the treatment effect.

# ### Overview of simulation study design
# 
# Assume we have 100 participants.  These participants enrolled at an average rate of 1 person per week.  Using the R file, we generated context covariates for all participants for the entire study period.  
# 
# Now, we generate the treatment indicator and outcome variables sequentially, updating them after every 2 study-months (defined as months from study day zero). Prior to the end of an individual's first two months in the study, a constant probability of 0.8 is used to send notifications.  After study day 56, we take the current dataset as all rows (from all participants) that would have been observed at this point in time. Then, using the current dataset, we:
# 
# 1. Generate observed outcomes using the true model and true coefficients (**To do:** The current version of the code uses data generated from the simple extreme 1 model.  Need to update code to use data generated under complicated extreme 2.
# 2. Using the current dataset, generate a new mega tables for all individuals currently enrolled in the study. (**In Progress:** The current version of the code is set up to use the LS4L algorithm.  Need to update code so we also update the mega table using the simple model from extreme 1 and the complicated model from extreme 2.)
#     * As part of the mega table update, we fit a model and can get coefficient estimates.  We can compare the coefficient estimates to the true coefficient values to assess how well our model is doing at this point with the current--and limited--dataset.  If we assess the coefficients after each mega table is update, we will hopefully see decreases in bias.
# 3. Now that we have new mega tables, we can determine when treatments are sent during this next batch.  This next batch covers the period of time between the end of the second study month fourth study month. For individuals who enroll during the third and fourth months (those who don't have mega tables), we will use the constant probability of 0.8. For all other individuals, we will determine randomization probabilities based on their megatables (which were updated at the end of the second study month).
# 
# Our next update occurs at the end of the fouth study month (day 112). At this point, the current dataset now consists of all rows (from all participants) that would have been observed at this point in time.  Then we:
# 1. Generate observed outcomes for the new batch of data using the true model and true coefficients.
# 2. Using the current dataset, generate new mega tables.
#     * Assess bias in model coefficients using fitted model.
# 3. Determine when treatments are sent during the next study month (between days 112 and 168).
# 
# Repeat at 168, 224, 280, 336, 392, 448, 504, 560, 616, 672, 728, 784, and 840  days.

# ## Useful functions

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import itertools
import logging
from formulae import design_matrices
import bambi as bm
from scipy.special import expit
import pickle
from datetime import datetime
import random
import time
import arviz as az
import seaborn as sns
import multiprocessing
import sys

from IPython.display import HTML
def View(df):
    css = """<style>
    table { border-collapse: collapse; border: 3px solid #eee; }
    table tr th:first-child { background-color: #eeeeee; color: #333; font-weight: bold }
    table thead th { background-color: #eee; color: #000; }
    tr, th, td { border: 1px solid #ccc; border-width: 1px 0 0 1px; border-collapse: collapse;
    padding: 3px; font-family: monospace; font-size: 10px }</style>
    """
    s  = '<script type="text/Javascript">'
    s += 'var win = window.open("", "Title", "toolbar=no, location=no, directories=no, status=no, menubar=no, scrollbars=yes, resizable=yes, width=780, height=200, top="+(screen.height-400)+", left="+(screen.width-840));'
    s += 'win.document.body.innerHTML = \'' + (df.to_html() + css).replace("\n",'\\') + '\';'
    s += '</script>'
    return(HTML(s+css))

def generate_app_open(batch_data, model_formula, true_coefs):
    """ Generate binary outcome (app_open) using true model and true coefficients
    Parameters
        ----------
        batch_data: dataframe, default=None
            Current batch of data that needs outcomes to be generated, includes both context covariates, treatment indicator (sent), and tx prob
        model_formula: string
            RHS of true model, something like "sent_centered+time_type+notification_type+notification_type:time_type"
        true_coefs: dataframe, default=None
            Contains true coefficients from the true_betas file generated using R
    """
    print("Sampling app_open using true model probabilities")
    # First, need to calculate sent_centered variable
    batch_data['sent_centered'] = batch_data['sent'] - batch_data['prob_sent']
    
    # Then create design matrix for your model
    #First, step in creating a design matrix is to specify possible levels of all the context covariates
    lvl_time_type = ['weekday', 'weekend', 'unknown']
    lvl_time_time_simple = ['midday', 'morning', 'night', 'unknown']
    lvl_situation_major = ["travelling","unknown","working","leisure","shopping","morningrituals","workingout","housework","social"]
    lvl_weather_temperature = ['cold', 'unknown', 'freezing', 'warm', 'hot']
    lvl_notification_type = ['grocery', 'restaurant']
    
    #Next, need to figure out which levels of each covariate apear in this batch of data
    time_type_values = batch_data['time_type'].values
    time_time_values = batch_data['time_time_simple'].values
    situation_values = batch_data['situation_major'].values
    weather_values = batch_data['weather_temperature'].values
    notification_values = batch_data['notification_type'].values
    
    #Now create sublists of levels containing the values in batch_data
    lvl_ttype = [t for t in lvl_time_type if t in time_type_values]
    #for time_type in lvl_time_type:
        #if time_type in time_type_values:
            #lvl_ttype.append(time_type)
    lvl_ttime = [t for t in lvl_time_time_simple if t in time_time_values]    
    lvl_wt = [t for t in lvl_weather_temperature if t in weather_values]
    lvl_sm = [t for t in lvl_situation_major if t in situation_values]
    lvl_nt = [t for t in lvl_notification_type if t in notification_values]
            
    #Finally assign sublists as levels to categorical variables, with the first level matching reference level from R
    batch_data['time_type'] = pd.Categorical(batch_data['time_type'], categories=lvl_ttype, ordered=True)
    batch_data['time_time_simple'] = pd.Categorical(batch_data['time_time_simple'], categories=lvl_ttime, ordered=True)
    batch_data['situation_major'] = pd.Categorical(batch_data['situation_major'], categories=lvl_sm, ordered=True)
    batch_data['weather_temperature'] = pd.Categorical(batch_data['weather_temperature'], categories=lvl_wt, ordered=True)
    batch_data['notification_type'] = pd.Categorical(batch_data['notification_type'], categories=lvl_nt, ordered=True)
    batch_data['engagement'] = pd.Categorical(batch_data['engagement'])
                                                     
    #Make design matrix
    dm = design_matrices(model_formula, batch_data)
    
    #Get true coefficients to multiply with deisgn matrix
    column_headers = list(dm.common.as_dataframe().columns.values)
    select_cols = true_coefs['param'].isin(column_headers)
    true_coefs_subset = true_coefs[select_cols].copy()

    #Check to make sure we are multiplying the correct coefficients with the correct values from desing matrix
    if not list(true_coefs_subset.param) == column_headers: #Are the lists exactly equal?
        if set(list(true_coefs_subset.param)) == set(column_headers): #If the lists aren't equal but they contain the same elements, then true_coefs_subset according to column_headers.
            order_dict = {column: index for index, column in enumerate(column_headers)}
            true_coefs_subset['sort_order'] = true_coefs_subset['param'].map(order_dict)
            true_coefs_subset = true_coefs_subset.sort_values(by='sort_order').drop(columns='sort_order')
            #true_coefs_subset = true_coefs_subset.set_index('param')
            #true_coefs_subset = true_coefs_subset[column_headers].reset_index()
            print('Is order equal now?')
            print(list(true_coefs_subset.param) == column_headers)
            print("Sorted true parameter values to they are in the same order as columns of design matrix.")
        else: 
            d1 = pd.DataFrame({'cols': column_headers})
            d2 = pd.DataFrame({'params': list(true_coefs_subset.param)})
            
            print(d1)
            print(d2)
            
            raise ValueError("Columns of design matrix aren't the same as rows of true beta values. Linear predictior is wrong.")
      
    #Calculate linear predictor
    linear_predictor = np.matmul(dm.common, true_coefs_subset.beta)
    
    # Convert from logit to prob scale
    odds = np.exp(linear_predictor)
    prob = odds / (1 + odds)
    
    # Then generate the outcome (for all observations in the current batch of data) using these probabilities
    batch_data['app_open'] = np.random.binomial(n = 1, p = prob)

    return batch_data


def generate_next_batch(batch_start_day, batch_end_day, data_so_far, context_covars, tx_prob_table, model_formula, true_coefs):
    """ Based on response (app_open) to tx (sent) in the prior batch of data, and new randomization probabilities for this batch of data,
    we generate outcomes for this batch of data using the true model and true model coefficients.  Based on this new batch of data, we 
    also generate an update version of the mega table

    Note that in this simple version, we don't consider missing values or categories of context variables that we have yet to observed...
    We also only consider two covariates (time_type and notification_type)
    Parameters
        ----------
        batch_start_day: scalar, default=None
            Day 0 for current batch, same as end day for prior batch, defines what subset of data we're generating new outcomes for
        batch_end_day: scalar, default=None
            Final day of the current batch, defines what data we're generating new outcomes for
        data_so_far: dataframe, default=None
            Past observed data, with context + sent + outcomes
        context_covars: dataframe, default=None
            All data (past, current, future batches), only need to contain context covariates
        tx_prob_table: dataframe, default=None
            Version of mega table that is current at the start of this batch. This table will be used for randomization in this current batch
        model_formula: string
            RHS of true model, something like "sent_centered+time_type+notification_type+notification_type:time_type"
        true_coefs: array, default=None
            Vector of true coefficients in the same order as the columns of the data matrix
    """
    print("Determining randomization probabilities")
    # batch data = current batch of data, context covariates, treatment indicator (sent), and tx prob from all times until start of current batch
    batch_data = context_covars[(context_covars.day > batch_start_day) & (context_covars.day <= batch_end_day)]
    batch_data = pd.merge(batch_data, tx_prob_table ,how='left', on=['time_type','notification_type'])
    # generate notirications by drawing from Bernoulli distribution with p = prob_sent
    print("Sending notifications")
    batch_data['sent'] = np.random.binomial(n = 1, p = batch_data['prob_sent'])
    new_outcome = generate_app_open(batch_data = batch_data, model_formula = model_formula, true_coefs = true_coefs)
    # generate outcomes for this batch
    print("Generating outcomes using true coefficients")
    batch_data = generate_app_open(batch_data = batch_data,
                               model_formula = "sent_centered+time_type+notification_type+notification_type:time_type",
                               true_coefs = true_betas.beta)
    # combine this batch of data with the previously observed data
    data_so_far = pd.concat([data_so_far, batch_data])
    
    ## once we've generated new outcomes for time span corresponding to this batch, then we have the data that we need to update the randomization probs
    #print("Updating true mega table")
    #tx_prob_table = calc_true_tx_prob(data = data_so_far, model_formula = model_formula, true_coefs = true_coefs)
    
    #print("Returning data and mega table corresponding to end of this batch")
    #return batch_data, tx_prob_table
    
    print("Returning current data to be used in mega table update")
    return batch_data

def specify_priors():
    """ 
    Set the priors for different types of model terms. We will use the same priors
    across the 3 algorithms we are considering: LS4L, EX1, and EX2.
    
    Parameters
        ----------
        none
        
    """
    # priors for fixed effects
    prior_intercept = bm.Prior("Cauchy", alpha = 0, beta = 10) # intercept
    prior_main_effect = bm.Prior("Cauchy", alpha = 0, beta = 2.5) # main effects
    prior_interaction = bm.Prior("Cauchy", alpha = 0, beta = 0.625) # two-way interation terms
    prior_3way_interaction = bm.Prior("Cauchy", alpha = 0, beta = 0.156) # three-way interation terms
    prior_4way_interaction = bm.Prior("Cauchy", alpha = 0, beta = 0.039) # four-way interation terms

    # priors for random effects
    ss_prior_intercept = bm.Prior("Normal", mu = 0, sigma = bm.Prior("HalfNormal", sigma = 100)) # intercept
    ss_prior_main_effect = bm.Prior("Normal", mu = 0, sigma = bm.Prior("HalfNormal", sigma = 25)) # main effects
    ss_prior_interaction = bm.Prior("Normal", mu = 0, sigma = bm.Prior("HalfNormal", sigma = 6.25)) # two-way interation terms
    ss_prior_3way_interaction = bm.Prior("Normal", mu = 0, sigma = bm.Prior("HalfNormal", sigma = 1.56)) # three-way interation terms
    
    priors = {"Intercept": prior_intercept,
            "C(S1)": prior_main_effect,
            "C(S2)": prior_main_effect,

            ## main effects ##
            "sent_centered": prior_main_effect,
            "C(time_type)": prior_main_effect,
            "C(time_time_simple)": prior_main_effect,
            "C(situation_major)": prior_main_effect,
            "C(notification_type)": prior_main_effect,
            "C(weather_temperature)": prior_main_effect,
            "C(engagement)": prior_main_effect,
            "C(gender)": prior_main_effect,
            "standardized_age": prior_main_effect,

            "sent_centered:C(S1)": prior_interaction,
            "C(time_type):C(S1)": prior_interaction,
            "C(time_time_simple):C(S1)": prior_interaction,
            "C(situation_major):C(S1)": prior_interaction,
            "C(notification_type):C(S1)": prior_interaction,
            "C(weather_temperature):C(S1)": prior_interaction,
            "C(engagement):C(S1)": prior_interaction,
            "C(gender):C(S1)": prior_interaction,
            "standardized_age:C(S1)": prior_interaction,

            "sent_centered:C(S2)": prior_interaction,
            "C(time_type):C(S2)": prior_interaction,
            "C(time_time_simple):C(S2)": prior_interaction,
            "C(situation_major):C(S2)": prior_interaction,
            "C(notification_type):C(S2)": prior_interaction,
            "C(weather_temperature):C(S2)": prior_interaction,
            "C(engagement):C(S2)": prior_interaction,
            "C(gender):C(S2)": prior_interaction,
            "standardized_age:C(S2)": prior_interaction,

            ## 2 way interactions ##

            "sent_centered:C(time_type)": prior_interaction,
            "sent_centered:C(time_time_simple)": prior_interaction,
            "sent_centered:C(situation_major)": prior_interaction,
            "sent_centered:C(notification_type)": prior_interaction,
            "sent_centered:C(weather_temperature)": prior_interaction,
            "sent_centered:C(engagement)": prior_interaction,
            "sent_centered:C(gender)": prior_interaction,
            "sent_centered:standardized_age": prior_interaction,

            "sent_centered:C(time_type):C(S1)": prior_3way_interaction,
            "sent_centered:C(time_time_simple):C(S1)": prior_3way_interaction,
            "sent_centered:C(situation_major):C(S1)": prior_3way_interaction,
            "sent_centered:C(notification_type):C(S1)": prior_3way_interaction,
            "sent_centered:C(weather_temperature):C(S1)": prior_3way_interaction,
            "sent_centered:C(engagement):C(S1)": prior_3way_interaction,
            "sent_centered:C(gender):C(S1)": prior_3way_interaction,
            "sent_centered:standardized_age:C(S1)": prior_3way_interaction,

            "sent_centered:C(time_type):C(S2)": prior_3way_interaction,
            "sent_centered:C(time_time_simple):C(S2)": prior_3way_interaction,
            "sent_centered:C(situation_major):C(S2)": prior_3way_interaction,
            "sent_centered:C(notification_type):C(S2)": prior_3way_interaction,
            "sent_centered:C(weather_temperature):C(S2)": prior_3way_interaction,
            "sent_centered:C(engagement):C(S2)": prior_3way_interaction,
            "sent_centered:C(gender):C(S2)": prior_3way_interaction,
            "sent_centered:standardized_age:C(S2)": prior_3way_interaction,

            # -

            "C(time_type):C(time_time_simple)": prior_interaction,
            "C(time_type):C(situation_major)": prior_interaction,
            "C(time_type):C(notification_type)": prior_interaction,
            "C(time_type):C(weather_temperature)": prior_interaction,
            "C(time_type):C(engagement)": prior_interaction,
            "C(time_type):C(gender)": prior_interaction,
            "C(time_type):standardized_age": prior_interaction,

            "C(time_type):C(time_time_simple):C(S1)": prior_3way_interaction,
            "C(time_type):C(situation_major):C(S1)": prior_3way_interaction,
            "C(time_type):C(notification_type):C(S1)": prior_3way_interaction,
            "C(time_type):C(weather_temperature):C(S1)": prior_3way_interaction,
            "C(time_type):C(engagement):C(S1)": prior_3way_interaction,
            "C(time_type):C(gender):C(S1)": prior_3way_interaction,
            "C(time_type):standardized_age:C(S1)": prior_3way_interaction,

            "C(time_type):C(time_time_simple):C(S2)": prior_3way_interaction,
            "C(time_type):C(situation_major):C(S2)": prior_3way_interaction,
            "C(time_type):C(notification_type):C(S2)": prior_3way_interaction,
            "C(time_type):C(weather_temperature):C(S2)": prior_3way_interaction,
            "C(time_type):C(engagement):C(S2)": prior_3way_interaction,
            "C(time_type):C(gender):C(S2)": prior_3way_interaction,
            "C(time_type):standardized_age:C(S2)": prior_3way_interaction,

            # -

            "C(time_time_simple):C(situation_major)": prior_interaction,
            "C(time_time_simple):C(notification_type)": prior_interaction,
            "C(time_time_simple):C(weather_temperature)": prior_interaction,
            "C(time_time_simple):C(engagement)": prior_interaction,
            "C(time_time_simple):C(gender)": prior_interaction,
            "C(time_time_simple):standardized_age": prior_interaction,

            "C(time_time_simple):C(situation_major):C(S1)": prior_3way_interaction,
            "C(time_time_simple):C(notification_type):C(S1)": prior_3way_interaction,
            "C(time_time_simple):C(weather_temperature):C(S1)": prior_3way_interaction,
            "C(time_time_simple):C(engagement):C(S1)": prior_3way_interaction,
            "C(time_time_simple):C(gender):C(S1)": prior_3way_interaction,
            "C(time_time_simple):standardized_age:C(S1)": prior_3way_interaction,

            "C(time_time_simple):C(situation_major):C(S2)": prior_3way_interaction,
            "C(time_time_simple):C(notification_type):C(S2)": prior_3way_interaction,
            "C(time_time_simple):C(weather_temperature):C(S2)": prior_3way_interaction,
            "C(time_time_simple):C(engagement):C(S2)": prior_3way_interaction,
            "C(time_time_simple):C(gender):C(S2)": prior_3way_interaction,
            "C(time_time_simple):standardized_age:C(S2)": prior_3way_interaction,


            # -

            "C(situation_major):C(notification_type)": prior_interaction,
            "C(situation_major):C(weather_temperature)": prior_interaction,
            "C(situation_major):C(engagement)": prior_interaction,
            "C(situation_major):C(gender)": prior_interaction,
            "C(situation_major):standardized_age": prior_interaction,

            "C(situation_major):C(notification_type):C(S1)": prior_3way_interaction,
            "C(situation_major):C(weather_temperature):C(S1)": prior_3way_interaction,
            "C(situation_major):C(engagement):C(S1)": prior_3way_interaction,
            "C(situation_major):C(gender):C(S1)": prior_3way_interaction,
            "C(situation_major):standardized_age:C(S1)": prior_3way_interaction,

            "C(situation_major):C(notification_type):C(S2)": prior_3way_interaction,
            "C(situation_major):C(weather_temperature):C(S2)": prior_3way_interaction,
            "C(situation_major):C(engagement):C(S2)": prior_3way_interaction,
            "C(situation_major):C(gender):C(S2)": prior_3way_interaction,
            "C(situation_major):standardized_age:C(S2)": prior_3way_interaction,

            # -

            "C(notification_type):C(weather_temperature)": prior_interaction,
            "C(notification_type):C(engagement)": prior_interaction,
            "C(notification_type):C(gender)": prior_interaction,
            "C(notification_type):standardized_age": prior_interaction,

            "C(notification_type):C(weather_temperature):C(S1)": prior_3way_interaction,
            "C(notification_type):C(engagement):C(S1)": prior_3way_interaction,
            "C(notification_type):C(gender):C(S1)": prior_3way_interaction,
            "C(notification_type):standardized_age:C(S1)": prior_3way_interaction,

            "C(notification_type):C(weather_temperature):C(S2)": prior_3way_interaction,
            "C(notification_type):C(engagement):C(S2)": prior_3way_interaction,
            "C(notification_type):C(gender):C(S2)": prior_3way_interaction,
            "C(notification_type):standardized_age:C(S2)": prior_3way_interaction,

            # -

            "C(weather_temperature):C(engagement)": prior_interaction,
            "C(weather_temperature):C(gender)": prior_interaction,
            "C(weather_temperature):standardized_age": prior_interaction,

            "C(weather_temperature):C(engagement):C(S1)": prior_3way_interaction,
            "C(weather_temperature):C(gender):C(S1)": prior_3way_interaction,
            "C(weather_temperature):standardized_age:C(S1)": prior_3way_interaction,

            "C(weather_temperature):C(engagement):C(S2)": prior_3way_interaction,
            "C(weather_temperature):C(gender):C(S2)": prior_3way_interaction,
            "C(weather_temperature):standardized_age:C(S2)": prior_3way_interaction,

            # -

            "C(engagement):C(gender)": prior_interaction,
            "C(engagement):standardized_age": prior_interaction,

            "C(engagement):C(gender):C(S1)": prior_3way_interaction,
            "C(engagement):standardized_age:C(S1)": prior_3way_interaction,

            "C(engagement):C(gender):C(S2)": prior_3way_interaction,
            "C(engagement):standardized_age:C(S2)": prior_3way_interaction,

            # -

            "C(gender):standardized_age": prior_interaction,

            "C(gender):standardized_age:C(S1)": prior_3way_interaction,

            "C(gender):standardized_age:C(S2)": prior_3way_interaction,

            ## 3 way interactions ##

            "sent_centered:C(time_type):C(time_time_simple)": prior_3way_interaction,
            "sent_centered:C(time_type):C(situation_major)": prior_3way_interaction,
            "sent_centered:C(time_type):C(notification_type)": prior_3way_interaction,
            "sent_centered:C(time_type):C(weather_temperature)": prior_3way_interaction,
            "sent_centered:C(time_type):C(engagement)": prior_3way_interaction,
            "sent_centered:C(time_type):C(gender)": prior_3way_interaction,
            "sent_centered:C(time_type):standardized_age": prior_3way_interaction,

            "sent_centered:C(time_type):C(time_time_simple):C(S1)": prior_4way_interaction,
            "sent_centered:C(time_type):C(situation_major):C(S1)": prior_4way_interaction,
            "sent_centered:C(time_type):C(notification_type):C(S1)": prior_4way_interaction,
            "sent_centered:C(time_type):C(weather_temperature):C(S1)": prior_4way_interaction,
            "sent_centered:C(time_type):C(engagement):C(S1)": prior_4way_interaction,
            "sent_centered:C(time_type):C(gender):C(S1)": prior_4way_interaction,
            "sent_centered:C(time_type):standardized_age:C(S1)": prior_4way_interaction,

            "sent_centered:C(time_type):C(time_time_simple):C(S2)": prior_4way_interaction,
            "sent_centered:C(time_type):C(situation_major):C(S2)": prior_4way_interaction,
            "sent_centered:C(time_type):C(notification_type):C(S2)": prior_4way_interaction,
            "sent_centered:C(time_type):C(weather_temperature):C(S2)": prior_4way_interaction,
            "sent_centered:C(time_type):C(engagement):C(S2)": prior_4way_interaction,
            "sent_centered:C(time_type):C(gender):C(S2)": prior_4way_interaction,
            "sent_centered:C(time_type):standardized_age:C(S2)": prior_4way_interaction,

            # -

            "sent_centered:C(time_time_simple):C(situation_major)": prior_3way_interaction,
            "sent_centered:C(time_time_simple):C(notification_type)": prior_3way_interaction,
            "sent_centered:C(time_time_simple):C(weather_temperature)": prior_3way_interaction,
            "sent_centered:C(time_time_simple):C(engagement)": prior_3way_interaction,
            "sent_centered:C(time_time_simple):C(gender)": prior_3way_interaction,
            "sent_centered:C(time_time_simple):standardized_age": prior_3way_interaction,

            "sent_centered:C(time_time_simple):C(situation_major):C(S1)": prior_4way_interaction,
            "sent_centered:C(time_time_simple):C(notification_type):C(S1)": prior_4way_interaction,
            "sent_centered:C(time_time_simple):C(weather_temperature):C(S1)": prior_4way_interaction,
            "sent_centered:C(time_time_simple):C(engagement):C(S1)": prior_4way_interaction,
            "sent_centered:C(time_time_simple):C(gender):C(S1)": prior_4way_interaction,
            "sent_centered:C(time_time_simple):standardized_age:C(S1)": prior_4way_interaction,

            "sent_centered:C(time_time_simple):C(situation_major):C(S2)": prior_4way_interaction,
            "sent_centered:C(time_time_simple):C(notification_type):C(S2)": prior_4way_interaction,
            "sent_centered:C(time_time_simple):C(weather_temperature):C(S2)": prior_4way_interaction,
            "sent_centered:C(time_time_simple):C(engagement):C(S2)": prior_4way_interaction,
            "sent_centered:C(time_time_simple):C(gender):C(S2)": prior_4way_interaction,
            "sent_centered:C(time_time_simple):standardized_age:C(S2)": prior_4way_interaction,

            # -

            "sent_centered:C(situation_major):C(notification_type)": prior_3way_interaction,
            "sent_centered:C(situation_major):C(weather_temperature)": prior_3way_interaction,
            "sent_centered:C(situation_major):C(engagement)": prior_3way_interaction,
            "sent_centered:C(situation_major):C(gender)": prior_3way_interaction,
            "sent_centered:C(situation_major):standardized_age": prior_3way_interaction,

            "sent_centered:C(situation_major):C(notification_type):C(S1)": prior_4way_interaction,
            "sent_centered:C(situation_major):C(weather_temperature):C(S1)": prior_4way_interaction,
            "sent_centered:C(situation_major):C(engagement):C(S1)": prior_4way_interaction,
            "sent_centered:C(situation_major):C(gender):C(S1)": prior_4way_interaction,
            "sent_centered:C(situation_major):standardized_age:C(S1)": prior_4way_interaction,

            "sent_centered:C(situation_major):C(notification_type):C(S2)": prior_4way_interaction,
            "sent_centered:C(situation_major):C(weather_temperature):C(S2)": prior_4way_interaction,
            "sent_centered:C(situation_major):C(engagement):C(S2)": prior_4way_interaction,
            "sent_centered:C(situation_major):C(gender):C(S2)": prior_4way_interaction,
            "sent_centered:C(situation_major):standardized_age:C(S2)": prior_4way_interaction,

            # -

            "sent_centered:C(notification_type):C(weather_temperature)": prior_3way_interaction,
            "sent_centered:C(notification_type):C(engagement)": prior_3way_interaction,
            "sent_centered:C(notification_type):C(gender)": prior_3way_interaction,
            "sent_centered:C(notification_type):standardized_age": prior_3way_interaction,

            "sent_centered:C(notification_type):C(weather_temperature):C(S1)": prior_4way_interaction,
            "sent_centered:C(notification_type):C(engagement):C(S1)": prior_4way_interaction,
            "sent_centered:C(notification_type):C(gender):C(S1)": prior_4way_interaction,
            "sent_centered:C(notification_type):standardized_age:C(S1)": prior_4way_interaction,

            "sent_centered:C(notification_type):C(weather_temperature):C(S2)": prior_4way_interaction,
            "sent_centered:C(notification_type):C(engagement):C(S2)": prior_4way_interaction,
            "sent_centered:C(notification_type):C(gender):C(S2)": prior_4way_interaction,
            "sent_centered:C(notification_type):standardized_age:C(S2)": prior_4way_interaction,

            # -

            "sent_centered:C(weather_temperature):C(engagement)": prior_3way_interaction,
            "sent_centered:C(weather_temperature):C(gender)": prior_3way_interaction,
            "sent_centered:C(weather_temperature):standardized_age": prior_3way_interaction,

            "sent_centered:C(weather_temperature):C(engagement):C(S1)": prior_4way_interaction,
            "sent_centered:C(weather_temperature):C(gender):C(S1)": prior_4way_interaction,
            "sent_centered:C(weather_temperature):standardized_age:C(S1)": prior_4way_interaction,

            "sent_centered:C(weather_temperature):C(engagement):C(S2)": prior_4way_interaction,
            "sent_centered:C(weather_temperature):C(gender):C(S2)": prior_4way_interaction,
            "sent_centered:C(weather_temperature):standardized_age:C(S2)": prior_4way_interaction,

            # -

            "sent_centered:C(engagement):C(gender)": prior_3way_interaction,
            "sent_centered:C(engagement):standardized_age": prior_3way_interaction,

            "sent_centered:C(engagement):C(gender):C(S1)": prior_4way_interaction,
            "sent_centered:C(engagement):standardized_age:C(S1)": prior_4way_interaction,

            "sent_centered:C(engagement):C(gender):C(S2)": prior_4way_interaction,
            "sent_centered:C(engagement):standardized_age:C(S2)": prior_4way_interaction,

            # -

            "sent_centered:C(gender):standardized_age": prior_3way_interaction,

            "sent_centered:C(gender):standardized_age:C(S1)": prior_4way_interaction,

            "sent_centered:C(gender):standardized_age:C(S2)": prior_4way_interaction,

            ## RANDOM EFFECTS ##

            "1|pid": ss_prior_intercept,
            "C(S1)|pid": ss_prior_main_effect,
            "C(S2)|pid": ss_prior_main_effect,

            ## main effects ##

            "sent_centered|pid": ss_prior_main_effect,
            "C(time_type)|pid": ss_prior_main_effect,
            "C(time_time_simple)|pid": ss_prior_main_effect,
            "C(situation_major)|pid": ss_prior_main_effect,
            "C(notification_type)|pid": ss_prior_main_effect,
            "C(weather_temperature)|pid": ss_prior_main_effect,
            "C(engagement)|pid": ss_prior_main_effect,

            "sent_centered:C(S1)|pid": ss_prior_interaction,
            "C(time_type):C(S1)|pid": ss_prior_interaction,
            "C(time_time_simple):C(S1)|pid": ss_prior_interaction,
            "C(situation_major):C(S1)|pid": ss_prior_interaction,
            "C(notification_type):C(S1)|pid": ss_prior_interaction,
            "C(weather_temperature):C(S1)|pid": ss_prior_interaction,
            "C(engagement):C(S1)|pid": ss_prior_interaction,

            "sent_centered:C(S2)|pid": ss_prior_interaction,
            "C(time_type):C(S2)|pid": ss_prior_interaction,
            "C(time_time_simple):C(S2)|pid": ss_prior_interaction,
            "C(situation_major):C(S2)|pid": ss_prior_interaction,
            "C(notification_type):C(S2)|pid": ss_prior_interaction,
            "C(weather_temperature):C(S2)|pid": ss_prior_interaction,
            "C(engagement):C(S2)|pid": ss_prior_interaction,

            ## interactions ##

            "sent_centered:C(time_type)|pid": ss_prior_interaction,
            "sent_centered:C(time_time_simple)|pid": ss_prior_interaction,
            "sent_centered:C(situation_major)|pid": ss_prior_interaction,
            "sent_centered:C(notification_type)|pid": ss_prior_interaction,
            "sent_centered:C(weather_temperature)|pid": ss_prior_interaction,
            "sent_centered:C(engagement)|pid": ss_prior_interaction,

            "sent_centered:C(time_type):C(S1)|pid": ss_prior_3way_interaction,
            "sent_centered:C(time_time_simple):C(S1)|pid": ss_prior_3way_interaction,
            "sent_centered:C(situation_major):C(S1)|pid": ss_prior_3way_interaction,
            "sent_centered:C(notification_type):C(S1)|pid": ss_prior_3way_interaction,
            "sent_centered:C(weather_temperature):C(S1)|pid": ss_prior_3way_interaction,
            "sent_centered:C(engagement):C(S1)|pid": ss_prior_3way_interaction,

            "sent_centered:C(time_type):C(S2)|pid": ss_prior_3way_interaction,
            "sent_centered:C(time_time_simple):C(S2)|pid": ss_prior_3way_interaction,
            "sent_centered:C(situation_major):C(S2)|pid": ss_prior_3way_interaction,
            "sent_centered:C(notification_type):C(S2)|pid": ss_prior_3way_interaction,
            "sent_centered:C(weather_temperature):C(S2)|pid": ss_prior_3way_interaction,
            "sent_centered:C(engagement):C(S2)|pid": ss_prior_3way_interaction,

            # -

            "C(time_type):C(time_time_simple)|pid": ss_prior_interaction,
            "C(time_type):C(situation_major)|pid": ss_prior_interaction,
            "C(time_type):C(notification_type)|pid": ss_prior_interaction,
            "C(time_type):C(weather_temperature)|pid": ss_prior_interaction,
            "C(time_type):C(engagement)|pid": ss_prior_interaction,

            "C(time_type):C(time_time_simple):C(S1)|pid": ss_prior_3way_interaction,
            "C(time_type):C(situation_major):C(S1)|pid": ss_prior_3way_interaction,
            "C(time_type):C(notification_type):C(S1)|pid": ss_prior_3way_interaction,
            "C(time_type):C(weather_temperature):C(S1)|pid": ss_prior_3way_interaction,
            "C(time_type):C(engagement):C(S1)|pid": ss_prior_3way_interaction,

            "C(time_type):C(time_time_simple):C(S2)|pid": ss_prior_3way_interaction,
            "C(time_type):C(situation_major):C(S2)|pid": ss_prior_3way_interaction,
            "C(time_type):C(notification_type):C(S2)|pid": ss_prior_3way_interaction,
            "C(time_type):C(weather_temperature):C(S2)|pid": ss_prior_3way_interaction,
            "C(time_type):C(engagement):C(S2)|pid": ss_prior_3way_interaction,

            # -

            "C(time_time_simple):C(situation_major)|pid": ss_prior_interaction,
            "C(time_time_simple):C(notification_type)|pid": ss_prior_interaction,
            "C(time_time_simple):C(weather_temperature)|pid": ss_prior_interaction,
            "C(time_time_simple):C(engagement)|pid": ss_prior_interaction,

            "C(time_time_simple):C(situation_major):C(S1)|pid": ss_prior_3way_interaction,
            "C(time_time_simple):C(notification_type):C(S1)|pid": ss_prior_3way_interaction,
            "C(time_time_simple):C(weather_temperature):C(S1)|pid": ss_prior_3way_interaction,
            "C(time_time_simple):C(engagement):C(S1)|pid": ss_prior_3way_interaction,

            "C(time_time_simple):C(situation_major):C(S2)|pid": ss_prior_3way_interaction,
            "C(time_time_simple):C(notification_type):C(S2)|pid": ss_prior_3way_interaction,
            "C(time_time_simple):C(weather_temperature):C(S2)|pid": ss_prior_3way_interaction,
            "C(time_time_simple):C(engagement):C(S2)|pid": ss_prior_3way_interaction,

            # -

            "C(situation_major):C(notification_type)|pid": ss_prior_interaction,
            "C(situation_major):C(weather_temperature)|pid": ss_prior_interaction,
            "C(situation_major):C(engagement)|pid": ss_prior_interaction,

            "C(situation_major):C(notification_type):C(S1)|pid": ss_prior_3way_interaction,
            "C(situation_major):C(weather_temperature):C(S1)|pid": ss_prior_3way_interaction,
            "C(situation_major):C(engagement):C(S1)|pid": ss_prior_3way_interaction,

            "C(situation_major):C(notification_type):C(S2)|pid": ss_prior_3way_interaction,
            "C(situation_major):C(weather_temperature):C(S2)|pid": ss_prior_3way_interaction,
            "C(situation_major):C(engagement):C(S2)|pid": ss_prior_3way_interaction,

            # -

            "C(notification_type):C(weather_temperature)|pid": ss_prior_interaction,
            "C(notification_type):C(engagement)|pid": ss_prior_interaction,

            "C(notification_type):C(weather_temperature):C(S1)|pid": ss_prior_3way_interaction,
            "C(notification_type):C(engagement):C(S1)|pid": ss_prior_3way_interaction,

            "C(notification_type):C(weather_temperature):C(S2)|pid": ss_prior_3way_interaction,
            "C(notification_type):C(engagement):C(S2)|pid": ss_prior_3way_interaction,

            # -

            "C(weather_temperature):C(engagement)|pid": ss_prior_interaction,

            "C(weather_temperature):C(engagement):C(S1)|pid": ss_prior_3way_interaction,

            "C(weather_temperature):C(engagement):C(S2)|pid": ss_prior_3way_interaction,

             }
    return priors


# ## Define LS4L algorithm functions
# 
# The functions below are from the LS4L algorithm used during the real study's mega table updates.  Here, we just redefine them to make them easier to use in the simulation study.  The "run_alg" function is the main function that calls all other functions defined below; "run_alg" takes the raw data as input and then returns the megatable for the specified individual's update date.
# 
# **IMPORTANT TEMPORARY CHANGES:**
# 
# * Currently, the number of MCMC iterations used when fitting the model is set to a low value; this value should be increased in the final simulations.
# 
# * Baseline covariates are not considered in the current simulation design (they are not in the simulated data or considered when defining the model).  If you want to include baseline covariates in the model, then you will need to uncomment some of the chunks labeled "MODIFIED"

# In[3]:


####################### DEFINE MODEL #######################
def define_model_complicated(dataframe, batch_id, target_ids):

    print(f"- - - - - - - - - - - - - - DEFINE MODEL - - - - - - - - - - - - - -")

    save_messages_name = 'alg_output_' + str(alg) + '_' + str(mech) + '/' + str(sys.argv[1]) + '_model_messages_batch' + str(batch_id) + '_g' + str(g) + '.txt' # MODIFIED
    logging.basicConfig(filename=save_messages_name,level=logging.INFO,
                           format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                           datefmt='%m-%d-%y %H:%M:%S',
                           filemode='w')

    #### Specify priors ####
    pr = specify_priors()

    #### Do these data contain enough information to fit a model? ####
    # check to see if there is zero variability in the treatment variable, if no variability, then don't fit model
    no_variability_in_tx = dataframe['sent_centered'].unique().size < 2
    # check to see if there is zero variability in the outcome, if no variability, then don't fit model
    no_variability_in_outcome = dataframe['app_open'].unique().size < 2

    if no_variability_in_outcome == True:
        print(f"data do not contain enough info to fit model.")
        logging.info('App_open is constant so data do not contain enough info to fit model')
        model_formula = None
        return model_formula

    elif no_variability_in_tx == True:
        print(f"data do not contain enough info to fit model.")
        logging.info('Treatment is constant so data do not contain enough info to fit model')
        model_formula = None
        return model_formula

    else:
        # if treatment variable contains at least some variability, then attempt to fit model

        #First, step in creating a design matrix is to specify possible levels of all the context covariates
        lvl_time_type = ['weekday', 'weekend', 'holiday', 'unknown']
        lvl_time_time_simple = ['midday', 'morning', 'night', 'unknown']
        lvl_situation_major = ["travelling","unknown","working","leisure","shopping","morningrituals","workingout","housework","social",'sleeping', "other"]
        lvl_weather_temperature = ['cold', 'unknown', 'freezing', 'warm', 'hot', 'very_hot']
        lvl_notification_type = ['grocery', 'restaurant', 'unknown']
        # lvl_engagement = [0, 1 ,'unknown']

        #Next, need to figure out which levels of each covariate apear in this batch of data
        time_type_values = dataframe['time_type'].values
        time_time_values = dataframe['time_time_simple'].values
        situation_values = dataframe['situation_major'].values
        weather_values = dataframe['weather_temperature'].values
        notification_values = dataframe['notification_type'].values
        # engagement_values = dataframe['engagement'].values

        #Now create sublists of levels containing the values in batch_data
        lvl_ttype = []
        for time_type in lvl_time_type:
            if time_type in time_type_values:
                lvl_ttype.append(time_type)

        lvl_ttime = []
        for time_time_simple in lvl_time_time_simple:
            if time_time_simple in time_time_values:
                lvl_ttime.append(time_time_simple)

        lvl_wt = []
        for weather_temperature in lvl_weather_temperature:
            if weather_temperature in weather_values:
                lvl_wt.append(weather_temperature)

        lvl_sm = []
        for situation_major in lvl_situation_major:
            if situation_major in situation_values:
                lvl_sm.append(situation_major)

        lvl_nt=[]
        for notification_type in lvl_notification_type:
            if notification_type in notification_values:
                lvl_nt.append(notification_type)

        #Finally assign sublists as levels to categorical variables, with the first level matching reference level from R
        dataframe['time_type'] = pd.Categorical(dataframe['time_type'], categories=lvl_ttype, ordered=True)
        dataframe['time_time_simple'] = pd.Categorical(dataframe['time_time_simple'], categories=lvl_ttime, ordered=True)
        dataframe['situation_major'] = pd.Categorical(dataframe['situation_major'], categories=lvl_sm, ordered=True)
        dataframe['weather_temperature'] = pd.Categorical(dataframe['weather_temperature'], categories=lvl_wt, ordered=True)
        dataframe['notification_type'] = pd.Categorical(dataframe['notification_type'], categories=lvl_nt, ordered=True)
        dataframe['engagement'] = pd.Categorical(dataframe['engagement'])
        
        #View(dataframe)
        
        # NEW: Do we want to include interactions with S1 or/and S2? Split data by time period when including covariates
        dataframe_S1 = dataframe[dataframe['S1'] == 1] # all data after the 1st optimization at 2mo.
        dataframe_S2 = dataframe[dataframe['S2'] == 1] # all data after the 2nd optimization at 4mo.
        
        #build model formula- include all interactions, even those with sent_centered
        model_str = 'app_open ~ sent_centered'
        
        if dataframe_S1.shape[0] > 4:
            model_str = model_str + ' + C(S1)'
            model_str = model_str + ' + sent_centered:C(S1)'
            
        if dataframe_S2.shape[0] > 4:
            model_str = model_str + '+ C(S2)'
            model_str = model_str + '+ sent_centered:C(S2)'
        
        main_effect_terms = ['time_type', 'time_time_simple', 'situation_major', 'notification_type', 'weather_temperature', 'engagement']
        
        for var in main_effect_terms:
            model_str += ' + C(' + var+ ')'
            
        for var in main_effect_terms:
            model_str += ' + sent_centered:C(' + var+ ')'
        
        if dataframe_S1.shape[0] > 4:
            for var in main_effect_terms:
                model_str += ' + C(' + var + '):C(S1)'
            for var in main_effect_terms: 
                model_str += ' + sent_centered:C(' + var+ '):C(S1)'
        
        if dataframe_S2.shape[0] > 4:
            for var in main_effect_terms:
                model_str += ' + C(' + var + '):C(S2)'
            for var in main_effect_terms:
                model_str += ' + sent_centered:C(' + var + '):C(S2)'

        interaction_terms = []
        for a in main_effect_terms:
            for b in main_effect_terms:
                 if (main_effect_terms.index(a) < main_effect_terms.index(b)):
                    interaction_terms.append('C(' + a + '):C(' + b + ')')
        
        for var in interaction_terms:
            model_str += ' + ' + var
        
        for var in interaction_terms:
            model_str += ' + sent_centered:' + var
               
        if dataframe_S1.shape[0] > 4:       
            for var in interaction_terms:
                model_str += ' + ' + var + ':C(S1)'
            for var in interaction_terms:
                model_str += ' + sent_centered:' + var + ':C(S1)'
               
        if dataframe_S2.shape[0] > 4:       
            for var in interaction_terms:
                model_str += ' + ' + var + ':C(S2)'
            for var in interaction_terms:
                model_str += ' + sent_centered:' + var + ':C(S2)'
            
        print(model_str)
        
        # now define model in bambi function 
        # note: to consider only complete data, use , dropna=True        
        model_formula = bm.Model(model_str, dataframe, family = "bernoulli", priors = pr, dropna=True)

        logging.info('Model:')
        logging.info(model_str)

        return model_formula
###################################################################################

# fits logistic regression model for Pr(app_open = 1) using pymc
# and save is as a pickle file using the following naming covention: fitted_model_pid[target_id]_YYYY-MM-DD.pkl
def fit_model(model_formula, batch_id):

    # NEW
    if (model_formula == None):
        print(f"skipping model fitting step")
        logging.info('Model not fit')
        results = None
    else:
        print(f"preparing to fit model")
        # set random seed for reproducibility
        my_seed = int(model_seed)
        logging.info('Random seed for fitting model = ' + str(my_seed))

        # Fit model (and store error/warning messages)
        logging.info("CPU count seen by Python:", multiprocessing.cpu_count())
        results = model_formula.fit(tune = 5000, draws = 1000, chains = 4, cores = 4, random_seed=my_seed)
        logging.info('Done with fitting model')

        print(f"done with fitting model")

        # Save fitted model
        save_mod_name = 'alg_output_' + str(alg) + '_' + str(mech) + '/' + str(sys.argv[1]) + '_fitted_model_batch' + str(batch_id) + '_g' + str(g) + '.pkl' # MODIFIED
        print('saving model as:', save_mod_name)
        with open(save_mod_name, 'wb') as files:
            pickle.dump(results, files)

    return results

########################################################################################


# constructs table of probabilities using fitted model
def make_table(dataframe, model_formula, fitted_model, batch_id, target_ids, epsilon):
        # dataframe = pooled dataset used to fit model
        # fitted_model = idata
        # model_formula = model formula
        # subject_id = person-specific id number from original data
        # prob_sent = treatment probability used to center treatment indicator in data used to fit model (defaults to 0.8)
        # epsilon = small value (e.g. 0.001) that is used to truncate treatment probabilities so that they are not exact
        #           equal to 0 or 1 for causal inference later

    if (model_formula == None):
        logging.info('Updated randomization probabilities not calculated')
        open_prob_table = None
    else:
        # Fitted model should be the model that is return as the 'results' object from the function fit_model()
        print('- - - - - - - - - - - - - - SUMMARY OF FITTED MODEL - - - - - - - - - - - - - -')
        print(az.summary(fitted_model))

        # Create dataframe that contains all possible context vectors
        time_type_options = dataframe['time_type'].unique().tolist() # 0 = weekday, 1 = weekend
        time_time_options = dataframe['time_time_simple'].unique().tolist() 
        situation_major_options = dataframe['situation_major'].unique().tolist()
        notification_type_options = dataframe['notification_type'].unique().tolist()
        weather_temperature_options = dataframe['weather_temperature'].unique().tolist()
        engagement_options = dataframe['engagement'].unique().tolist()
        # result contains dictionary of all possible combinations
        combinations_list = list(itertools.product(time_type_options, time_time_options, situation_major_options,
                                                   notification_type_options, weather_temperature_options, engagement_options))

        combinations = pd.DataFrame(combinations_list, columns = ['time_type', 'time_time_simple', 'situation_major',
                                                                  'notification_type', 'weather_temperature', 'engagement'])

        # Add ID column for target person specified by target_id and their standardized age and gender
        combinations['batch'] = batch_id
        #data_i = dataframe.loc[dataframe['pid'] == target_id]
        # MODIFIED (no baseline covariates)
#             # NEW
#             # if data_i has no rows, then use target_gender, target_age (but need to standardize target_age)
#             if data_i.shape[0] == 0:
#                 #raw_dataframe = pd.read_csv(pandas_file_name)
#                 raw_dataframe = pd.read_pickle(pandas_file_name)

#                 # DEC 6: only include observations with daysInStudy <= 197
#                 raw_dataframe = raw_dataframe[(raw_dataframe.daysInStudy <= 197)]

#                 baseline_data = raw_dataframe[['pid', 'age', 'gender']]
#                 baseline_data = baseline_data.drop_duplicates()
#                 avg_age = baseline_data['age'].mean() # NEW
#                 stdev_age = baseline_data['age'].std(ddof = 0) # denominator is n
#                 combinations['standardized_age'] = (target_age - avg_age) / stdev_age
#                 combinations['gender'] = target_gender
#             else:
#                 # if data_i has >0 rows, then proceed as usual:
#                 combinations['standardized_age'] = data_i['standardized_age'].iloc[0]
#                 combinations['gender'] = data_i['gender'].iloc[0]

        # Add columns corresponding to study period: S1 and S2
        # Note that we only need to predictors for contexts with S1=0 & S2 = 0 AND S1 = 1 & S2 = 0 AND S1 = 1 & S2 = 1 (since S1 = 0 & S2 = 1 is
        #   impossible)
        #For S1 = 0 and S2 = 0, randomization probability can vary because we switched to batch style updates
        combinations_S0 = combinations.copy()
        combinations_S0['S1'] = [0 for i in range(combinations.shape[0])]
        combinations_S0['S2'] = [0 for i in range(combinations.shape[0])]
        # For S1 = 1 and S2 = 0
        combinations_S1 = combinations.copy()
        combinations_S1['S1'] = [1 for i in range(combinations.shape[0])]
        combinations_S1['S2'] = [0 for i in range(combinations.shape[0])]
        # For S1 = 1 and S2 = 1
        combinations_S2 = combinations.copy()
        combinations_S2['S1'] = [1 for i in range(combinations.shape[0])]
        combinations_S2['S2'] = [1 for i in range(combinations.shape[0])]
        # Combine dataframes back into a single one for prediction
        combinations = pd.concat([combinations_S0, combinations_S1, combinations_S2])

        # account for treatment prob centering
        # NOTE: this prob is not necessarily the actual treatment probability for this subject (or for all contexts);
        #.       using this prob will not give us the correct predicted probabilities from our model but will still allow
        #.       us to compare Pr(app_open = 1 | context, A = 1) vs. Pr(app_open = 1 | context, A = 0) to update the tx prob.
        prob_sent = 0.8 # default value, not necessarily true treatment probability

        # Create dataframes of all possible context vector for each treatment scenario (A = 0 or 1)
        combinations_sent_centered0 = combinations.copy(); combinations_sent_centered0['sent_centered'] = 0 - prob_sent
        combinations_sent_centered1 = combinations.copy(); combinations_sent_centered1['sent_centered'] = 1 - prob_sent

        # Make predictions with new data (on mean response scale, i.e., estimated bernoulli mean for each posterior sample)
        pp_sent_centered0 = model_formula.predict(fitted_model, kind='mean', data=combinations_sent_centered0, inplace=False)#sent = 0
        pp_sent_centered1 = model_formula.predict(fitted_model, kind='mean', data=combinations_sent_centered1, inplace=False)#sent = 1

        # Extract predicted probabilities (on scale of bernoulli mean)
        pred_sent_centered0 = pp_sent_centered0['posterior']['app_open_mean']
        pred_sent_centered1 = pp_sent_centered1['posterior']['app_open_mean']

        # Determine if P(open = 1 | sent = 1) or P(open = 1 | sent = 0) is larger
        compare_probs = (pred_sent_centered1 > pred_sent_centered0).values.mean(axis=0).mean(axis=0)
        # clip treatment probability to prevent values of exact 0 or 1
        compare_probs = np.maximum(epsilon, compare_probs) # impose lower bound
        compare_probs = np.minimum(1 - epsilon, compare_probs) # impose upper bound

        # Create final treatment probability table that contains context vector, subject id, and updated treatment probability
        tx_prob_table = combinations.copy()
        tx_prob_table['tx_prob'] = compare_probs
        #tx_prob_table['time_type'].replace([0,1], ['weekday', 'weekend'], inplace = True) # switch [0,1] back to [weekday, weekend]
        # remove age and gender columns from the table
        # MODIFIED
        # tx_prob_table.drop(['standardized_age', 'gender'], axis=1, inplace=True)

        # save table
        save_table_name = 'alg_output_' + str(alg) + '_' + str(mech) + '/' + str(sys.argv[1]) + '_tx_prob_table_small_batch' + str(batch_id) + '_g' + str(g) + '.csv'
        tx_prob_table.to_csv(save_table_name, index=False)

        return tx_prob_table
        
        
########################################################################################

# Calculates probabilities corresponding to the unknown level for all context variables (and all combos of context variables = unknown)
def fill_in_unknown_probs(clean_dataframe, updated_table, batch_id, target_ids): # NEW
    engagement_full = [0, 1, 'unknown']
    time_time_simple_full = ['morning', 'midday', 'night', 'unknown']
    time_type_full = ['weekend', 'weekday', 'holiday', 'unknown']
    notification_type_full = ['grocery', 'restaurant', 'unknown']
    situation_major_full = ['working', 'morningrituals', 'travelling', 'leisure', 'shopping',
                            'workingout', 'housework', 'sleeping', 'social', 'unknown']
    weather_temperature_full = ['freezing', 'cold', 'warm', 'hot', 'very_hot', 'unknown']
    # and list all the possible context variables
    context_covars = ['time_type', 'time_time_simple', 'situation_major', 'notification_type', 'weather_temperature', 'engagement']
    # Note that we assume that S1 and S2 are ALWAYS known

    #### 2. set up dataframe to store the expanded set of tx probs with updated probs for 'unknown' category of each covariate
    #      (if not already in tx table)
    tx_tab_for_unknowns = pd.DataFrame(columns = context_covars)
    tx_tab_for_unknowns['tx_prob'] = []
    tx_tab_for_unknowns.sort_index(axis = 1, inplace = True)

    # STEP 1: if variable is measured as 'unknown' but 'unknown' is NOT in the model, then take weighted average across all predicted probabilities
    # ignoring that variable, where weights depend on the frequency of each combo of context vectors in the original dataset
    max_j = len(context_covars)
    j = 0
    dataframe = clean_dataframe.copy() # create a copy of the clean data
    while j < max_j:
        Xj = context_covars[j]
        # print('----------------------------------------')
        # print('--> Current covariate: ' + Xj)
        # determine if Xj = 'unknown' is in the tx table
        cats_in_tab = updated_table[Xj].unique() # levels in tx table
        # print('Categories in table are', cats_in_tab)
        try:
            unknown_Xj_in_tab = 'unknown' in cats_in_tab
        except Exception as e:
            print(e)
            #unknown_Xj_in_tab = 'unknown' in cats_in_tab
            return None
        # print('Does the tx prob table X = unknown for ALL of', Xj, '?', unknown_Xj_in_tab)
        # if Xj = 'unknown' is not in the table, then create a predicted prob by taking wtd avg across all predicted probs ignoring Xj
        if unknown_Xj_in_tab == False:
            context_covars_not_X = list(set(context_covars)^set([Xj]))
            context_covars_not_X = context_covars_not_X + ['S1', 'S2'] # NEW APRIL 21: to prevent averaging over S1, S2
            # # Uncomment of mega table should use weights...
            # # how many observations of each category of Xj are in the clean data? (across all subjects)
            # dataframe['weight_j'] = dataframe[Xj].map(dataframe[Xj].value_counts())
            # dataframe['weight_j'] = dataframe['weight_j'] / sum(dataframe['weight_j']) * dataframe.shape[0]
            # updated_table = pd.merge(updated_table, dataframe[[Xj, 'weight_j']], on=Xj, how='left')
            # updated_table['wtd_tx_prob'] = updated_table['weight_j'] * updated_table['tx_prob']
            # dataframe.drop('weight_j', inplace=True, axis=1)
            # updated_table.drop('weight_j', inplace=True, axis=1)
            # avg_tx_probs_over_X = updated_table[context_covars + ['wtd_tx_prob']].groupby(context_covars_not_X, as_index=False).mean()
            # # ...end of weights
            avg_tx_probs_over_X = updated_table.groupby(context_covars_not_X, as_index=False).mean()
            avg_tx_probs_over_X[Xj] = 'unknown'
            # print('    Adding rows for all of X = unknown to the table')
            tx_tab_for_unknowns = pd.concat([tx_tab_for_unknowns, avg_tx_probs_over_X], ignore_index=True)
            #print(avg_tx_probs_over_X.head(4))
            # Now, check to see if both Xj = 'unknown' and Xj_prime = 'unknown' are not in the table
            j_prime = j + 1
            while j_prime < max_j:
                Xj_prime = context_covars[j_prime]
                # determine if Xj_prime = 'unknown' is in the tx table
                cats_in_tab_prime = updated_table[Xj_prime].unique() # levels in tx table
                unknown_Xjprime_in_tab = 'unknown' in cats_in_tab_prime
                # print('Does the tx prob table X = unknown for ALL of', Xj, Xj_prime, '?', unknown_Xjprime_in_tab)
                if unknown_Xjprime_in_tab == False:
                    context_covars_not_X = list(set(context_covars)^set([Xj, Xj_prime]))
                    context_covars_not_X = context_covars_not_X + ['S1', 'S2'] # NEW APRIL 21: to prevent averaging over S1, S2
                    # # Uncomment of mega table should use weights...
                    # weight_table = dataframe.groupby(by=[Xj, Xj_prime], as_index=False).size()
                    # dataframe = pd.merge(dataframe, weight_table, on=[Xj, Xj_prime], how='left')
                    # dataframe['weight_j'] = dataframe['size'] / sum(dataframe['size']) * dataframe.shape[0]
                    # updated_table = pd.merge(updated_table, dataframe[[Xj, Xj_prime, 'weight_j']], on=[Xj, Xj_prime], how='left')
                    # updated_table['wtd_tx_prob'] = updated_table['weight_j'] * updated_table['tx_prob']
                    # dataframe.drop(['size','weight_j'], inplace=True, axis=1)
                    # updated_table.drop('weight_j', inplace=True, axis=1)
                    # avg_tx_probs_over_X = updated_table[context_covars + ['wtd_tx_prob']].groupby(context_covars_not_X, as_index=False).mean()
                    # # ... end of weights
                    avg_tx_probs_over_X = updated_table.groupby(context_covars_not_X, as_index=False).mean()
                    avg_tx_probs_over_X[Xj] = 'unknown'
                    avg_tx_probs_over_X[Xj_prime] = 'unknown'
                    # print('    Adding rows for all of X = unknown to the table')
                    tx_tab_for_unknowns = pd.concat([tx_tab_for_unknowns, avg_tx_probs_over_X], ignore_index=True)
                    # Now, check to see if both Xj, Xj_prime, and Xj_prime2 = 'unknown' are not in the table
                    j_prime2 = j_prime + 1
                    while j_prime2 < max_j:
                        Xj_prime2 = context_covars[j_prime2]
                        # determine if Xj_prime = 'unknown' is in the tx table
                        cats_in_tab_prime = updated_table[Xj_prime2].unique() # levels in tx table
                        unknown_Xjprime2_in_tab = 'unknown' in cats_in_tab_prime
                        # print('Does the tx prob table X = unknown for ALL of', Xj, Xj_prime, Xj_prime2, '?', unknown_Xjprime2_in_tab)
                        if unknown_Xjprime2_in_tab == False:
                            context_covars_not_X = list(set(context_covars)^set([Xj, Xj_prime, Xj_prime2]))
                            context_covars_not_X = context_covars_not_X + ['S1', 'S2'] # NEW APRIL 21: to prevent averaging over S1, S2
                            # # Uncomment of mega table should use weights...
                            # weight_table = dataframe.groupby(by=[Xj, Xj_prime, Xj_prime2], as_index=False).size()
                            # dataframe = pd.merge(dataframe, weight_table, on=[Xj, Xj_prime, Xj_prime2], how='left')
                            # dataframe['weight_j'] = dataframe['size'] / sum(dataframe['size']) * dataframe.shape[0]
                            # updated_table = pd.merge(updated_table, dataframe[[Xj, Xj_prime, Xj_prime2, 'weight_j']], on=[Xj, Xj_prime, Xj_prime2], how='left')
                            # updated_table['wtd_tx_prob'] = updated_table['weight_j'] * updated_table['tx_prob']
                            # dataframe.drop(['size','weight_j'], inplace=True, axis=1)
                            # updated_table.drop('weight_j', inplace=True, axis=1)
                            # avg_tx_probs_over_X = updated_table[context_covars + ['wtd_tx_prob']].groupby(context_covars_not_X, as_index=False).mean()
                            # # ... end of weights
                            avg_tx_probs_over_X = updated_table.groupby(context_covars_not_X, as_index=False).mean()
                            avg_tx_probs_over_X[Xj] = 'unknown'
                            avg_tx_probs_over_X[Xj_prime] = 'unknown'
                            avg_tx_probs_over_X[Xj_prime2] = 'unknown'
                            # print('    Adding rows for all of X = unknown to the table')
                            tx_tab_for_unknowns = pd.concat([tx_tab_for_unknowns, avg_tx_probs_over_X], ignore_index=True)
                            # Now, check to see if Xj-Xj_prime3 = 'unknown' are all missing from table
                            j_prime3 = j_prime2 + 1
                            while j_prime3 < max_j:
                                Xj_prime3 = context_covars[j_prime3]
                                # determine if Xj_prime = 'unknown' is in the tx table
                                cats_in_tab_prime = updated_table[Xj_prime3].unique() # levels in tx table
                                unknown_Xjprime3_in_tab = 'unknown' in cats_in_tab_prime
                                # print('Does the tx prob table X = unknown for ALL of', Xj, Xj_prime, Xj_prime2, Xj_prime3, '?', unknown_Xjprime3_in_tab)
                                if unknown_Xjprime3_in_tab == False:
                                    context_covars_not_X = list(set(context_covars)^set([Xj, Xj_prime, Xj_prime2, Xj_prime3]))
                                    context_covars_not_X = context_covars_not_X + ['S1', 'S2'] # NEW APRIL 21: to prevent averaging over S1, S2
                                    # # Uncomment of mega table should use weights...
                                    # weight_table = dataframe.groupby(by=[Xj, Xj_prime, Xj_prime2, Xj_prime3], as_index=False).size()
                                    # dataframe = pd.merge(dataframe, weight_table, on=[Xj, Xj_prime, Xj_prime2, Xj_prime3], how='left')
                                    # dataframe['weight_j'] = dataframe['size'] / sum(dataframe['size']) * dataframe.shape[0]
                                    # updated_table = pd.merge(updated_table, dataframe[[Xj, Xj_prime, Xj_prime2, Xj_prime3, 'weight_j']],on=[Xj, Xj_prime, Xj_prime2, Xj_prime3], how='left')
                                    # updated_table['wtd_tx_prob'] = updated_table['weight_j'] * updated_table['tx_prob']
                                    # dataframe.drop(['size','weight_j'], inplace=True, axis=1)
                                    # updated_table.drop('weight_j', inplace=True, axis=1)
                                    # avg_tx_probs_over_X = updated_table[context_covars + ['wtd_tx_prob']].groupby(context_covars_not_X, as_index=False).mean()
                                    # # ... end of weights
                                    avg_tx_probs_over_X = updated_table.groupby(context_covars_not_X, as_index=False).mean()
                                    avg_tx_probs_over_X[Xj] = 'unknown'
                                    avg_tx_probs_over_X[Xj_prime] = 'unknown'
                                    avg_tx_probs_over_X[Xj_prime2] = 'unknown'
                                    avg_tx_probs_over_X[Xj_prime3] = 'unknown'
                                    # print('    Adding rows for all of X = unknown to the table')
                                    tx_tab_for_unknowns = pd.concat([tx_tab_for_unknowns, avg_tx_probs_over_X], ignore_index=True)
                                    # Now, check to see if Xj-Xj_prime3 = 'unknown' are all missing from table
                                    j_prime4 = j_prime3 + 1
                                    while j_prime4 < max_j:
                                        Xj_prime4 = context_covars[j_prime4]
                                        # determine if Xj_prime = 'unknown' is in the tx table
                                        cats_in_tab_prime = updated_table[Xj_prime4].unique() # levels in tx table
                                        unknown_Xjprime4_in_tab = 'unknown' in cats_in_tab_prime
                                        # print('Does table X = unknown for ALL of', Xj, Xj_prime, Xj_prime2, Xj_prime3, Xj_prime4, '?', unknown_Xjprime4_in_tab)
                                        if unknown_Xjprime4_in_tab == False:
                                            context_covars_not_X = list(set(context_covars)^set([Xj, Xj_prime, Xj_prime2, Xj_prime3, Xj_prime4]))
                                            context_covars_not_X = context_covars_not_X + ['S1', 'S2'] # NEW APRIL 21: to prevent averaging over S1, S2
                                            # # Uncomment of mega table should use weights...
                                            # weight_table = dataframe.groupby(by=[Xj, Xj_prime, Xj_prime2, Xj_prime3, Xj_prime4], as_index=False).size()
                                            # dataframe = pd.merge(dataframe, weight_table, on=[Xj, Xj_prime, Xj_prime2, Xj_prime3, Xj_prime4], how='left')
                                            # dataframe['weight_j'] = dataframe['size'] / sum(dataframe['size']) * dataframe.shape[0]
                                            # updated_table = pd.merge(updated_table, dataframe[[Xj, Xj_prime, Xj_prime2, Xj_prime3, Xj_prime4, 'weight_j']], on=[Xj, Xj_prime, Xj_prime2, Xj_prime3, Xj_prime4], how='left')
                                            # updated_table['wtd_tx_prob'] = updated_table['weight_j'] * updated_table['tx_prob']
                                            # dataframe.drop(['size','weight_j'], inplace=True, axis=1)
                                            # updated_table.drop('weight_j', inplace=True, axis=1)
                                            # avg_tx_probs_over_X = updated_table[context_covars + ['wtd_tx_prob']].groupby(context_covars_not_X, as_index=False).mean()
                                            # # ... end of weights
                                            avg_tx_probs_over_X = updated_table.groupby(context_covars_not_X, as_index=False).mean()
                                            avg_tx_probs_over_X[Xj] = 'unknown'
                                            avg_tx_probs_over_X[Xj_prime] = 'unknown'
                                            avg_tx_probs_over_X[Xj_prime2] = 'unknown'
                                            avg_tx_probs_over_X[Xj_prime3] = 'unknown'
                                            avg_tx_probs_over_X[Xj_prime4] = 'unknown'
                                            # print('    Adding rows for all of X = unknown to the table')
                                            tx_tab_for_unknowns = pd.concat([tx_tab_for_unknowns, avg_tx_probs_over_X], ignore_index=True)
                                        j_prime4 = j_prime4 + 1 
                                j_prime3 = j_prime3 + 1
                        j_prime2 = j_prime2 + 1
                j_prime = j_prime + 1
        j = j + 1

    updated_table = pd.concat([updated_table, tx_tab_for_unknowns])
    updated_table['batch'] = batch_id
    # updated_table['tx_prob'] = updated_table['wtd_tx_prob'] 
    # updated_table.drop('wtd_tx_prob', inplace=True, axis=1)
    # print(updated_table.head())

    # print('- - - - - - - - - - - - - - DONE WITH CALCULATING PROBS FOR ALL VARIABLES = UNKNOWN - - - - - - - - - - - - - -')

    return updated_table

########################################################################################


# Create table that serves as key to convert b/n variables in raw data and variables used to fit model
def create_category_key(raw_dataframe, updated_table, batch_id, target_ids, min_count):
    #### 1. Set up some tables ####
    # First, import the original data, which will be a smaller dataset that will define how we collapse categories
    # raw_dataframe = pd.read_pickle(pandas_file_name) # TEMP CHANGE
    #raw_dataframe = pd.read_csv(pandas_file_name)

    # DEC 6: only include observations with daysInStudy <= 197
    raw_dataframe = raw_dataframe[(raw_dataframe.daysInStudy <= 197)]

    #raw_dataframe = pd.read_csv(pandas_file_name)
    # Select the subset of columns that correspond to the context variables that we want to collapse
    orig_dataframe = raw_dataframe[['time_type', 'time_time', 'situation_major', 'notification_type','weather_temperature', 'engagement']].copy()

    # Second, create all_categories_with_key, which is a larger dataset that will contain the full and collapsed categories.
    # This dataset will be the template for creating the mega table
    # We start by creating a dataframe that contains all possible levels of the context variables, plus unknown
    all_situation_major = ['working', 'morningrituals', 'travelling', 'leisure', 'shopping',
                            'workingout', 'housework', 'sleeping', 'social', 'unknown']
    all_time_type = ['weekday', 'weekend', 'holiday', 'unknown']
    all_time_time =  ['breakfast', 'earlymorning', 'morning', 'afternoon', 'beforelunch', 'lunch', 'dinner', 'evening', 'latenight', 'night', 'unknown']
    all_weather_temperature = ['freezing', 'cold', 'warm', 'hot',  'very_hot',  'unknown']
    all_notification_type = ['restaurant', 'grocery', 'unknown']
    all_engagement = ['0', '1', 'unknown']
    # The following list will contains dictionary of all possible combinations
    list_of_all_categories = list(itertools.product(all_time_type, all_time_time, all_situation_major,
                                                    all_notification_type, all_weather_temperature, all_engagement))
    # This is the dataset w/ long/uncondensed categories that will be condensed according to this subject's data (i.e. per orig_dataframe)
    all_categories = pd.DataFrame(list_of_all_categories, columns = ['time_type', 'time_time', 'situation_major',
                                                                     'notification_type', 'weather_temperature', 'engagement'])
    # This is the dataset w/ long/uncondensed categories that will stay in the long format and form template for the mega table
    all_categories_full = pd.DataFrame(list_of_all_categories, columns = ['time_type_all', 'time_time_all', 'situation_major_all',
                                                                        'notification_type_all', 'weather_temperature_all', 'engagement_all'])

    #### 2. Condense full categories into levels that are used when fitting the model ####
    ## Select the subset of columns that we need from the original data
    new_dataframe = raw_dataframe[['time_type', 'time_time', 'situation_major', 'notification_type', 'weather_temperature', 'engagement', 'daysInStudy']].copy()

    # The following steps are the same steps taken in setup_data prior to fitting the model
    ## Condense time_time to three levels (morning, midday, night)
    all_categories['time_time_simple'] = np.select(
        [(all_categories['time_time'].isin(['breakfast', 'earlymorning', 'morning'])),
         (all_categories['time_time'].isin(['afternoon', 'beforelunch', 'lunch'])),
         (all_categories['time_time'].isin(['dinner', 'evening', 'latenight', 'night']))],
        ['morning', 'midday', 'night'], default = 'unknown'
    )

    ## Check counts in situation major categories and combine categories if needed (step 1 not needed here)
    # step 2: tally number of times each situation major was observed
    obs_counts = pd.DataFrame(new_dataframe['situation_major'].value_counts())
    obs_counts.reset_index(inplace = True)
    obs_counts.columns = ['situation_major', 'counts']

    # step 3: combine categories with fewer than 'min_count' observations with unknown
    obs_counts['too_few'] = np.where(obs_counts['counts'] < min_count, True, False)
    keep_categories = obs_counts['situation_major'][obs_counts['too_few'] == False]
    keep_categories = pd.concat([keep_categories, pd.Series(['unknown'])]) # don't replace 'unknown' with 'other'
    new_situation_major_col = all_categories['situation_major'].where(all_categories['situation_major'].isin(keep_categories), 'other') # NEW123
    all_categories['situation_major'] = new_situation_major_col
    # NEW STEP 3: combine categories (except 'unknown') with fewer than 'min_count' observations into 'other'
    #obs_counts['too_few'] = np.where(obs_counts['counts'] < min_count, True, False)
    #categories_in_other = obs_counts['situation_major'][obs_counts['too_few'] == True]
    #categories_in_other = categories_in_other.mask(lambda x: x.eq('unknown')).dropna()
    #new_situation_major_col = all_categories['situation_major'].where(~all_categories['situation_major'].isin(categories_in_other), 'other') # NEW123
    #all_categories['situation_major'] = new_situation_major_col

    ## Check counts in weather temperature and combine categories if needed
    # step 1: tally number of times each weather temperature was observed
    obs_counts = pd.DataFrame(new_dataframe['weather_temperature'].value_counts())
    obs_counts.reset_index(inplace = True)
    obs_counts.columns = ['weather_temp', 'counts']
    # step 2: determine which categories have fewer than 'min_count' observations
    obs_counts['too_few'] = np.where(obs_counts['counts'] < min_count, True, False)
    collapse_categories = obs_counts['weather_temp'][obs_counts['too_few'] == True]
    weather_temp_col = new_dataframe['weather_temperature'] # for making collapsing rules
    new_weather_temp_col = new_dataframe['weather_temperature'] # for making collapsing rules
    weather_temp_col2 = all_categories['weather_temperature'] # the data that we actually want to colapse
    new_weather_temp_col2 = all_categories['weather_temperature'] # the data that we actually want to collapse

    if len(collapse_categories.index)>0:
        # at least one category does not have enough observations so need to condense
        # step 3: if cold or freezing are observed fewer than 'min_count' times, combine into a single category called cold
        if 'freezing' or 'cold' in collapse_categories.values:
            new_weather_temp_col = new_weather_temp_col.where(~weather_temp_col.isin(['cold', 'freezing']), 'cold')
            new_weather_temp_col2 = new_weather_temp_col2.where(~weather_temp_col2.isin(['cold', 'freezing']), 'cold')

        # step 4: if hot or very hot are observed fewer than 'min_count' times, combine into a single category called hot
        if 'very_hot' or 'hot' in collapse_categories.values:
            new_weather_temp_col = new_weather_temp_col.where(~weather_temp_col.isin(['hot', 'very_hot']), 'hot')
            new_weather_temp_col2 = new_weather_temp_col2.where(~weather_temp_col2.isin(['hot', 'very_hot']), 'hot')

        # step 5: now check that hot and cold have enough observations; if not, combine with warm
        obs_counts = pd.DataFrame(new_weather_temp_col.value_counts())
        obs_counts.reset_index(inplace = True)
        obs_counts.columns = ['weather_temp', 'counts']
        obs_counts['too_few'] = np.where(obs_counts['counts'] < min_count, True, False)
        collapse_categories = obs_counts['weather_temp'][obs_counts['too_few'] == True]

        if 'cold' in collapse_categories.values:
            new_weather_temp_col = new_weather_temp_col.where(~new_weather_temp_col.isin(['cold']), 'warm')
            new_weather_temp_col2 = new_weather_temp_col2.where(~new_weather_temp_col2.isin(['cold']), 'warm')
        if 'hot' in collapse_categories.values:
            new_weather_temp_col = new_weather_temp_col.where(~new_weather_temp_col.isin(['hot']), 'warm')
            new_weather_temp_col2 = new_weather_temp_col2.where(~new_weather_temp_col2.isin(['hot']), 'warm')

        # step 6: now check that warm has enough observations; if not, combine with hot or cold (whichever has more)
        if 'warm' in collapse_categories.values:
            if obs_counts['counts'][obs_counts['weather_temp'] == 'hot'].values.size == 0:
                count_hot = 0
            else:
                count_hot = obs_counts['counts'][obs_counts['weather_temp'] == 'hot'].values[0]

            if obs_counts['counts'][obs_counts['weather_temp'] == 'cold'].values.size == 0:
                count_cold = 0
            else:
                count_cold = obs_counts['counts'][obs_counts['weather_temp'] == 'cold'].values[0]

            if count_hot >= count_cold:
                move_warm_here = 'hot'
            else:
                move_warm_here = 'cold'
            new_weather_temp_col = new_weather_temp_col.where(~new_weather_temp_col.isin(['warm']), move_warm_here)
            new_weather_temp_col2 = new_weather_temp_col2.where(~new_weather_temp_col2.isin(['warm']), move_warm_here)

        all_categories['weather_temperature'] = new_weather_temp_col2

    # select just the columns we want
    all_categories = all_categories[['time_type', 'time_time_simple', 'situation_major',
                                      'notification_type', 'weather_temperature', 'engagement']]
    # combine condensed and full tables
    category_key = pd.concat([all_categories_full, all_categories], axis=1)
    category_key = category_key.reindex(sorted(category_key.columns), axis=1)
    category_key['engagement_all'] = category_key.engagement_all.astype(str)
    category_key['engagement'] = category_key.engagement.astype(str)

    # add in S1 and S2
    # For S1 = 0 and S2 = 0
    category_key_S0 = category_key.copy()
    category_key_S0['S1'] = [0 for i in range(category_key.shape[0])]
    category_key_S0['S2'] = [0 for i in range(category_key.shape[0])]    
    # For S1 = 1 and S2 = 0
    category_key_S1 = category_key.copy()
    category_key_S1['S1'] = [1 for i in range(category_key.shape[0])]
    category_key_S1['S2'] = [0 for i in range(category_key.shape[0])]
    # For S1 = 1 and S2 = 1
    category_key_S2 = category_key.copy()
    category_key_S2['S1'] = [1 for i in range(category_key.shape[0])]
    category_key_S2['S2'] = [1 for i in range(category_key.shape[0])]
    # Combine back together
    category_key = pd.concat([category_key_S0, category_key_S1, category_key_S2])

    # save this in case it's needed for debugging later
    save_cat_key = 'alg_output_' + str(alg) + '_' + str(mech) + '/' + str(sys.argv[1]) + '_category_key_batch' + str(batch_id) + '_g' + str(g) + '.csv'
    category_key.to_csv(save_cat_key, index=False)

    return category_key

    
########################################################################################

def read_dataframe(pandas_file_name):
    # MODIFIED
    # output = pd.read_pickle(pandas_file_name) # TEMP CHANGE
    output = pd.read_csv(pandas_file_name) # TEMP CHANGE

    # DEC 6: only include observations with daysInStudy <= 197
    output = output[(output.daysInStudy <= 197)]

    output
    #output = pd.read_csv(pandas_file_name)
    #output = output.sample(n = 200) # TEMP: use for testing with smaller amount of data (so code runs faster)

    return output


###########################################################################################

# sets up data and condenses categories with few (< min_count) observations ###
def setup_data(dataframe, batch_id, target_ids, min_count):

    # dataframe = current data
    # target id = id for subject-of-interest
    # min count = min # of observations needed in each category of a categorical variable for category to be included

    # MODIFIED (don't save)
    # # save raw data in case results need to be checked later
    # save_raw_data = 'data_raw_pid' + str(target_id) + '_g' + str(g) + '.csv'
    # dataframe.to_csv(save_raw_data)

    ## Create a new binary indicator that describes the period (0-2mo, 2-4mo, 4+mo) in study for each obs
    dataframe['S1'] =  (dataframe['daysInStudy'] > 56) * 1  # takes value of 1 if daysInStudy is after 2mo. optimization point
    dataframe['S2'] =  (dataframe['daysInStudy'] > 112) * 1  # takes value of 1 if daysInStudy is after 4mo. mark

    ## Create new variable that is treatment indicator centered at treatment probability
    dataframe['sent_centered'] = dataframe['sent'] - dataframe['prob_sent'] 

    ## Select the subset of columns that we need
    # MODIFIED (no baseline covariates)
    #new_dataframe = dataframe[['time_type', 'time_time', 'situation_major', 'notification_type', 'weather_temperature', 'engagement',
    #                    'sent_centered','app_open', 'gender', 'S1', 'S2', 'daysInStudy']].copy()
    new_dataframe = dataframe[['time_type', 'time_time', 'situation_major', 'notification_type', 'weather_temperature', 'engagement',
                        'sent_centered','app_open', 'S1', 'S2', 'daysInStudy']].copy()

    ## For time_type: replace weekday/weekend with 0,1
    #new_dataframe.replace(['weekday', 'weekend'], [0,1], inplace = True)
    #new_dataframe.replace(['Weekday', 'Weekend'], [0,1], inplace = True)

    ## Condense time_time to three levels (morning, midday, night)
    new_dataframe['time_time_simple'] = np.select(
        [(new_dataframe['time_time'].isin(['breakfast', 'earlymorning', 'morning'])),
         (new_dataframe['time_time'].isin(['afternoon', 'beforelunch', 'lunch'])),
         (new_dataframe['time_time'].isin(['dinner', 'evening', 'latenight', 'night']))],
        ['morning', 'midday', 'night'], default = 'unknown'
    )

    ## Convert app open to integer
    new_dataframe['app_open'] = pd.to_numeric(new_dataframe['app_open'])

    ## Check counts in situation major categories and combine categories if needed
    # step 1: combine observations in which situation major is missing with unknown
    new_dataframe['situation_major'].replace(['-'], ['unknown'], inplace = True)

    # step 2: tally number of times each situation major was observed
    obs_counts = pd.DataFrame(new_dataframe['situation_major'].value_counts())
    obs_counts.reset_index(inplace = True)
    obs_counts.columns = ['situation_major', 'counts']

    # step 3: combine categories with fewer than 'min_count' observations with unknown
    #obs_counts['too_few'] = np.where(obs_counts['counts'] < min_count, True, False)
    #keep_categories = obs_counts['situation_major'][obs_counts['too_few'] == False]
    #new_situation_major_col = new_dataframe['situation_major'].where(new_dataframe['situation_major'].isin(keep_categories), 'other') # NEW123
    #new_dataframe['situation_major'] = new_situation_major_col
    # NEW STEP 3: combine categories (except 'unknown') with fewer than 'min_count' observations into 'other'
    obs_counts['too_few'] = np.where(obs_counts['counts'] < min_count, True, False)
    categories_in_other = obs_counts['situation_major'][obs_counts['too_few'] == True]
    categories_in_other = categories_in_other.mask(lambda x: x.eq('unknown')).dropna()
    new_situation_major_col = new_dataframe['situation_major'].where(~new_dataframe['situation_major'].isin(categories_in_other), 'other') # NEW123
    new_dataframe['situation_major'] = new_situation_major_col

    ## Check counts in weather temperature and combine categories if needed
    # step 1: tally number of times each weather temperature was observed
    obs_counts = pd.DataFrame(new_dataframe['weather_temperature'].value_counts())
    obs_counts.reset_index(inplace = True)
    obs_counts.columns = ['weather_temp', 'counts']

    # step 2: determine which categories have fewer than 'min_count' observations
    obs_counts['too_few'] = np.where(obs_counts['counts'] < min_count, True, False)
    collapse_categories = obs_counts['weather_temp'][obs_counts['too_few'] == True]
    weather_temp_col = new_dataframe['weather_temperature']
    new_weather_temp_col = new_dataframe['weather_temperature']

    if len(collapse_categories.index)>0:
        # at least one category does not have enough observations so need to condense
        # step 3: if cold or freezing are observed fewer than 'min_count' times, combine into a single category called cold
        if 'freezing' or 'cold' in collapse_categories.values:
            new_weather_temp_col = new_weather_temp_col.where(~weather_temp_col.isin(['cold', 'freezing']), 'cold')

        # step 4: if hot or very hot are observed fewer than 'min_count' times, combine into a single category called hot
        if 'very_hot' or 'hot' in collapse_categories.values:
            new_weather_temp_col = new_weather_temp_col.where(~weather_temp_col.isin(['hot', 'very_hot']), 'hot')

        # step 5: now check that hot and cold have enough observations; if not, combine with warm
        obs_counts = pd.DataFrame(new_weather_temp_col.value_counts())
        obs_counts.reset_index(inplace = True)
        obs_counts.columns = ['weather_temp', 'counts']
        obs_counts['too_few'] = np.where(obs_counts['counts'] < min_count, True, False)
        collapse_categories = obs_counts['weather_temp'][obs_counts['too_few'] == True]

        if 'cold' in collapse_categories.values:
            new_weather_temp_col = new_weather_temp_col.where(~new_weather_temp_col.isin(['cold']), 'warm')
        if 'hot' in collapse_categories.values:
            new_weather_temp_col = new_weather_temp_col.where(~new_weather_temp_col.isin(['hot']), 'warm')

        # step 6: now check that warm has enough observations; if not, combine with hot or cold (whichever has more)
        if 'warm' in collapse_categories.values:
            if obs_counts['counts'][obs_counts['weather_temp'] == 'hot'].values.size == 0:
                count_hot = 0
            else:
                count_hot = obs_counts['counts'][obs_counts['weather_temp'] == 'hot'].values[0]

            if obs_counts['counts'][obs_counts['weather_temp'] == 'cold'].values.size == 0:
                count_cold = 0
            else:
                count_cold = obs_counts['counts'][obs_counts['weather_temp'] == 'cold'].values[0]

            if count_hot >= count_cold:
                move_warm_here = 'hot'
            else:
                move_warm_here = 'cold'
            new_weather_temp_col = new_weather_temp_col.where(~new_weather_temp_col.isin(['warm']), move_warm_here)

        new_dataframe['weather_temperature'] = new_weather_temp_col

    # add user ids back into dataframe
    new_dataframe['pid'] = dataframe['pid']
    # also keep current tx probability in the dataframe
    new_dataframe['prob_sent'] = dataframe['prob_sent']

    # MODIFIED (no baseline covariates)
    # # NEW: standardize age: (age - avg_age)/stdev(age)
    # baseline_data = dataframe[['pid', 'age', 'gender']]
    # baseline_data = baseline_data.drop_duplicates()
    # avg_age = baseline_data['age'].mean() # NEW
    # centered_age = baseline_data['age'] - avg_age # NEW
    # stdev_age = baseline_data['age'].std(ddof = 0) # denominator is n
    # standardized_age = centered_age / stdev_age
    # baseline_data['standardized_age'] = standardized_age
    # # join standardized age back with full data
    # new_dataframe = new_dataframe.merge(baseline_data[['pid', 'standardized_age']], on='pid', how='left')

     # NEW TESTING: if fewer than 5 observations of 'unknonw', then drop 'unknown from model        
    if (new_dataframe['time_type'] == 'unknown').sum() < 3:
        new_dataframe = new_dataframe[new_dataframe['time_type'] != 'unknown']
        # print('dropping unknown from time_type')
    if (new_dataframe['time_time_simple'] == 'unknown').sum() < 3:
        new_dataframe = new_dataframe[new_dataframe['time_time_simple'] != 'unknown']
        # print('dropping unknown from time_time_simple')
    if (new_dataframe['situation_major'] == 'unknown').sum() < 3:
        new_dataframe = new_dataframe[new_dataframe['situation_major'] != 'unknown']
        # print('dropping unknown from situation_major')
    if (new_dataframe['notification_type'] == 'unknown').sum() < 3:
        new_dataframe = new_dataframe[new_dataframe['notification_type'] != 'unknown']
        # print('dropping unknown from notification_type')
    if (new_dataframe['weather_temperature'] == 'unknown').sum() < 3:
        new_dataframe = new_dataframe[new_dataframe['weather_temperature'] != 'unknown']
        # print('dropping unknown from weather_temperature')
    if (new_dataframe['engagement'] == 'unknown').sum() < 3:
        new_dataframe = new_dataframe[new_dataframe['engagement'] != 'unknown']
        # print('dropping unknown from engagement')
    # if (new_dataframe['gender'] == 'unknown').sum() < 3:
    #     new_dataframe = new_dataframe[new_dataframe['gender'] != 'unknown']
    #     print('dropping unknown from gender')

    # save clean data in case results need to be checked later
    save_clean_data = 'alg_output_' + str(alg) + '_' + str(mech) + '/' + str(sys.argv[1]) + '_data_clean_batch' + str(batch_id) + '_g' + str(g) + '.csv'
    new_dataframe.to_csv(save_clean_data)

    return new_dataframe

########################################################################################

def run_alg_complicated(raw_data_filename, batch_id, target_ids):
    alg='complicated'
    mech='extreme2'
    raw_dataframe = read_dataframe(raw_data_filename)
    dataframe = setup_data(raw_dataframe, batch_id, target_ids, min_count = 5) # NEW, min_count = 25?

    result = None
    
    try:
        model_formula = define_model_complicated(dataframe, batch_id, target_ids)
    except Exception as e:
        print(e)
        print("Error in definition of model formula")
    else:
        try:
            #model fitting may raise an error if the model is too complex for the data we observed
            model_fit = fit_model(model_formula, batch_id)
        except Exception as e:
            #code to execute if the model cannot be fit
            print(e)
            print("Complicated model could not be fit to the observed data from this batch. Megatables were not updated, will continue to use last avilable megatable.")
        else: 
            #code to execute if the model CAN be fit
            updated_table = make_table(dataframe, model_formula, model_fit, batch_id, target_ids, epsilon = 0.05)
            #updated_table_longer = self.fill_in_unknown_probs(updated_table, target_id)
            #category_key = create_category_key(raw_dataframe, updated_table, batch_id, target_ids, min_count = 5)
            #mega_table = make_mega_table(dataframe, updated_table, category_key, batch_id, target_ids)            
            result = updated_table
    finally:
        #do this irregardless of the model being fit
        return result # result is a table of probabilities (if there's enough data to fit the model)



# # Simulation

# ## Load context covariates and true coefficients

# In[4]:

# context covariates for all participants over all days
mech = 'extreme2'

#random seed for data
data_seed_table = pd.read_csv('data/data_generation_seeds.csv')
data_seed_table = data_seed_table.loc[data_seed_table.mechanism == mech ]
if int(sys.argv[1])%50 == 0:
    g = int(data_seed_table.loc[data_seed_table.ids == 50, 'data_seed'])
else:
    g = int(data_seed_table.loc[data_seed_table.ids == int(sys.argv[1])%50, 'data_seed'])

#random seed for pymc3 models
model_fitting_seed_table = pd.read_csv('data/pymc3_seeds.csv')
model_seed = int(model_fitting_seed_table.loc[model_fitting_seed_table.array_index == int(sys.argv[1]), 'seed'])

# load the raw data for this setting (either extreme 1 or 2)
sim_dat_all = pd.read_csv('data/sim_dat_' + str(mech) + '_g' + str(g) + '.csv')
        
# true coefficients for model of app_open ~ sent + context + ...
true_betas = pd.read_csv('data/true_beta_' + str(mech) + '_g336.csv')

## Determine update days

# Updates should occur every 56 days until the end of the study. 'day' tells us on which study day the observatins were recorded.
update_dates = pd.DataFrame({'update_id' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                             'update' : [ 56, 112, 168, 224, 280, 336, 392, 448, 504, 560, 616, 672, 728, 784, 840]})


# ## Simulation FOR loop
# 
# Convert the above simulation outline into a for loop structure. There are a total of 869 study days. If we update the megatables every 56 days (approximately two months), then this corresponds to 14 iterations of the loop. 

# In[ ]:

def main(): 
    mech = 'extreme2'

    for batch in range(update_dates.shape[0]): #index from 0 to 14 corresponds to rows in update_dates
        # set target batch ID for this update
        update_id = update_dates['update_id'].iloc[batch]
        
        if batch!= 0:
            prev_update_id = update_dates['update_id'].iloc[batch - 1]

        # set final day of data to be used in mega table update
        update_day = update_dates['update'].iloc[batch]
        
        if batch!= 0:
            prev_update_day = update_dates['update'].iloc[batch - 1]
        if batch== 0:
            prev_update_day = -1
            
        print('------------------------------------------------------------------------------')    
        print('Updating mega tables for batch ' + str(update_id) + ' on day ' + str(update_day))

        # Step 1: select the current set of partcipants for this batch
        batch_data = sim_dat_all[(sim_dat_all.day <= update_day) & (sim_dat_all.day > prev_update_day)].copy()
    # batch_data['engagement']=batch_data.engagement.astype('str')

        # set target participant IDs for this update
        target_ids = batch_data['pid'].unique().tolist()

        # Step 2: sample treatment indicator A
        if batch == 0 :
            batch_data['prob_sent'] = 0.8
        else:
            # If a megatable already exists for a participant, then use the tx prob contained in this table
            batch_data = batch_data.reset_index(drop=True)
            batch_data = pd.merge(batch_data, mega_tabs, how = 'left', on = ['pid', 'time_type', 'notification_type', 'time_time_simple',
                                                                    'situation_major', 'weather_temperature', 'engagement',
                                                                    'S1', 'S2'])
            batch_data['engagement']=batch_data.engagement.astype('category')

            # If a participant does not yet have a megatab or a new context is encountered, then fill in the missing tx probs with 0.8
            batch_data['tx_prob'] = batch_data.tx_prob.fillna(0.8)

            # Rename tx_prob as prob_sent
            batch_data['prob_sent'] = batch_data['tx_prob']
            batch_data.drop(['tx_prob'], axis = 'columns', inplace = True)
            
        # Generate notifications by drawing from Bernoulli distribution with p = prob_sent
        batch_data['sent'] = np.random.binomial(n = 1, p = batch_data['prob_sent'])

        # Step 3: Generate outcome (app open or not)
        batch_data = generate_app_open(batch_data = batch_data,
                                    model_formula = "sent_centered+S1+S2+sent_centered:S1+sent_centered:S2+time_type+time_time_simple+situation_major+notification_type+weather_temperature+engagement+sent_centered:time_type+sent_centered:time_time_simple+sent_centered:situation_major+sent_centered:notification_type+sent_centered:weather_temperature+sent_centered:engagement+S1:time_type+S1:time_time_simple+S1:situation_major+S1:notification_type+S1:weather_temperature+S1:engagement+S2:time_type+S2:time_time_simple+S2:situation_major+S2:notification_type+S2:weather_temperature+S2:engagement+sent_centered:S1:time_type+sent_centered:S1:time_time_simple+sent_centered:S1:situation_major+sent_centered:S1:notification_type+sent_centered:S1:weather_temperature+sent_centered:S1:engagement+sent_centered:S2:time_type+sent_centered:S2:time_time_simple+sent_centered:S2:situation_major+sent_centered:S2:notification_type+sent_centered:S2:weather_temperature+sent_centered:S2:engagement+time_type:time_time_simple+time_type:situation_major+time_type:notification_type+time_type:weather_temperature+time_type:engagement+time_time_simple:situation_major+time_time_simple:notification_type+time_time_simple:weather_temperature+time_time_simple:engagement+situation_major:notification_type+situation_major:weather_temperature+situation_major:engagement+notification_type:weather_temperature+notification_type:engagement+weather_temperature:engagement+sent_centered:time_type:time_time_simple+sent_centered:time_type:situation_major+sent_centered:time_type:notification_type+sent_centered:time_type:weather_temperature+sent_centered:time_type:engagement+sent_centered:time_time_simple:situation_major+sent_centered:time_time_simple:notification_type+sent_centered:time_time_simple:weather_temperature+sent_centered:time_time_simple:engagement+sent_centered:situation_major:notification_type+sent_centered:situation_major:weather_temperature+sent_centered:situation_major:engagement+sent_centered:notification_type:weather_temperature+sent_centered:notification_type:engagement+sent_centered:weather_temperature:engagement+S1:time_type:time_time_simple+S1:time_type:situation_major+S1:time_type:notification_type+S1:time_type:weather_temperature+S1:time_type:engagement+S1:time_time_simple:situation_major+S1:time_time_simple:notification_type+S1:time_time_simple:weather_temperature+S1:time_time_simple:engagement+S1:situation_major:notification_type+S1:situation_major:weather_temperature+S1:situation_major:engagement+S1:notification_type:weather_temperature+S1:notification_type:engagement+S1:weather_temperature:engagement+S2:time_type:time_time_simple+S2:time_type:situation_major+S2:time_type:notification_type+S2:time_type:weather_temperature+S2:time_type:engagement+S2:time_time_simple:situation_major+S2:time_time_simple:notification_type+S2:time_time_simple:weather_temperature+S2:time_time_simple:engagement+S2:situation_major:notification_type+S2:situation_major:weather_temperature+S2:situation_major:engagement+S2:notification_type:weather_temperature+S2:notification_type:engagement+S2:weather_temperature:engagement+sent_centered:S1:time_type:time_time_simple+sent_centered:S1:time_type:situation_major+sent_centered:S1:time_type:notification_type+sent_centered:S1:time_type:weather_temperature+sent_centered:S1:time_type:engagement+sent_centered:S1:time_time_simple:situation_major+sent_centered:S1:time_time_simple:notification_type+sent_centered:S1:time_time_simple:weather_temperature+sent_centered:S1:time_time_simple:engagement+sent_centered:S1:situation_major:notification_type+sent_centered:S1:situation_major:weather_temperature+sent_centered:S1:situation_major:engagement+sent_centered:S1:notification_type:weather_temperature+sent_centered:S1:notification_type:engagement+sent_centered:S1:weather_temperature:engagement+sent_centered:S2:time_type:time_time_simple+sent_centered:S2:time_type:situation_major+sent_centered:S2:time_type:notification_type+sent_centered:S2:time_type:weather_temperature+sent_centered:S2:time_type:engagement+sent_centered:S2:time_time_simple:situation_major+sent_centered:S2:time_time_simple:notification_type+sent_centered:S2:time_time_simple:weather_temperature+sent_centered:S2:time_time_simple:engagement+sent_centered:S2:situation_major:notification_type+sent_centered:S2:situation_major:weather_temperature+sent_centered:S2:situation_major:engagement+sent_centered:S2:notification_type:weather_temperature+sent_centered:S2:notification_type:engagement+sent_centered:S2:weather_temperature:engagement",
                                    true_coefs = true_betas) # TRUE COEFFICIENTS
        
        #Model Formula for Extreme1: "sent_centered+S1+S2+time_type+time_time_simple+situation_major+notification_type+weather_temperature+engagement+S1:time_type+S1:time_time_simple+S1:situation_major+S1:notification_type+S1:weather_temperature+S1:engagement+S2:time_type+S2:time_time_simple+S2:situation_major+S2:notification_type+S2:weather_temperature+S2:engagement+time_type:time_time_simple+time_type:situation_major+time_type:notification_type+time_type:weather_temperature+time_type:engagement+time_time_simple:situation_major+time_time_simple:notification_type+time_time_simple:weather_temperature+time_time_simple:engagement+situation_major:notification_type+situation_major:weather_temperature+situation_major:engagement+notification_type:weather_temperature+notification_type:engagement+weather_temperature:engagement+S1:time_type:time_time_simple+S1:time_type:situation_major+S1:time_type:notification_type+S1:time_type:weather_temperature+S1:time_type:engagement+S1:time_time_simple:situation_major+S1:time_time_simple:notification_type+S1:time_time_simple:weather_temperature+S1:time_time_simple:engagement+S1:situation_major:notification_type+S1:situation_major:weather_temperature+S1:situation_major:engagement+S1:notification_type:weather_temperature+S1:notification_type:engagement+S1:weather_temperature:engagement+S2:time_type:time_time_simple+S2:time_type:situation_major+S2:time_type:notification_type+S2:time_type:weather_temperature+S2:time_type:engagement+S2:time_time_simple:situation_major+S2:time_time_simple:notification_type+S2:time_time_simple:weather_temperature+S2:time_time_simple:engagement+S2:situation_major:notification_type+S2:situation_major:weather_temperature+S2:situation_major:engagement+S2:notification_type:weather_temperature+S2:notification_type:engagement+S2:weather_temperature:engagement",
        #Model Formula for Extreme2: "sent_centered+S1+S2+sent_centered:S1+sent_centered:S2+time_type+time_time_simple+situation_major+notification_type+weather_temperature+engagement+sent_centered:time_type+sent_centered:time_time_simple+sent_centered:situation_major+sent_centered:notification_type+sent_centered:weather_temperature+sent_centered:engagement+S1:time_type+S1:time_time_simple+S1:situation_major+S1:notification_type+S1:weather_temperature+S1:engagement+S2:time_type+S2:time_time_simple+S2:situation_major+S2:notification_type+S2:weather_temperature+S2:engagement+sent_centered:S1:time_type+sent_centered:S1:time_time_simple+sent_centered:S1:situation_major+sent_centered:S1:notification_type+sent_centered:S1:weather_temperature+sent_centered:S1:engagement+sent_centered:S2:time_type+sent_centered:S2:time_time_simple+sent_centered:S2:situation_major+sent_centered:S2:notification_type+sent_centered:S2:weather_temperature+sent_centered:S2:engagement+time_type:time_time_simple+time_type:situation_major+time_type:notification_type+time_type:weather_temperature+time_type:engagement+time_time_simple:situation_major+time_time_simple:notification_type+time_time_simple:weather_temperature+time_time_simple:engagement+situation_major:notification_type+situation_major:weather_temperature+situation_major:engagement+notification_type:weather_temperature+notification_type:engagement+weather_temperature:engagement+sent_centered:time_type:time_time_simple+sent_centered:time_type:situation_major+sent_centered:time_type:notification_type+sent_centered:time_type:weather_temperature+sent_centered:time_type:engagement+sent_centered:time_time_simple:situation_major+sent_centered:time_time_simple:notification_type+sent_centered:time_time_simple:weather_temperature+sent_centered:time_time_simple:engagement+sent_centered:situation_major:notification_type+sent_centered:situation_major:weather_temperature+sent_centered:situation_major:engagement+sent_centered:notification_type:weather_temperature+sent_centered:notification_type:engagement+sent_centered:weather_temperature:engagement+S1:time_type:time_time_simple+S1:time_type:situation_major+S1:time_type:notification_type+S1:time_type:weather_temperature+S1:time_type:engagement+S1:time_time_simple:situation_major+S1:time_time_simple:notification_type+S1:time_time_simple:weather_temperature+S1:time_time_simple:engagement+S1:situation_major:notification_type+S1:situation_major:weather_temperature+S1:situation_major:engagement+S1:notification_type:weather_temperature+S1:notification_type:engagement+S1:weather_temperature:engagement+S2:time_type:time_time_simple+S2:time_type:situation_major+S2:time_type:notification_type+S2:time_type:weather_temperature+S2:time_type:engagement+S2:time_time_simple:situation_major+S2:time_time_simple:notification_type+S2:time_time_simple:weather_temperature+S2:time_time_simple:engagement+S2:situation_major:notification_type+S2:situation_major:weather_temperature+S2:situation_major:engagement+S2:notification_type:weather_temperature+S2:notification_type:engagement+S2:weather_temperature:engagement+sent_centered:S1:time_type:time_time_simple+sent_centered:S1:time_type:situation_major+sent_centered:S1:time_type:notification_type+sent_centered:S1:time_type:weather_temperature+sent_centered:S1:time_type:engagement+sent_centered:S1:time_time_simple:situation_major+sent_centered:S1:time_time_simple:notification_type+sent_centered:S1:time_time_simple:weather_temperature+sent_centered:S1:time_time_simple:engagement+sent_centered:S1:situation_major:notification_type+sent_centered:S1:situation_major:weather_temperature+sent_centered:S1:situation_major:engagement+sent_centered:S1:notification_type:weather_temperature+sent_centered:S1:notification_type:engagement+sent_centered:S1:weather_temperature:engagement+sent_centered:S2:time_type:time_time_simple+sent_centered:S2:time_type:situation_major+sent_centered:S2:time_type:notification_type+sent_centered:S2:time_type:weather_temperature+sent_centered:S2:time_type:engagement+sent_centered:S2:time_time_simple:situation_major+sent_centered:S2:time_time_simple:notification_type+sent_centered:S2:time_time_simple:weather_temperature+sent_centered:S2:time_time_simple:engagement+sent_centered:S2:situation_major:notification_type+sent_centered:S2:situation_major:weather_temperature+sent_centered:S2:situation_major:engagement+sent_centered:S2:notification_type:weather_temperature+sent_centered:S2:notification_type:engagement+sent_centered:S2:weather_temperature:engagement",
                
        if batch == 0:
            data_so_far = batch_data.copy()
        if batch != 0:
            # read in raw data from previous update
            prev_raw_data = pd.read_csv('alg_output_complicated_extreme2/' + str(sys.argv[1]) + '_data_raw_batch' + str(prev_update_id) + '_g' + str(g) + '.csv')
            # combine previous raw data and raw data for this update to get the current dataset used for this mega tab update
            data_so_far = pd.concat([prev_raw_data, batch_data])

        # save data_so_far as raw data so that it can be called when running the LS4L algorithm
        if batch != 14:
            save_raw_data = 'alg_output_complicated_extreme2/' + str(sys.argv[1]) + '_data_raw_batch' + str(update_id) + '_g' + str(g) + '.csv'
            data_so_far.to_csv(save_raw_data, index=False)
        elif batch == 14:
            save_raw_data = 'alg_output_complicated_extreme2/results_complicated_extreme2/' + str(sys.argv[1]) + '_data_raw_batch' + str(update_id) + '_g' + str(g) + '.csv'
            data_so_far.to_csv(save_raw_data, index=False)
            break

        # Step 4: Run LS4L algorithm (process data + fit model + generate megatable)
        alg='complicated'
        result=run_alg_complicated(save_raw_data, update_id, target_ids)
        
        if not (result is None):
            # read in current mega table
            mega_tabs = pd.read_csv('alg_output_' + str(alg) + '_' + str(mech) + '/' + str(sys.argv[1]) + '_tx_prob_table_small_batch' + str(update_id) + '_g' + str(g) + '.csv')
            mega_tabs = mega_tabs.reset_index(drop=True)

            #since mega-table isn't subject specific, copy it for each individual in target_ids
            all_mega_tabs = []

            for i in target_ids:
                # Create a copy of mega_tabs
                copy_df = mega_tabs.copy()
                copy_df = copy_df.drop(columns=['batch'], errors='ignore')
                # Add the person_id column
                copy_df['pid'] = i
                # Append the modified copy to the list
                all_mega_tabs.append(copy_df)

            # Concatenate all the DataFrames in the list into a single DataFrame
            mega_tabs = pd.concat(all_mega_tabs, ignore_index=True)
            
        
        #save this round of target_ids for the next iteration
        prev_target_ids = target_ids
            
        
if __name__ == '__main__':
    from multiprocessing import freeze_support
    
    freeze_support()  # Only needed on Windows if the program is frozen into an executable
    alg='complicated'
    main()  # Ensure multiprocessing is called within the main guard


