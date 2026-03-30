
# Header ------------------------------------------------------------------
#Author: Rachel Gonzalez
#Purpose: Calculate average regret under each scenario for LS4L simulations
#Date: 2024-10-27

# Preliminaries -----------------------------------------------------------
getwd() # Get current working directory
setwd("/Volumes/sph-wdem/LS4L") # Set working directory (turbo storage volume on Great Lakes)
library(tidyverse)
library(colorspace)
library(patchwork)

for (mech in c("extreme1", "extreme2")) {
  for (alg in c("simple", "complicated")) {
    
# Read in Data ------------------------------------------------------------

#formula string based on data generating mechanism
if (mech == 'extreme2') {
  formula_str = "app_open~sent+S1+S2+sent:S1+sent:S2+time_type+time_time_simple+situation_major+notification_type+weather_temperature+engagement+time_type:sent+time_time_simple:sent+situation_major:sent+notification_type:sent+weather_temperature:sent+engagement:sent+time_type:S1+time_time_simple:S1+situation_major:S1+notification_type:S1+weather_temperature:S1+engagement:S1+time_type:S2+time_time_simple:S2+situation_major:S2+notification_type:S2+weather_temperature:S2+engagement:S2+time_type:sent:S1+time_time_simple:sent:S1+situation_major:sent:S1+notification_type:sent:S1+weather_temperature:sent:S1+engagement:sent:S1+time_type:sent:S2+time_time_simple:sent:S2+situation_major:sent:S2+notification_type:sent:S2+weather_temperature:sent:S2+engagement:sent:S2+time_time_simple:time_type+situation_major:time_type+notification_type:time_type+weather_temperature:time_type+engagement:time_type+situation_major:time_time_simple+notification_type:time_time_simple+weather_temperature:time_time_simple+engagement:time_time_simple+notification_type:situation_major+weather_temperature:situation_major+engagement:situation_major+weather_temperature:notification_type+engagement:notification_type+engagement:weather_temperature+time_time_simple:time_type:sent+situation_major:time_type:sent+notification_type:time_type:sent+weather_temperature:time_type:sent+engagement:time_type:sent+situation_major:time_time_simple:sent+notification_type:time_time_simple:sent+weather_temperature:time_time_simple:sent+engagement:time_time_simple:sent+notification_type:situation_major:sent+weather_temperature:situation_major:sent+engagement:situation_major:sent+weather_temperature:notification_type:sent+engagement:notification_type:sent+engagement:weather_temperature:sent+time_time_simple:time_type:S1+situation_major:time_type:S1+notification_type:time_type:S1+weather_temperature:time_type:S1+engagement:time_type:S1+situation_major:time_time_simple:S1+notification_type:time_time_simple:S1+weather_temperature:time_time_simple:S1+engagement:time_time_simple:S1+notification_type:situation_major:S1+weather_temperature:situation_major:S1+engagement:situation_major:S1+weather_temperature:notification_type:S1+engagement:notification_type:S1+engagement:weather_temperature:S1+time_time_simple:time_type:S2+situation_major:time_type:S2+notification_type:time_type:S2+weather_temperature:time_type:S2+engagement:time_type:S2+situation_major:time_time_simple:S2+notification_type:time_time_simple:S2+weather_temperature:time_time_simple:S2+engagement:time_time_simple:S2+notification_type:situation_major:S2+weather_temperature:situation_major:S2+engagement:situation_major:S2+weather_temperature:notification_type:S2+engagement:notification_type:S2+engagement:weather_temperature:S2+time_time_simple:time_type:S1:sent+situation_major:time_type:S1:sent+notification_type:time_type:S1:sent+weather_temperature:time_type:S1:sent+engagement:time_type:S1:sent+situation_major:time_time_simple:S1:sent+notification_type:time_time_simple:S1:sent+weather_temperature:time_time_simple:S1:sent+engagement:time_time_simple:S1:sent+notification_type:situation_major:S1:sent+weather_temperature:situation_major:S1:sent+engagement:situation_major:S1:sent+weather_temperature:notification_type:S1:sent+engagement:notification_type:S1:sent+engagement:weather_temperature:S1:sent+time_time_simple:time_type:S2:sent+situation_major:time_type:S2:sent+notification_type:time_type:S2:sent+weather_temperature:time_type:S2:sent+engagement:time_type:S2:sent+situation_major:time_time_simple:S2:sent+notification_type:time_time_simple:S2:sent+weather_temperature:time_time_simple:S2:sent+engagement:time_time_simple:S2:sent+notification_type:situation_major:S2:sent+weather_temperature:situation_major:S2:sent+engagement:situation_major:S2:sent+weather_temperature:notification_type:S2:sent+engagement:notification_type:S2:sent+engagement:weather_temperature:S2:sent"
} else if (mech == 'extreme1'){
  formula_str = "app_open~sent+S1+S2+time_type+time_time_simple+situation_major+notification_type+weather_temperature+engagement+time_type:S1+time_time_simple:S1+situation_major:S1+notification_type:S1+weather_temperature:S1+engagement:S1+time_type:S2+time_time_simple:S2+situation_major:S2+notification_type:S2+weather_temperature:S2+engagement:S2+time_time_simple:time_type+situation_major:time_type+notification_type:time_type+weather_temperature:time_type+engagement:time_type+situation_major:time_time_simple+notification_type:time_time_simple+weather_temperature:time_time_simple+engagement:time_time_simple+notification_type:situation_major+weather_temperature:situation_major+engagement:situation_major+weather_temperature:notification_type+engagement:notification_type+engagement:weather_temperature+time_time_simple:time_type:S1+situation_major:time_type:S1+notification_type:time_type:S1+weather_temperature:time_type:S1+engagement:time_type:S1+situation_major:time_time_simple:S1+notification_type:time_time_simple:S1+weather_temperature:time_time_simple:S1+engagement:time_time_simple:S1+notification_type:situation_major:S1+weather_temperature:situation_major:S1+engagement:situation_major:S1+weather_temperature:notification_type:S1+engagement:notification_type:S1+engagement:weather_temperature:S1+time_time_simple:time_type:S2+situation_major:time_type:S2+notification_type:time_type:S2+weather_temperature:time_type:S2+engagement:time_type:S2+situation_major:time_time_simple:S2+notification_type:time_time_simple:S2+weather_temperature:time_time_simple:S2+engagement:time_time_simple:S2+notification_type:situation_major:S2+weather_temperature:situation_major:S2+engagement:situation_major:S2+weather_temperature:notification_type:S2+engagement:notification_type:S2+engagement:weather_temperature:S2"
}
formula_str = gsub("sent", "sent_centered", formula_str)

#Read in each dataset in folder to a list
data_list <- list.files(path = paste0('alg_output_', alg, '_', mech, '/results_', alg, '_', mech), pattern = "*.csv", full.names = TRUE) 
split <-word(word(data_list, -1, sep="/"), 1, sep="_") %>% as.numeric()
data_list <- data_list[order(split)]
seeds =  word(data_list, -1, sep = fixed('_'))
data_list <- data_list %>% map(read_csv, show_col_types = FALSE)

#Read in true coefficients for each scenario
if (mech == "extreme2") {
  betas0 = read_csv(paste0('data/true_beta_', mech, '_g336.csv'), show_col_types = FALSE)
} else if (mech == "extreme1"){
  betas0 = read_csv(paste0('data/true_beta_', mech, '_g838.csv'), show_col_types = FALSE)
  }

#for each dataset in data_list, set reference levels and make sure the reamaining levels are in the correct order
data_list <- data_list %>% map(~mutate(., time_type = relevel(fct(time_type), ref = "weekday"),
                                       time_time_simple = relevel(fct(time_time_simple), ref = "midday"),
                                       situation_major = relevel(fct(situation_major), ref = "travelling"),
                                       notification_type = relevel(fct(notification_type), ref = "grocery"),
                                       weather_temperature = relevel(fct(weather_temperature), ref = "cold")))

data_list <- data_list %>% map(~mutate(., time_type = factor(time_type, levels=c('weekday', 'weekend', 'unknown')),
                                       time_time_simple = factor(time_time_simple, levels=c('midday', 'morning', 'night', 'unknown')),
                                       situation_major = factor(situation_major, levels=c('travelling', 'unknown', 'working', 'leisure', 'shopping', 'morningrituals', 'workingout', 'housework', 'social')),
                                       weather_temperature = factor(weather_temperature, levels=c('cold', 'unknown', 'freezing', 'warm', 'hot'))))

# Calculate Regret --------------------------------------------------------

# Write function to calculate avg regret for each week for a single dataset
calc_regret <- function(df, true_betas){
  # Define new 'week' variable
  df <- df %>% mutate(week = floor(day/7))
  # Compute optimal decision every time a notification was sent using true betas
  # Use the glm function to create a design matrix
  df0 <- df %>% mutate(sent=0, sent_centered=sent-prob_sent)
  model0 <- glm(formula_str, data = df0,
               family = binomial(link = "logit"),
               control = list(maxit = 1, epsilon = 10))
  X0 <- model.matrix(model0)
  
  df1 <- df %>% mutate(sent=1, sent_centered=sent-prob_sent)
  model1 <- glm(formula_str, data = df1,
                family = binomial(link = "logit"),
                control = list(maxit = 1, epsilon = 10))
  X1 <- model.matrix(model1)

  #Compute linear predictors
  eta0 <- X0 %*% true_betas$beta
  eta0 <- as.numeric(eta0)
  eta1 <- X1 %*% true_betas$beta
  eta1 <- as.numeric(eta1)
  
  #Compute probability of app being opened under each action
  df$p0 <- exp(eta0)/(1+exp(eta0))
  df$p1 <- exp(eta1)/(1+exp(eta1))
  #Compare probabilities under different decisions to determine optimal action
  #Define observed action
  df <- df %>% mutate(A_star= ifelse(p0 > p1, 0, 1),
                      A_hat = sent)
  #Compute probability of response under optional and realized action
  df <- df %>% mutate(p_star = p0*(1-A_star) + p1*A_star,
                      p_minus_star = p0*(A_star) + p1*(1-A_star),
                      p_hat = p0*(1-A_hat) + p1*A_hat)
  #Compute regret for each decision- taking into account that the best we can do is send
    #notifications with probability .95
  df <- df %>% mutate(regret = (.95*p_star + .05*p_minus_star - p_hat))
  #save ouput of intermediate dataset
  write_csv(df, paste0('regret/', mech, '_', alg, '_debug_regret_', i, '.csv'))
  #Compute total regret for each week
  df <- df %>% group_by(week) %>% summarise(weekly_regret = sum(regret))
  #compute "cumulative regret" for each week
  df <- df %>% mutate(cumulative_regret = cumsum(weekly_regret)) 
  #df with week and cumulative_weekly_regret
  df <- df %>% select(week, cumulative_regret)
  return(df)
}


# Apply function to each dataset in data_list and save results to a dataframe where columns are week and cumulative_weekly_regret
regret_df <- data.frame(week = seq(0, floor(869/7), 1)) #869 is the last study day that notifications were sent
for (i in 1:length(data_list)){
  regret_df <- inner_join(regret_df, calc_regret(data_list[[i]], betas0), by = join_by(week==week)) 
  print(i)
}
colnames(regret_df) <- c('week', 1:length(data_list))
write_csv(regret_df, paste0('regret/', mech, '_', alg, '_all_regret_curves.csv'))

# Calculate average regret for each week across all datasets
avg_regret_df <- regret_df %>% mutate(avg_regret = rowMeans(select(., -week))) %>%
  select(week, avg_regret)

#Save results to csv file called 'avg_regret.csv" in a folder called regret
write_csv(avg_regret_df,  paste0('regret/', mech, '_', alg ,'_avg_regret.csv'))

  }
}


# Plotting Results --------------------------------------------------------

# Define consistent colors
alg_colors <- c("Simple" = "#1b9e77", "LS4L" = "#d95f02", "Complicated" = "#7570b3")

#Extreme1
LS4L_ex1 <- read_csv('regret/extreme1_LS4L_avg_regret.csv') %>% select(week, avg_regret) %>% rename(LS4L= avg_regret)
Simple_ex1 <- read_csv('regret/extreme1_simple_avg_regret.csv') %>% select(week, avg_regret) %>% rename(Simple = avg_regret)
Complicated_ex1 <- read_csv('regret/extreme1_complicated_avg_regret.csv') %>% select(week, avg_regret) %>% rename(Complicated = avg_regret)

complicated_spaghetti <- read_csv("regret/extreme1_complicated_all_regret_curves.csv") %>% 
  pivot_longer(cols = -week, names_to= 'sim', values_to = "avg_regret") %>%
  mutate(alg = "Complicated", type = "individual")

simple_spaghetti <- read_csv('regret/extreme1_simple_all_regret_curves.csv') %>%
  pivot_longer(cols = -week, names_to = 'sim', values_to = 'cumulative_regret') %>%
  group_by(week) %>%
  summarise(lwr_simple = quantile(cumulative_regret, .25), upr_simple = quantile(cumulative_regret, .75))

ls4l_spaghetti <- read_csv('regret/extreme1_LS4L_all_regret_curves.csv') %>%
  pivot_longer(cols = -week, names_to = 'sim', values_to = 'cumulative_regret') %>%
  group_by(week) %>% 
  summarise(lwr_ls4l = quantile(cumulative_regret, .25), upr_ls4l = quantile(cumulative_regret, .75))

bands <- full_join(ls4l_spaghetti, simple_spaghetti)

ex1 <- full_join(LS4L_ex1, Simple_ex1) %>% full_join(Complicated_ex1)  %>%
  pivot_longer(cols = c(-week), names_to = 'alg', values_to = 'avg_regret') %>% 
  mutate(type = "average", sim=NA)

ex1 <- bind_rows(ex1, complicated_spaghetti) %>% group_by(sim)

p1 <- ggplot() +
  #Lines for all algorithm averages
  geom_line(data = filter(ex1, type == "average"), aes(x = week, y = avg_regret, color = alg), linetype=1, linewidth = 1.2) +
  #Lines for each individual complicated simulation
  geom_line(data = filter(ex1, type == "individual"), aes(x=week, y = avg_regret, group=sim), alpha = .8, linetype = 2, color=alg_colors["Complicated"]) +
  # Ribbon for Simple
  geom_ribbon(data = bands,
              aes(x = week, ymin = lwr_simple, ymax = upr_simple), fill = alg_colors["Simple"],alpha = 0.15) +
  # Ribbon for LS4L
  geom_ribbon(data = bands,
              aes(x = week, ymin = lwr_ls4l, ymax = upr_ls4l),fill=alg_colors["LS4L"],alpha = 0.15) +
  #Formatting
  labs(title = 'No Contextual Moderators',
       x = 'Week', y = 'Cumulative Regret') +
  theme_minimal() + 
 theme(legend.position = "bottom") +  
  scale_color_manual(name = "Algorithm", values = alg_colors) +
  scale_fill_manual(values = alg_colors) +
  scale_x_continuous(limits = c(0,120), breaks = c(0, 20, 40, 60, 80, 100, 120)) +
  scale_y_continuous(limits = c(0,160), breaks =seq(0, 160, 20))
  
p1

ggsave('regret/extreme1_avg_regret.png', width = 6, height=6*.618)

#Extreme2
LS4L_ex2 <- read_csv('regret/extreme2_LS4L_avg_regret.csv') %>% rename(LS4L= avg_regret)
Simple_ex2 <- read_csv('regret/extreme2_simple_avg_regret.csv') %>% rename(Simple = avg_regret)
Complicated_ex2 <- read_csv('regret/extreme2_complicated_avg_regret.csv') %>% rename(Complicated= avg_regret)

simple_spaghetti <- read_csv('regret/extreme2_simple_all_regret_curves.csv') %>%
  pivot_longer(cols = -week, names_to = 'sim', values_to = 'cumulative_regret') %>%
  group_by(week) %>%
  summarise(lwr_simple = quantile(cumulative_regret, .25), upr_simple = quantile(cumulative_regret, .75))

ls4l_spaghetti <- read_csv('regret/extreme2_LS4L_all_regret_curves.csv') %>%
  pivot_longer(cols = -week, names_to = 'sim', values_to = 'cumulative_regret') %>%
  group_by(week) %>% 
  summarise(lwr_ls4l = quantile(cumulative_regret, .25), upr_ls4l = quantile(cumulative_regret, .75))

bands <- full_join(ls4l_spaghetti, simple_spaghetti)

complicated_spaghetti <- read_csv("regret/extreme2_complicated_all_regret_curves.csv") %>% 
  pivot_longer(cols = -week, names_to= 'sim', values_to = "avg_regret") %>%
  mutate(alg = "Complicated", type = "individual")

ex2 <- full_join(LS4L_ex2, Simple_ex2) %>% full_join(Complicated_ex2) %>%
  pivot_longer(cols = c(-week), names_to = 'alg', values_to = 'avg_regret') %>%
  mutate(type = "average", sim=NA)

ex2 <- bind_rows(ex2, complicated_spaghetti) 

p2 <- ggplot() +
  #Lines for all algorithm averages
  geom_line(data = filter(ex2, type == "average"), aes(x = week, y = avg_regret, color = alg), linetype=1, linewidth = 1.2) +
  #Lines for each individual complicated simulation
  geom_line(data = filter(ex2, type == "individual"), aes(x=week, y = avg_regret, group=sim), alpha = .8, linetype = 2, color=alg_colors["Complicated"]) +
  # Ribbon for Simple
  geom_ribbon(data = bands,
              aes(x = week, ymin = lwr_simple, ymax = upr_simple), fill = alg_colors["Simple"],alpha = 0.15) +
  # Ribbon for LS4L
  geom_ribbon(data = bands,
              aes(x = week, ymin = lwr_ls4l, ymax = upr_ls4l),fill=alg_colors["LS4L"],alpha = 0.15) +
  #Formatting
  labs(title = 'Many Contextual Moderators',
       x = 'Week', y = 'Cumulative Regret') +
  scale_color_manual(name = "Algorithm", values = alg_colors) +
  scale_fill_manual(values = alg_colors) +
  scale_x_continuous(limits = c(0,120), breaks = c(0, 20, 40, 60, 80, 100, 120)) +
  scale_y_continuous(limits = c(0,160), breaks = seq(0, 160, 20)) +
  theme_minimal() +
  theme(legend.position = "bottom")

p2

ggsave('regret/extreme2_avg_regret.png', width = 6, height=6*.618)

(p1 | p2) + plot_annotation(tag_levels = 'A') + 
  plot_layout(guides = "collect") & 
  theme(legend.position = "bottom") 
ggsave("regret/avg_regret_figure.png", width=8, height=8*.618)


# Sensitivity Analysis ----------------------------------------------------

#Extreme1
complicated_spaghetti <- read_csv("regret/extreme1_complicated_all_regret_curves.csv") %>% 
  pivot_longer(cols = -week, names_to= 'sim', values_to = "avg_regret") %>%
  mutate(alg = "Complicated", type = "individual")

simple_spaghetti <- read_csv('regret/extreme1_simple_all_regret_curves.csv') %>%
  pivot_longer(cols = -week, names_to = 'sim', values_to = 'cumulative_regret') %>%
  group_by(week) %>%
  summarise(lwr_simple = quantile(cumulative_regret, .25), upr_simple = quantile(cumulative_regret, .75))

ls4l_spaghetti <- read_csv('regret/extreme1_LS4L_all_regret_curves.csv') %>%
  pivot_longer(cols = -week, names_to = 'sim', values_to = 'cumulative_regret') %>%
  group_by(week) %>% 
  summarise(lwr_ls4l = quantile(cumulative_regret, .25), upr_ls4l = quantile(cumulative_regret, .75))

bands <- full_join(ls4l_spaghetti, simple_spaghetti)

ex1 <- full_join(LS4L_ex1, Simple_ex1) %>% full_join(Complicated_ex1)  %>%
  pivot_longer(cols = c(-week), names_to = 'alg', values_to = 'avg_regret') %>% 
  mutate(type = "average", sim=NA)

ex1 <- bind_rows(ex1, complicated_spaghetti) %>% group_by(sim)

p1 <- ggplot() +
  #Lines for all algorithm averages
  geom_line(data = filter(ex1, type == "average"), aes(x = week, y = avg_regret, color = alg), linetype=1, linewidth = 1.2) +
  #Lines for each individual complicated simulation
  geom_line(data = filter(ex1, type == "individual"), aes(x=week, y = avg_regret, group=sim), alpha = .8, linetype = 2, color=alg_colors["Complicated"]) +
  # Ribbon for Simple
  geom_ribbon(data = bands,
              aes(x = week, ymin = lwr_simple, ymax = upr_simple), fill = alg_colors["Simple"],alpha = 0.15) +
  # Ribbon for LS4L
  geom_ribbon(data = bands,
              aes(x = week, ymin = lwr_ls4l, ymax = upr_ls4l),fill=alg_colors["LS4L"],alpha = 0.15) +
  #Formatting
  labs(title = 'No Contextual Moderators',
       x = 'Week', y = 'Cumulative Regret') +
  theme_minimal() + 
  theme(legend.position = "bottom") +  
  scale_color_manual(name = "Algorithm", values = alg_colors) +
  scale_fill_manual(values = alg_colors) +
  scale_x_continuous(limits = c(0,120), breaks = c(0, 20, 40, 60, 80, 100, 120)) +
  scale_y_continuous(limits = c(0,160), breaks =seq(0, 160, 20))

p1

ggsave('regret/extreme1_avg_regret.png', width = 6, height=6*.618)




